import sys
import os
import argparse
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
sys.path.append(os.path.abspath("."))
os.environ["WANDB_DISABLED"] = "true"
from torch.utils.data import Subset
import pickle
from torch.utils.data import Subset
from transformers import set_seed, Trainer, T5Tokenizer
from utils import load_config, get_reg_data, do_train, do_eval, get_trainingargs, get_iso_cal_table, get_norm_q, get_y_hat
from Dataset import *
from models import T5Classification
from calibration.mcdropout import MCDropout

def get_model(config):
    pretrained_path = config["model"]["pretrained_path"]    
    if "train_frac" in config["data"].keys() and float(config["data"]["train_frac"])==0.0: # zero-shot
        do_train = True
    else:
        do_train = config["training"]["do_train"]
    num_labels = 1

    if do_train:
        path_to_model = config["model"]["path_to_model"]
        model = T5Classification.from_pretrained(path_to_model, num_labels=num_labels) # e.g. NT5
    else:
        model = T5Classification.from_pretrained(config["training"]["path_to_save"],num_labels=num_labels) # Finetuned models
    tokenizer = T5Tokenizer.from_pretrained(pretrained_path, cls_token="<cls>", eos_token="</s>")
    return model, tokenizer
    
def eval_mse_r2_mape(inference, cat):
    y_true, y_pred = inference[f"{cat}_labels"], inference[f"{cat}_preds"]
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mse, r2, mape

def main(config):
    device = eval(config['model']['device'])
    # DATA
    train_set, dev_set, test_set = get_reg_data(config)
    if "train_frac" in config["data"].keys(): # Few-shot
        frac = config["data"]["train_frac"]
        n = range(int(len(train_set)*frac))
        train_set = Subset(train_set, n)
        config['training']["output_dir"] =  config['training']['output_dir']+"_" + str(config["data"]["train_frac"])
    # Model
    set_seed(0)
    model, tokenizer = get_model(config)    
    collator = QACollator(tokenizer, device)
    model.to(device)
    args = config['training']
    args['device']=device
    training_args = get_trainingargs(args)
    # TRAINER
    if not args['do_train']:
        model.eval()
    trainer = Trainer(model= model, 
                    args = training_args,
                    data_collator = collator,
                    train_dataset=train_set, 
                    eval_dataset=dev_set
                    )
    
    # Training
    if args["do_train"]:
        trainer = do_train(trainer, config)
    if args["do_eval"]:
        trainer = do_eval(trainer)
    if args["do_pred"]:
        dev_outs = trainer.predict(dev_set)
        test_outs = trainer.predict(test_set)
        mcdp = MCDropout(config, 50) #TODO: set 50 iterations 
        dev_inference, test_inference = mcdp.run()
        dev_inference["dev_preds"] = dev_outs.predictions.squeeze()
        test_inference["test_preds"] = test_outs.predictions.squeeze()
        # Regression evaluation metrics
        dev_mse, dev_r2, dev_mape = eval_mse_r2_mape(dev_inference, "dev")
        test_mse,  test_r2, test_mape = eval_mse_r2_mape(test_inference, "test")
        dev_inference['mse'], dev_inference['r2'], dev_inference['mape'] = dev_mse, dev_r2, dev_mape
        test_inference['mse'], test_inference['r2'], test_inference['mape'] = test_mse,  test_r2, test_mape
        # Save regression and mcdropout result
        if dev_inference is not None:
            with open(os.path.join(args["output_dir"], "dev_inference_50.pickle"), "wb") as f:
                pickle.dump(dev_inference, file=f)
        if test_inference is not None:
            with open(os.path.join(args["output_dir"], "test_inference_50.pickle"), "wb") as f:
                pickle.dump(test_inference, file=f)      
    else:
        try:
            with open(os.path.join(args["output_dir"], "dev_inference_50.pickle"), "rb") as f:
                dev_inference = pickle.load(f)
        except FileNotFoundError:
            print("Must flag --do_pred to evaluate on validation/test set first!")
        try:
            with open(os.path.join(args["output_dir"], "test_inference_50.pickle"), "rb") as f:
                test_inference = pickle.load(f)
        except FileNotFoundError:
            print("Must flag --do_pred to evaluate on validation/test set first!")
    # Calibration: Isotonic Regression
    calibration = config["calibration"]
    if calibration is not None: 
        truth_dev = np.array(dev_inference['dev_labels']).reshape((-1,1))
        mu_dev, sigma_dev = np.array(dev_inference["dev_mc_preds"]), np.sqrt(dev_inference["dev_mc_sigmas"])
        truth_test = np.array(test_inference['test_labels']).reshape(-1, 1)
        mu_test, sigma_test  = np.array(test_inference["test_mc_preds"]), np.sqrt(test_inference["test_mc_sigmas"])
        
        n_t_test = 4096
        t_list_test = np.linspace(np.min(mu_test) - np.max(sigma_test),
                            np.max(mu_test) + np.max(sigma_test),
                            n_t_test).reshape(1, -1)
        q_dev, q_hat_dev = get_iso_cal_table(truth_dev, mu_dev, sigma_dev)
        q_test, q_hat_test = get_iso_cal_table(truth_test, mu_test, sigma_test)
        if calibration == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(q_dev, q_hat_dev)
            # Calibrate on test set
            q_base, s_base = get_norm_q(mu_test.ravel(), sigma_test.ravel(), t_list_test.ravel())
            q_iso = calibrator.predict(q_base.ravel()).reshape(np.shape(q_base))
            s_iso = np.diff(q_iso, axis=1) / \
                    (t_list_test[0, 1:] - t_list_test[0, :-1]).ravel().reshape(1, -1).repeat(len(test_inference['test_labels']), axis=0)
            y_test_cal = get_y_hat(t_list_test.ravel(), s_iso)
            try:
                q_test_cal = np.array([np.nonzero(s_iso[i])[0][-1]/s_iso.shape[1] for i in range(s_iso.shape[0])])
            except IndexError:
                q_test_cal = []
                for i in range(s_iso.shape[0]):
                    nom = np.nonzero(s_iso[i])[0]
                    if len(nom) == 0:
                        q_test_cal.append(0) 
                    else:
                        q_test_cal.append(nom[-1]/s_iso.shape[1])
                q_test_cal = np.array(q_test_cal)
            _, q_test_hat_cal = get_iso_cal_table(None, None, None, q_raw=q_test_cal)
        test_inference["test_y_cal"] = y_test_cal
        test_inference["q_test"] =  q_test
        test_inference["q_hat_test"] =  q_hat_test
        test_inference["q_test_cal"] = q_test_cal
        test_inference["q_test_hat_cal"] = q_test_hat_cal
        with open(os.path.join(args["output_dir"], "test_inference_{}.pickle".format(calibration)), "wb") as f:
            pickle.dump(test_inference, file=f)
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="-c", default=None)
    parser.add_argument("--do_train", action="store_true", help="Flag to train regression models")
    parser.add_argument("--do_eval", action="store_true", help="Flag to evaluate model on development set")
    parser.add_argument("--do_pred", action="store_true", help="Flag to mc dropout inference on test set")
    parser.add_argument("--calib", default=None, help="Post-hoc calibration method: 'isotonic'")
    parser.add_argument("--train_frac", type=float)
    parser.add_argument("--ft_model", default = None, help="Used for domain shift, path to finetuned LM model")

    args = parser.parse_args()
    if args.config is None:
        raise ValueError("A configuration file is needed!")
    else:
        CONFIG_PATH = args.config
    config = load_config(CONFIG_PATH)
    config["training"]["do_train"] = args.do_train
    config["training"]["do_eval"] = args.do_eval
    config["training"]["do_pred"] = args.do_pred
    if not args.do_pred:
        args.sample_question = False
    if args.ft_model is not None:
        config["training"]["path_to_save"] = args.ft_model
        appendix = args.ft_model.split("/")[-2].split("_")
        config['training']["output_dir"] +=  "_by_{}_{}".format(appendix[1], appendix[2])
    if args.train_frac is not None:
        assert 0 <= args.train_frac and args.train_frac <= 1
        config["data"]["train_frac"] = args.train_frac
        config['training']['path_to_save'] += "_"+str(config['data']['train_frac'])
    if args.calib is not None and args.calib !=  "isotonic":
        raise ValueError("Cannot override config. Must be 'isotonic'")
    else:
        config['calibration'] = args.calib
    main(config)

