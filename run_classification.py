"""
Fine-tune Classification Task
"""
import sys
import os
sys.path.append(os.path.abspath("."))
os.environ["WANDB_DISABLED"] = "true"
import pickle
import argparse
from torch.utils.data import Subset
from transformers import Trainer, T5Tokenizer, set_seed
from utils import do_eval, get_trainingargs, load_config, get_clf_data, do_train
from Dataset import *
from models import T5Classification
from calibration import post_hoc


def get_model(config, classes):
    MODELMAP = {
        "nt5_clf": T5Classification,
    }
    pretrained_path = config["model"]["pretrained_path"]
    modelname = config['model']['name']
    if "train_frac" in config["data"].keys() and config["data"]["train_frac"]==0:
        do_train = True
    else:
        do_train = config["training"]["do_train"]
    num_labels = len(classes)
    if do_train:
        path_to_model = config["model"]["path_to_model"]
        model = T5Classification.from_pretrained(path_to_model, num_labels=num_labels) # e.g. NT5
    else:
        model = MODELMAP[modelname].from_pretrained(config["training"]["path_to_save"], num_labels=num_labels)

    tokenizer = T5Tokenizer.from_pretrained(pretrained_path, cls_token="<cls>", eos_token="</s>")
    return model, tokenizer


def main(config):
    device = eval(config['model']['device'])
    # DATA
    train_set, dev_set, test_set, classes = get_clf_data(config)
    if "train_frac" in config["data"].keys(): #Few-shot 
        frac = config["data"]["train_frac"]
        n = range(int(len(train_set)*frac))
        train_set = Subset(train_set, n)
        config['training']["output_dir"] +=  "_" + str(config["data"]["train_frac"])
        config['training']["path_to_save"] +=  "_" + str(config["data"]["train_frac"])
    # Model
    set_seed(0)
    model,tokenizer = get_model(config, classes)
    collator = QACollator(tokenizer, device, is_int_label=True)
    model.to(device)
    # TRAINER
    args = config['training']
    args["device"] = device
    
    training_args = get_trainingargs(args)
    
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
        training_args.fp16 = False
        if type(classes[0]) is str:
            class_ids = tokenizer.convert_tokens_to_ids(classes)
        else:
            class_ids = None 
        calibration = config["calibration"]
        test_trainer = Trainer(model= model, 
                    args = training_args,
                    data_collator = collator,
                    eval_dataset=test_set
                    )
        # Calibration
        if calibration is not None:
            if calibration == "temp_scale":
                cal_dev = post_hoc.calibration_by_temp_scaling(trainer, class_ids, labels=classes)
                best_temperature = cal_dev["temperature"]
                cal_test = post_hoc.calibration_by_temp_scaling(test_trainer, class_ids, labels=classes, best_sigma=best_temperature)
            
            # Save calibration result
            if cal_dev is not None:
                with open(os.path.join(args["output_dir"], "cal_dev_{}.pickle".format(calibration)), "wb") as f:
                    pickle.dump(cal_dev, file=f)
            if cal_test is not None:
                with open(os.path.join(args["output_dir"], "cal_test_{}.pickle".format(calibration)), "wb") as f:
                    pickle.dump(cal_test, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="-c", default=None)
    parser.add_argument("--calib", default=None, help="Post-hoc calibration methods: 'temp_scale'")
    parser.add_argument("--do_train", action="store_true", help="Flag to train model")
    parser.add_argument("--do_eval", action="store_true", help="Flag for validation")
    parser.add_argument("--do_pred", action="store_true", help="Flag for evaluation on test set")
    parser.add_argument("--train_frac", type=float, help="Fraction of traning set, must be between [0, 1)")
    parser.add_argument("--ft_model", default = None, help="Used for domain shift, path to a finetuned LM model")

    args = parser.parse_args()
    if args.config is None:
        raise ValueError("A configuration file is needed!")
    else:
        CONFIG_PATH = args.config
    config = load_config(CONFIG_PATH)
    config["training"]["do_train"] = args.do_train
    config["training"]["do_eval"] = args.do_eval
    config["training"]["do_pred"] = args.do_pred
    if args.train_frac is not None:
        assert 0 <= args.train_frac and args.train_frac <= 1
        config["data"]["train_frac"] = args.train_frac
        config['training']['path_to_save'] += "_"+str(config['data']['train_frac'])
    if not args.do_pred:
        args.sample_question = False
    if args.ft_model is not None:
        config["training"]["path_to_save"] = args.ft_model
        appendix = args.ft_model.split("/")[-1].split("_")
        config['training']["output_dir"] +=  "_by_{}_{}".format(appendix[1], appendix[2])
    if args.calib not in ["temp_scale"]:
        raise ValueError("Cannot override config. Must be 'temp_scale'")
    else:
        config['calibration'] = args.calib
    main(config)
