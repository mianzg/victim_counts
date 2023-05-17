"""
Fine-tune Text Generation Task
"""
import sys
import os
from torch.utils.data import Subset
sys.path.append(os.path.abspath("."))
os.environ["WANDB_DISABLED"] = "true"
import argparse
from transformers import Trainer, set_seed
from utils import get_lm_data, get_nt5_model, get_trainingargs, do_train, do_eval, load_config
from Dataset import QACollator, QADataset


def main(config):
    device = eval(config['model']['device'])
    # DATA
    train_set, dev_set, test_set = get_lm_data(config)
    if "train_frac" in config["data"].keys():
        n = range(int(len(train_set)*config["data"]["train_frac"]))
        train_set = Subset(train_set, n)
    # Model
    set_seed(0)
    modelname = config['model']['name']
    if "t5" in modelname:
        model,tokenizer = get_nt5_model(config)
    else:
        raise NotImplementedError
    collator = QACollator(tokenizer, device, is_str_label=True)
    model.to(device)
    # TRAINER
    args = config['training']
    args['device'] = device 
    if "train_frac" in config["data"].keys():
        args["output_dir"] +=  "_" + str(config["data"]["train_frac"])
    training_args = get_trainingargs(args)#TODO:
    if "t5" in modelname:
        trainer = Trainer(model= model, 
                        args = training_args,
                        data_collator = collator,
                        train_dataset=train_set, 
                        eval_dataset=dev_set
                        )
    if args["do_train"]:
        trainer = do_train(trainer, config)
    if args["do_eval"]:
        trainer = do_eval(trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="-c", default=None, help="Path to the training configuration file")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--train_frac", type=float)
    args = parser.parse_args()
    if args.config is None:
        raise ValueError("A configuration file is needed!")
    else:
        CONFIG_PATH = args.config
    config = load_config(CONFIG_PATH)
    config["training"]["do_train"] = args.do_train
    config["training"]["do_eval"] = args.do_eval
    if args.train_frac is not None:
        assert 0 < args.train_frac and args.train_frac <= 1
        config["data"]["train_frac"] = args.train_frac
        config['training']['path_to_save'] += "_"+str(config['data']['train_frac'])
    main(config)


