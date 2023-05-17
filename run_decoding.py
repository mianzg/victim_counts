import argparse
import sys
sys.path.append("./")
from calibration import post_hoc
from inference import run_inference
from utils import get_nt5_model, load_config, save_inference_result, get_inference_data


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--use_whole", action="store_true")
    parser.add_argument("--ft_model", default = None, help="Used for domain shift, path to finetuned LM model")
    parser.add_argument("--calib", default=None, help="Method of post-hoc calibration: temp_scale")
    parser.add_argument("--num_beams", default=None, type=int, help="Beam size")
    args = parser.parse_args()
    
    # Process configuration
    CONFIG_PATH = args.config
    config = load_config(CONFIG_PATH)
    device = eval(config['model']['device'])
    config["data"]["use_whole"] = args.use_whole
    config["training"] = {}
    if args.ft_model is not None:
        config["training"]["path_to_save"] = args.ft_model
        config["training"]["do_train"] = False
    else:
        config["training"]["do_train"] = True
    if args.num_beams is not None:
        config["inference"]["num_beams"] = args.num_beams
        if config["inference"]["num_beams"] != config["inference"]["num_return_sequences"]:
            config["inference"]["num_return_sequences"] = config["inference"]["num_beams"]
    # Inference
    modelname = config['model']['name']
    
    if "t5" in modelname:
        model,tokenizer = get_nt5_model(config)
    else:
        raise NotImplementedError
    model.to(device)
    model.eval()
    ds, labels = get_inference_data(config)
    inference_result = run_inference(ds, model, tokenizer, config)
    inference_result["labels"] = labels
    
    if args.calib is not None:
        if args.calib == "temp_scale":
            dev_set, dev_labels = get_inference_data(config, is_dev=True)
            dev_result = run_inference(dev_set, model, tokenizer, config)
            dev_result["labels"] = dev_labels
            cal_result = post_hoc.calibration_by_temp_scaling_lm(dev_result, None)
            best_sigma = cal_result["temperature"]
            cali_results = post_hoc.calibration_by_temp_scaling_lm(inference_result, best_sigma)
            inference_result.update(cali_results)
    save_inference_result(config, inference_result)
