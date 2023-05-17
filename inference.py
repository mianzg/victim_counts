import numpy as np
import torch
from torch.utils.data import DataLoader
from text2digits import text2digits
from tqdm import tqdm
import sys
sys.path.append("./")
from calibration.post_hoc import softmax

# generate answer
def get_allowed_answer(tokenizer, num_only):
    """
    Helper function to only output digit token in decoding
    """
    num_ids = [tokenizer.convert_tokens_to_ids("{}".format(i)) for i in range(10)]
    allowed = num_ids + [tokenizer.eos_token_id]
    if num_only:
        allowed_answer = {tokenizer.pad_token_id:allowed, 
                num_ids[0]: allowed,
                num_ids[1]: allowed,
                num_ids[2]:allowed, 
                num_ids[3]:allowed, 
                num_ids[4]:allowed,
                num_ids[5]:allowed,
                num_ids[6]: allowed, 
                num_ids[7]: allowed, 
                num_ids[8]: allowed, 
                num_ids[9]: allowed
                }
    else:
        allowed_answer=None
    return allowed_answer

def generate_from_config(model, tokenizer, encoded_query, config):
    """
    Return
    hf.BeamSearchEncoderDecoderOutput
    """
    # INFERENCE PARAMS
    num_beams = config["inference"]["num_beams"]
    max_length = config["inference"]["max_length"]
    is_return_dict = config["inference"]["is_return_dict"]
    is_out_scores = config["inference"]["is_out_scores"]
    num_return_sequences = config["inference"]["num_return_sequences"]
    # TODO: assertion on numonly, badwords and prefix
    use_prefix = config["inference"]["num_only"] and config["inference"]["use_prefixfn"]
    allowed_answer = get_allowed_answer(tokenizer, use_prefix) #TODO:FIXME
    prefix_allowed_tokens_fn = (lambda bid, sent: allowed_answer[int(sent[-1])]) if use_prefix else None
    
    generated_answer = model.generate(input_ids=encoded_query["input_ids"], 
                                        attention_mask=encoded_query["attention_mask"], 
                                        max_length=max_length, 
                                        num_beams = num_beams,
                                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                        num_return_sequences=num_return_sequences,
                                        return_dict_in_generate=is_return_dict,
                                        output_scores=is_out_scores)
    return generated_answer

def counts_generate(query, model, tokenizer, config):
    """
    Answer generation
    """
    # Tokenizer Params
    padding = config["tokenizer"]["padding"]
    device = eval(config["model"]["device"])
    max_length = config["tokenizer"]["max_length"]

    encoded_query = tokenizer(
                    query, 
                    return_tensors='pt', 
                    padding=padding, 
                    truncation=True, 
                    max_length=max_length).to(device)
    
    generated_answer = generate_from_config(model, tokenizer, encoded_query, config)
    return generated_answer

def decode_num_only_one(answer_tokens):
    """Decode number-only on one sample"""
    answer = "".join(answer_tokens.split(" ")) #TODO
    if answer == "":
        return 0 # Not answerable?
    else:
        return int(answer)

def decode_num_only(tokenizer, answer_tokens):
    decoded_answers = tokenizer.batch_decode(answer_tokens)
    decoded_answers_int = [decode_num_only_one(a) for a in decoded_answers]
    return decoded_answers_int

def decode_no_restriction_one(tokenizer, answer_tokens, t2d):
    """Decode without number-only restriction on one sample"""
    num_token_mask = np.bool8(tokenizer.get_special_tokens_mask(answer_tokens, already_has_special_tokens=True))
    if num_token_mask.sum()==0: #No digit tokenn
        decoded_answer = t2d.convert(tokenizer.decode(answer_tokens)) # convert any text format number
    else:
        decoded_answer = "".join(tokenizer.convert_ids_to_tokens(answer_tokens[num_token_mask]))
    try:
        decoded_answer = int(decoded_answer)
    except ValueError:
        pass
    return decoded_answer

def decode_no_restriction(tokenizer, answer_tokens):
    t2d = text2digits.Text2Digits()
    num_token_masks = [np.bool8(tokenizer.get_special_tokens_mask(ans, already_has_special_tokens=True)) for ans in answer_tokens]
    # decoded_answers = tokenizer.batch_decode(answer_tokens)
    decoded_answers_int = [decode_no_restriction_one(tokenizer,a,t2d) for a in answer_tokens]
    return decoded_answers_int

def counts_decode(generated_answer, tokenizer, config):
    num_only=config["inference"]["num_only"]
     # TODO: assertion on numonly, badwords and prefix
    if type(generated_answer) is not torch.Tensor:
        answer_tokens = generated_answer['sequences'].cpu().numpy()
    else:
        answer_tokens = generated_answer.cpu().numpy()
    end_idx = np.where(answer_tokens==1)[1]
    answer_tokens = [answer[1:end_idx[i]] for i, answer in enumerate(answer_tokens)]

    if num_only: # Number token only
        decoded_answers = decode_num_only(tokenizer, answer_tokens)
    else:
        decoded_answers = decode_no_restriction(tokenizer, answer_tokens)
    return decoded_answers

def run_inference(dataset, model, tokenizer, config):
    """
    Make Inference over the dataset according to the config file parameters
    
    Return:
    result (dict): Return token id sequences, their beam scores, their decoded answers, and config
    """
    num_beams = config["inference"]["num_beams"]
    if num_beams > 1:
        is_beam_search = True
    else:
        is_beam_search = False
    # Get Dataloader
    dataloader = DataLoader(dataset, batch_size=1) #CHANGE ME
    #  Inference
    sequences = []    
    sequences_scores = []
    decoded_answers = []
    for batch_ndx, (queries,labels) in tqdm(enumerate(dataloader)):
        generated_answer = counts_generate(queries, model, tokenizer, config)
        decoded_answers.append(counts_decode(generated_answer, tokenizer, config))
        sequences.append(generated_answer.sequences.cpu().numpy())
        if is_beam_search:
            sequences_scores.append(generated_answer.sequences_scores.cpu().numpy())
        else:
            scores=np.array([softmax(i.cpu().numpy()).max(1) for i in generated_answer['scores']]).prod(0) # proba score 
            sequences_scores.append(scores[0])
    result = {"sequences": sequences,
              "sequences_scores": sequences_scores,
              "config": config,
              "decoded_answers": decoded_answers,
              "is_beam_search": is_beam_search,
    }
    return result
