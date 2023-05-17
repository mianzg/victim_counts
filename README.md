# Extracting Victim Counts from Text

This is the code repository for the EACL 2023 long paper [*Extracting Victim Counts from Text*](https://aclanthology.org/2023.eacl-main.141/) by Mian Zhong(@MianZhong), Shehzaad Dhuliawala(@shehzaadzd), and Niklas Stoehr(@niklas_stoehr).

🖼️ [Slides](https://drive.google.com/file/d/1_9NCyR47PUkaTZnqgE8_0BvYM3phMIpk/view?usp=share_link) 
## Installation and Repo Structure
We use python 3.8 and provide relevant packages in `requirements.txt`. A quick start would be:

```{python}
# Create a virtual environment
python -m venv counts
source counts/bin/activate

# Install requirement packages
python -m pip install -r requirements.txt
```

The following repository structure is used for this work:

```
|-- baseline/
|   |-- depmodel.py
|   |-- regexmodel.py
|   |-- run_dep.py
|   |-- run_regex.py
|   |-- run_srl.py
|   |-- srlmodel.py
|-- calibration/
|   |-- mcdropout.py
|   |-- post_hoc.py
|-- data/
|   |-- raw/
|   |-- processed/
|-- notebooks/
|   |-- baselines.ipynb
|   |-- data.ipynb
|   |-- domain_shift.ipynb
|   |-- fewshot.ipynb
|-- Dataset.py 
|-- inference.py
|-- models.py
|-- run_classification.py
|-- run_decoding.py
|-- run_generation.py
|-- run_regression.py
|-- requirements.txt
|-- utils.py
```

## Data
We use several public available datasets (e.g., [World Atrociti Dataset (WAD)](https://eventdata.parusanalytics.com/data.dir/atrocities.html)). After getting original data, put it under `data/raw` directory as illustrated with the WAD dataset in the repo structure.

The script `Dataset.py` provides some illustration how we process the datasets for the experiments. In general, you need to provide event descriptions. For training, you also need to provide the relevant victim counts as labels. You may refer to the class `WAD` serves as a running example. 

## Baseline Models
The baseline implementaions include regex (`regexmodel.py`), dependency parsing (`depmodel.py`), and semantic role labeling (`srlmodel.py`) which can be found inside `baseline` directory. 

To run on the data, we first process our datasets into an easy-to-use version and save them into the `data/processed` directory, you may refer to the notebook `notebook/data.ipynb` for more details. Once the processed data is ready, use `run_regex.py`, `run_dep.py`, and `run_srl.py` with the following template `python run_?.py --data ? --label ?` with an example shown in below:

```{python}
python run_regex.py --data wad --label death
```

## Fine-tuning NT5 Model for Different Task Formulations
### 1. Prepare Configuration File
We prepare a configuration file in json format. You may change the variables' value accordingly:

- `datasets`: The name used in `DATAMAP` in `utils.py`, e.g., "WAD".
- `question`: For training purpose, we used "How many people were killed?" for death counts or "How many people were injured?" for injury counts. You may use another relevant (and ethical) question.
- `target`: "death" or "injury"
- `training`: You may change the hyper-params with different values, such as `num_epochs`, `per_device_train_batch_size`

We usually name the file as `{model}_{data}_{label}_{task}.json`, for example, `nt5_wad_death_clf.json`. An example configuration is shown as follows

```
{"data":{
    "datadirec": "/path/to/data/",  //Change Me
    "datasets": ["WAD"],
    "question": "How many people were killed?",
    "target": "death",
    "include_zero": true,
    "num_classes": 3   //Only for classification task
 },
 "tokenizer":{
    "padding": "max_length",
    "cls_token": "<cls>"
 },
 "model":{
    "name": "nt5_clf",
    "pretrained_path": "nielsr/nt5-small-rc1",
    "path_to_model": null,
    "device": "'cuda' if torch.cuda.is_available() else 'cpu'"
 },
 "training":{
    "output_dir": "/path/to/experiment/result/dir/",  //Change Me
    "num_epochs": 100,
    "evaluation_strategy": "epoch",
    "path_to_save": "/path/to/save/model/",  //Change Me
    "dataloader_pin_memory": false,
    "logging_strategy": "epoch",
    "per_device_train_batch_size": 8,
    "save_total_limit":1
 }
}
```

### 2. Run Experiments
As argued in the paper, we conducted text classifiaction, regression, and generation task for victim counts extraction by fine-tuning on a numeracy-rich model NT5. The fine-tuning is preferably with GPU access. 

For classification task, you may fine-tune the model similar to the following command:

```
python classification/run_classification.py --config path/to/config_dir/nt5_wad_death_clf.json --do_train --do_eval --do_pred
```

For regression task, you may fine-tune the model similar to the following command:

```
python classification/run_regression.py --config path/to/config_dir/nt5_wad_death_reg.json --do_train --do_eval --do_pred
```
For text Generation task, we need two steps:

1. Fine-tune the model with
```
python finetune_nt5.py --config /path/to/config_dir/nt5_wad_death_gen.json --do_train --do_eval 
```
2. Decoding with the fine-tuned model. We need add the following block in the configuration file for the inference 

```
{
    ...
    "inference":{
        "num_only": true,
        "num_beams":5,
        "max_length":512,
        "is_return_dict":true,
        "is_out_scores":true,
        "use_prefixfn": true,
        "num_return_sequences": 5,
        "save_path": "/path/to/save/inference/results/"
    }
}
```

Then, we can decode the answers by

```
python generation/run_nt5_inference.py --config /path/to/config_dir/config.json
--ft_model /path/to/saved_models/model/
```

### 3. Calibration
For the above experiments, you may check what calibration methods are available and flag `--calib` when running experiments if calibration is wanted.
