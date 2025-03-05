# Visualising Policy-Reward Interplay to Inform Zeroth-Order Preference Optimisation of Large Language Models
This is the code relative to the paper `Visualising Policy-Reward Interplay to Inform Zeroth-Order Preference Optimisation of Large Language Models`.

Folder structure:
```sh
├── vis_zopro/
│   ├── visualize.py  # visualize and analyze given weights. Will project in 2D, 3D and do some analysis. Note this takes a lot of memory (recommended 120GB+) 
│   ├── dataset_utils.py  # utilities for *using* datasets
│   ├── extract_preference_pairs.py  # utilities pre-processing datasets such that they become a list of {"prompt": ..., "accepted": ..., "rejected": ...}
│   ├── reward_trainer_general.py  # Contains the code for a custom reward model trainer, mostly copied from trl's but adds support for MT datasets with score instead of pairs.
│   ├── trainers.py  # The main file used for training. Contains the code for iterative refinement and supports all the methods.
│   └── zopro.py  # Contains the code for our method. Uses the skeleton of RLOO from TRL.
├── requirements.txt  # python libraries needed
└── README.md  # this file
```

## Quick-start
To quickly start using Zeroth-Order Preference Optimisation, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/alessioGalatolo/VisZOPrO
    cd VisZOPrO
    ```

2. **Install the required Python libraries**:
    ```sh
    # create and activate your favourite virtual environment, then:
    pip install -r requirements.txt
    ```
Note, the requirements uses a personal version of TRL that fixes problems with DDP and LoRA support in some trainers: OnlineDPO (add/fix PEFT support), PPO (fix DDP), RLOO (add/fix PEFT). 

3. **Run the main training script**:
    ```sh
    python vis_zopro/trainers.py
    ```

The [`vis_zopro/trainers.py`](vis_zopro/trainers.py) file is used to train reward-policy models, given a pretrained model, *using LoRA* with various preference optimisation methods. To use it, run the script from the command line with the desired arguments. You can specify the task with `--task` (e.g., `'chat'`), choose a preference optimisation method using `--po-method`, options: `'ppo'`, `'dpo'`, `'rloo'`, `'zorloo'` (our own); and provide the model and dataset names via `--model-name` and `--dataset-name`. LoRA parameters, such as rank (`--lora-r`) and alpha (`--lora-alpha`), can be customised, along with other training settings like batch size (`--batch-size`), number of policy learning epochs (`--pl-epochs`), and learning rate (`--learning-rate`). To save training progress, specify `--log-dir` (required). Optional flags include `--intraepoch-refinement` to enable intra-epoch reward model refinement and `--keep-rm-close` to constrain reward model drift between iterations. The script also supports resuming from a previous checkpoint using `--resume`, however this only resumes from the end of the last iteration. For a complete list of arguments, run `python vis_zopro/trainers.py --help`.

For reproducibility: this project was run with Python 3.11.5 and CUDA 12.4.0.
## Acknowledgements
Part of this code was borrowed from [princeton-nlp/MeZO](https://github.com/princeton-nlp/MeZO), [eric-mitchell/direct-preference-optimization/](https://github.com/eric-mitchell/direct-preference-optimization/), [huggingface/transformers](https://github.com/huggingface/transformers) and [huggingface/trl](https://github.com/huggingface/trl).
