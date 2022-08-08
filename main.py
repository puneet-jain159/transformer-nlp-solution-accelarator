
%autoindent

import numpy as np
import datasets
import sys
import random
import os
import logging

from functools import partial
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    default_data_collator,
    set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import MLflowCallback
from datasets import load_metric, list_metrics

from nlp_sa.preprocess import preprocess_function
from nlp_sa.ModelBuilder import ModelBuilder
from nlp_sa.data_loader import DataLoader
from nlp_sa import ConfLoader
from nlp_sa.utils.callbacks import CustomMLflowCallback
from nlp_sa.evaluate import compute_metrics

logger = logging.getLogger("runner.log")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    filename="runner.log",
    filemode='a')

conf = ConfLoader()

log_level = conf.training_args.log_level
logger.setLevel(log_level)

set_seed(conf.training_args.seed)

DataSet = DataLoader(conf)
Model = ModelBuilder(conf, DataSet)


# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(conf.training_args.output_dir) and conf.training_args.do_train and not conf.training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(conf.training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(conf.training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({conf.training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and conf.training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

preprocess = partial(preprocess_function,conf= conf, Dataset= DataSet,Model= Model)


with conf.training_args.main_process_first( desc="dataset map train pre-processing"):
    DataSet.train = DataSet.train.map(
        preprocess,
        batched=True,
        # load_from_cache_file=not conf.data_args.overwrite_cache,
        desc="Running tokenizer on test dataset")


with conf.training_args.main_process_first(desc="dataset map test pre-processing"):
    DataSet.test = DataSet.test.map(
        preprocess,
        batched=True,
        # load_from_cache_file=not conf.data_args.overwrite_cache,
        desc="Running tokenizer on dataset")

# Get the metric function
if conf.training_args.metric_for_best_model is not None:
    metric = load_metric(conf.training_args.metric_for_best_model)


# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
# we already did the padding.
if conf.data_args.pad_to_max_length:
    data_collator = default_data_collator
elif  conf.training_args.fp16:
    data_collator = DataCollatorWithPadding(Model.tokenizer, pad_to_multiple_of=8)
else:
    data_collator = None


compute_m = partial(compute_metrics,conf = conf,metric = metric)

# Initialize our Trainer
trainer = Trainer(
    model=Model.model,
    args=conf.training_args,
    train_dataset=DataSet.train if conf.training_args.do_train else None,
    eval_dataset=DataSet.test if conf.training_args.do_eval else None,
    compute_metrics=compute_m,
    tokenizer=Model.tokenizer
)

os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = 'True'
os.environ["MLFLOW_EXPERIMENT_NAME"] = 'banking_nlp_classifier'
os.environ["MLFLOW_TAGS"] = '{"runner" : "puneet" ,"model":"albert-base-v2"}'
os.environ["CREATE_MFLOW_MODEL"] = 'True'
os.environ["MLFLOW_TRACKING_URI"] = 'sqlite:///mlflow.db'
os.environ["MLFLOW_NESTED_RUN"] = 'True'
os.environ["TOKENIZERS_PARALLELISM"] = 'True'

trainer.remove_callback(MLflowCallback)
trainer.add_callback(CustomMLflowCallback)

conf.training_args.max_token_length = conf.data_args.max_seq_length

# Training
if conf.training_args.do_train:
    if conf.training_args.resume_from_checkpoint is not None:
        checkpoint = conf.training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

# Evaluation
if conf.training_args.do_eval:
    logger.info("*** Evaluate ***")
    trainer.evaluate(eval_dataset=DataSet.test)
