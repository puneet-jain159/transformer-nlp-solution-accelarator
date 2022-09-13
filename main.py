%autoindent

import os
import yaml
import logging

from src.utils.train_utils import detect_checkpoint, apply_preprocessing,\
                                  get_metric_callable, combine_training_args, \
                                  log_conf_as_yaml, get_check_point
from src.utils.callbacks import CustomMLflowCallback
from src.utils.config import ConfLoader
from src.data_loader import DataLoader
from src.model_builder import ModelBuilder
from transformers.integrations import MLflowCallback
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    default_data_collator,
    set_seed)


yaml.SafeDumper.yaml_representers[None] = lambda self, data: \
    yaml.representer.SafeRepresenter.represent_str(
        self,
        str(data),
    )


logger = logging.getLogger("runner.log")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    filename="runner.log",
    filemode='a')

conf = ConfLoader('conf/model_ner.yaml')

log_level = conf.training_args.log_level
logger.setLevel(log_level)

set_seed(conf.training_args.seed)

DataSet = DataLoader(conf= conf)
Model = ModelBuilder(conf= conf, dataset = DataSet)


# Detecting last checkpoint.
last_checkpoint = detect_checkpoint(conf)

# Apply preprocessing
apply_preprocessing(conf, DataSet, Model)

# Get the metric function
compute_m = get_metric_callable(conf, DataSet)


# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
# we already did the padding.
if conf.args.pad_to_max_length:
    data_collator = default_data_collator
elif conf.training_args.fp16:
    data_collator = DataCollatorWithPadding(
        Model.tokenizer, pad_to_multiple_of=8)
else:
    data_collator = None


# combine the arguements for trainig
combine_training_args(conf, DataSet)

# log the conf as conf.yaml

log_conf_as_yaml(conf)


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
os.environ["MLFLOW_EXPERIMENT_NAME"] = conf.args.experiment_location
os.environ["MLFLOW_TAGS"] = '{"runner" : "puneet" ,"model":"bert_base_uncased","task_name" : "ner"}'
os.environ["CREATE_MFLOW_MODEL"] = 'True'
os.environ["MLFLOW_TRACKING_URI"] = 'sqlite:///mlflow.db'
os.environ["MLFLOW_NESTED_RUN"] = 'True'
os.environ["TOKENIZERS_PARALLELISM"] = 'True'

trainer.remove_callback(MLflowCallback)
trainer.add_callback(CustomMLflowCallback)



# Training
if conf.training_args.do_train:
    checkpoint = get_check_point(conf, last_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

# Evaluation
if conf.training_args.do_eval:
    logger.info("*** Evaluate ***")
    trainer.evaluate(eval_dataset=DataSet.test)

