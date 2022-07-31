%autoindent
import logging
import os
import random
import sys

import datasets
import numpy as np
import mlflow
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.integrations import MLflowCallback

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_utils import get_last_checkpoint

from utils import get_config,DataTrainingArguments,ModelArguments
from utils.callbacks import CustomMLflowCallback

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file("config.json")

logger = logging.getLogger("runner.log")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    filename= "runner.log",
    filemode='a')

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)

logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")


set_seed(training_args.seed)

padding = "max_length"

train= load_dataset(
    data_args.dataset_name,
    data_args.dataset_config_name,
    cache_dir=model_args.cache_dir,
    use_auth_token=True if model_args.use_auth_token else None,
    split = 'train'
)

test = load_dataset(
    data_args.dataset_name,
    data_args.dataset_config_name,
    cache_dir=model_args.cache_dir,
    use_auth_token=True if model_args.use_auth_token else None,
    split = 'test'
)

label_list = train.unique("label")
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)


config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
)

non_label_column_names = [name for name in train.column_names if name != "label"]
sentence1_key, sentence2_key = non_label_column_names[0], None
label_to_id = {v: i for i, v in enumerate(label_list)}

if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
elif data_args.task_name is not None and not is_regression:
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}

if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        examples[sentence1_key] if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    print("exmples :",len(examples))

    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    print("result :",len(result))
    return result


def tokenize(batch):
    """
    Tokenizer for non-streaming read. Additional features are available when using
    the map function of a dataset.Dataset instead of a dataset.IterableDataset, 
    therefore different tokenizer functions are used for each case.
    """
    return tokenizer(batch['text'], 
                     padding='max_length', 
                     truncation=True)
  
  # See the docs for mapping a function to a DatasetDict at
  # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.DatasetDict.map
train_set = datasets.Dataset.from_dict(tokenize(train))
train_set = train_set.add_column(name="label", column=train['label'])
train_set = train_set.add_column(name="text", column=train['text'])


test_set = datasets.Dataset.from_dict(tokenize(test))
test_set = test_set.add_column(name="label", column=test['label'])
test_set = test_set.add_column(name="text", column=test['text'])
# test = tokenize(test)

# with training_args.main_process_first(desc="dataset map pre-processing"):
#     raw_datasets['train'] = train.map(
#         preprocess_function,
#         batched=True,
#         # load_from_cache_file=not data_args.overwrite_cache,
#         desc="Running tokenizer on dataset",
#     )

#     raw_datasets['test'] = test.map(
#         preprocess_function,
#         batched=True,
#         # load_from_cache_file=not data_args.overwrite_cache,
#         desc="Running tokenizer on dataset",
    # )






if training_args.do_train:
    train_dataset = train_set
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

# if training_args.do_eval:
#     if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
#         raise ValueError("--do_eval requires a validation dataset")
#     eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "test"]
#     if data_args.max_eval_samples is not None:
#         max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
#         eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Get the metric function
if data_args.task_name is not None:
    metric = load_metric("glue", data_args.task_name)
else:
    metric = load_metric("accuracy")

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    if data_args.task_name is not None:
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    elif is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
# we already did the padding.
if data_args.pad_to_max_length:
    data_collator = default_data_collator
elif training_args.fp16:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
else:
    data_collator = None

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
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
is_regression = False
last_checkpoint = get_last_checkpoint(training_args.output_dir)
training_args.max_token_length = 128
    # Training
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=None)
if training_args.do_eval:
    trainer.evaluate()


if training_args.do_eval:
    logger.info("*** Evaluate ***")
    trainer.evaluate()

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(raw_datasets["validation_mismatched"])
        combined = {}

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)