import shutil
import os
from functools import partial
import pandas as pd
import yaml

from transformers.trainer_utils import get_last_checkpoint
from datasets import load_metric
from transformers.utils import logging

from src.utils import add_args_from_dataclass
from src.evaluate import compute_metrics
from src.preprocess import preprocess_function

logger = logging.get_logger(__name__)
# logger = get_logger()


def detect_checkpoint(conf):
    # todo: Clarify why this is necessary
    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(conf.training_args.output_dir)
        and conf.training_args.do_train
        and not conf.training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(conf.training_args.output_dir)
        if (
            last_checkpoint is None
            and len(os.listdir(conf.training_args.output_dir)) > 0
        ):
            raise ValueError(
                f"Output directory ({conf.training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and conf.training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    elif conf.training_args.overwrite_output_dir:
        try:
            shutil.rmtree(conf.training_args.output_dir)
        except OSError as e:
            logger.warn(
                f"the log directory: {conf.training_args.output_dir} does not exist "
            )

    return last_checkpoint


def apply_preprocessing(conf, dataset, model):
    # todo: revisit to understand why to mutate dataset, why model builder is needed here
    preprocess = partial(preprocess_function, conf=conf, dataset=dataset, model=model)

    with conf.training_args.main_process_first(desc="dataset map train pre-processing"):
        dataset.train = dataset.train.map(
            preprocess,
            batched=True,
            # load_from_cache_file=not conf.args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    with conf.training_args.main_process_first(desc="dataset map test pre-processing"):
        dataset.test = dataset.test.map(
            preprocess,
            batched=True,
            # load_from_cache_file=not conf.args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )


def get_metric_callable(conf, data_loader):
    # Get the metric function
    if conf.training_args.metric_for_best_model is not None:
        metric = load_metric(conf.args.evaluate_metric)

    return partial(compute_metrics, conf=conf, metric=metric, dataset=data_loader)


def get_check_point(conf, last_checkpoint):
    if conf.training_args.do_train:
        if conf.training_args.resume_from_checkpoint is not None:
            return conf.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            return last_checkpoint
        else:
            return None


def combine_training_args(conf, dataset):
    conf.training_args.input_example = pd.DataFrame(dataset.train[:5])[
        conf.args.feature_col
    ].to_frame()
    add_args_from_dataclass(conf.training_args, conf.args)


def log_conf_as_yaml(conf):
    filename = os.path.join(conf.training_args.output_dir, "code/conf.yaml")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    conf.training_args.loc = filename
    print("filename:", filename)
    f = open(filename, "w+")
    yaml.safe_dump(conf.training_args.__dict__, f, allow_unicode=True, encoding="utf-8")
