from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ModelBuilder:
    """
    Class for loading Models from Hugging Face including Tokenizer,
    Model and configuration file
    """

    def __init__(self, conf, Dataset=None):
        self.conf = conf
        self.Dataset = Dataset
        self.label_to_id = None
        self._load_model_config()
        self._load_tokenizer()
        self._load_model()

    def _load_model_config(self):
        # Load Config
        self.Dataset.get_num_class()

        self.config = AutoConfig.from_pretrained(
            self.conf.model_args.config_name
            if self.conf.model_args.config_name
            else self.conf.model_args.model_name_or_path,
            num_labels=self.Dataset.num_labels,
            finetuning_task=self.conf.data_args.task_name,
            cache_dir=self.conf.model_args.cache_dir,
            revision=self.conf.model_args.model_revision,
            use_auth_token=True
            if self.conf.model_args.use_auth_token
            else None,
        )

    def _load_tokenizer(self):
        """
        Function to load tokenizer by specifying location or path in the config
        """
        if self.config.model_type in {"gpt2", "roberta"}:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.conf.model_args.tokenizer_name
                if self.conf.model_args.tokenizer_name
                else self.conf.model_args.model_name_or_path,
                cache_dir=self.conf.model_args.cache_dir,
                use_fast=self.conf.model_args.use_fast_tokenizer,
                revision=self.conf.model_args.model_revision,
                use_auth_token=True
                if self.conf.model_args.use_auth_token
                else None,
                add_prefix_space=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.conf.model_args.tokenizer_name
                if self.conf.model_args.tokenizer_name
                else self.conf.model_args.model_name_or_path,
                cache_dir=self.conf.model_args.cache_dir,
                use_fast=self.conf.model_args.use_fast_tokenizer,
                revision=self.conf.model_args.model_revision,
                use_auth_token=True
                if self.conf.model_args.use_auth_token
                else None,
            )

        # Correct the sequence length incase of any mismatch
        if (
            self.conf.data_args.max_seq_length
            > self.tokenizer.model_max_length
        ):
            logger.warning(
                f"""The max_seq_length passed
                    ({self.conf.data_args.max_seq_length})
                    is larger than the maximum length for the model
                    ({self.tokenizer.model_max_length}).
                    Using max_seq_length={self.tokenizer.model_max_length}."""
            )
            self.conf.max_seq_length = min(
                self.conf.training_args.max_seq_length,
                self.tokenizer.model_max_length,
            )

    def _load_model(self):
        """
        Function to load the model architecture with the specific task name
        """

        if self.conf.data_args.task_name == "multi-class":
            from_tf = bool(
                ".ckpt" in self.conf.model_args.model_name_or_path
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.conf.model_args.model_name_or_path,
                from_tf=from_tf,
                config=self.config,
                cache_dir=self.conf.model_args.cache_dir,
                revision=self.conf.model_args.model_revision,
                use_auth_token=self.conf.model_args.use_auth_token,
                ignore_mismatched_sizes=self.conf.model_args.ignore_mismatched_sizes,
            )
        elif self.conf.data_args.task_name == "ner":
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.conf.model_args.model_name_or_path,
                from_tf=bool(
                    ".ckpt" in self.conf.model_args.model_name_or_path
                ),
                config=self.config,
                cache_dir=self.conf.model_args.cache_dir,
                revision=self.conf.model_args.model_revision,
                use_auth_token = self.conf.model_args.use_auth_token,
                ignore_mismatched_sizes=self.conf.model_args.ignore_mismatched_sizes,
            )
        else:
            raise ValueError(
                "Model not created for the particular task"
            )

    def correct_label_to_id(self):
        """
        Function to correct label_to_id
        """
        if self.conf.data_args.task_name is not None:
            if self.conf.data_args.task_name == "multi-class":
                self._correct_multiclass_label()
            elif self.conf.data_args.task_name == "ner":
                self._correct_ner_label()

    def _correct_multiclass_label(self):
        """
        Function to correct the label2id for multiclass
        """

        if (self.Dataset is not None) and (
            self.conf.data_args.task_name is not None
        ):

            self.label_to_id = {
                v: i for i, v in enumerate(self.Dataset.label_list)
            }

        if self.label_to_id is not None:
            self.model.config.label2id = self.label_to_id
            self.model.config.id2label = {
                id: label
                for label, id in self.config.label2id.items()
            }
        elif (
            self.data_args.task_name is not None
            and not self.conf.model_args.is_regression
        ):
            self.model.config.label2id = {
                l: i for i, l in enumerate(self.Dataset.label_list)
            }
            self.model.config.id2label = {
                id: label
                for label, id in self.config.label2id.items()
            }

    def _correct_ner_label(self):
        """
        Function to correct the label2id for NER
        """
        # Model has labels -> use them.
        if (
            self.model.config.label2id
            != PretrainedConfig(
                num_labels=self.Dataset.num_labels
            ).label2id
        ):
            if list(
                sorted(self.model.config.label2id.keys())
            ) == list(sorted(self.Dataset.label_list)):
                # Reorganize `label_list` to match the ordering of the model.
                labels_are_int = isinstance(
                    self.Dataset.train.features[
                        self.conf.data_args.label_col
                    ].feature,
                    int,
                )
                if labels_are_int:
                    self.Dataset.label_to_id = {
                        i: int(self.model.config.label2id[l])
                        for i, l in enumerate(self.Dataset.label_list)
                    }
                    self.label_list = [
                        self.model.config.id2label[i]
                        for i in range(self.Dataset.num_labels)
                    ]
                else:
                    self.Dataset.label_list = [
                        self.model.config.id2label[i]
                        for i in range(self.Dataset.num_labels)
                    ]
                    self.label_to_id = {
                        l: i
                        for i, l in enumerate(self.Dataset.label_list)
                    }
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(self.model.config.label2id.keys()))}, dataset labels:"
                    f" {list(sorted(self.Dataset.label_list))}.\nIgnoring the model labels as a result.",
                )

        # Set the correspondences label/ID inside the model config
        self.model.config.label2id = {
            l: i for i, l in enumerate(self.Dataset.label_list)
        }
        self.model.config.id2label = {
            i: l for i, l in enumerate(self.Dataset.label_list)
        }
