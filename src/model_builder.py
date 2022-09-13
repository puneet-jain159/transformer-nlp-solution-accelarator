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

    def __init__(
        self,
        dataset,
        conf=None,
        label_col=None,
        use_auth_token=None,
        ignore_mismatched_sizes=True,
        task_name=None,
        cache_dir=None,
        model_revision=None,
        model_name_or_path=None,
        tokenizer_name_or_path=None,
        config_name_or_path=None,
        use_fast_tokenizer=None,
        max_seq_length=None,
    ):
        self.conf = conf
        self._dataset = dataset
        self._label_col = self.conf.args.label_col if label_col is None else label_col
        self._label_to_id = None
        self._use_auth_token = (
            self.conf.args.use_auth_token if use_auth_token is None else use_auth_token
        )
        self._ignore_mismatched_sizes = (
            self.conf.args.ignore_mismatched_sizes
            if ignore_mismatched_sizes is None
            else ignore_mismatched_sizes
        )
        self._model_name_or_path = (
            self.conf.args.model_name_or_path
            if model_name_or_path is None
            else model_name_or_path
        )

        self._config_name_or_path = (
            self.conf.args.model_name_or_path
            if config_name_or_path is None
            else model_name_or_path
        )
        self._task_name = self.conf.args.task_name if task_name is None else task_name
        self._cache_dir = self.conf.args.cache_dir if cache_dir is None else cache_dir
        self._model_revision = (
            self.conf.args.model_revision if model_revision is None else model_revision
        )
        self._use_fast_tokenizer = (
            self.conf.args.use_fast_tokenizer
            if use_fast_tokenizer is None
            else use_fast_tokenizer
        )
        self._max_seq_length = (
            self.conf.args.max_seq_length if max_seq_length is None else max_seq_length
        )
        self._tokenizer_name_or_path = (
            self.conf.args.model_name_or_path
            if tokenizer_name_or_path is None
            else tokenizer_name_or_path
        )

        self._load_model_config()
        self._load_tokenizer()
        self._load_model()
        self.correct_label_to_id()

    def _load_model_config(self):
        # Load Config
        self._dataset.get_num_class()

        # Load Config
        self._config = AutoConfig.from_pretrained(
            self._config_name_or_path,
            num_labels=self._dataset._num_labels,
            finetuning_task=self._dataset._task_name,
            cache_dir=self._cache_dir,
            revision=self._model_revision,
            use_auth_token=self._model_revision,
        )

    def _load_tokenizer(self):
        """
        Function to load tokenizer by specifying location or path in the config
        """
        if self._config.model_type in ["gpt2", "roberta"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name_or_path,
                cache_dir=self._cache_dir,
                use_fast=self._use_fast_tokenizer,
                add_prefix_space=True,
                revision=self._model_revision,
                use_auth_token=self._use_auth_token,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name_or_path,
                cache_dir=self._cache_dir,
                use_fast=self._use_fast_tokenizer,
                revision=self._model_revision,
                use_auth_token=self._use_auth_token,
            )

        # Correct the sequence length incase of any mismatch
        if self._max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"""The max_seq_length passed
                    ({self._max_seq_length})
                    is larger than the maximum length for the model
                    ({self.tokenizer.model_max_length}).
                    Using max_seq_length={self.tokenizer.model_max_length}."""
            )
            self._max_seq_length = min(
                self._max_seq_length,
                self.tokenizer.model_max_length,
            )

    def _load_model(self):
        """
        Function to load the model architecture with the specific task name
        """
        if self._dataset._task_name == "multi-class":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name_or_path,
                from_tf=bool(".ckpt" in self._model_name_or_path),
                config=self._config,
                cache_dir=self._cache_dir,
                revision=self._model_revision,
                use_auth_token=True if self._use_auth_token else None,
                ignore_mismatched_sizes=self._ignore_mismatched_sizes,
            )
        elif self._dataset._task_name == "ner":
            self.model = AutoModelForTokenClassification.from_pretrained(
                self._model_name_or_path,
                from_tf=bool(".ckpt" in self._model_name_or_path),
                config=self._config,
                cache_dir=self._cache_dir,
                revision=self._model_revision,
                use_auth_token=True if self._use_auth_token else None,
                ignore_mismatched_sizes=self._ignore_mismatched_sizes,
            )
        else:
            raise ValueError("Model not created for the particular task")

    def correct_label_to_id(self):
        """
        Function to correct label_to_id
        """
        if self._task_name is not None:
            if self._task_name == "multi-class":
                self._correct_multiclass_label()
            elif self._task_name == "ner":
                self._correct_ner_label()

    def _correct_multiclass_label(self):
        """
        Function to correct the label2id for multiclass
        """

        if (self._dataset is not None) and (self._task_name is not None):

            self.label_to_id = {v: i for i, v in enumerate(self._dataset._label_list)}

        if self.label_to_id is not None:
            self._label2id = self.label_to_id
            self._id2label = {id: label for label, id in self._config.label2id.items()}
        elif self._task_name is not None and not self._is_regression:
            self.model._config.label2id = {
                l: i for i, l in enumerate(self._dataset._label_list)
            }
            self.model._config.id2label = {
                id: label for label, id in self._config.label2id.items()
            }

    def _correct_ner_label(self):
        """
        Function to correct the label2id for NER
        """
        # Model has labels -> use them.
        if (
            self.model.config.label2id
            != PretrainedConfig(num_labels=self._dataset._num_labels).label2id
        ):
            if list(sorted(self.model.config.label2id.keys())) == list(
                sorted(self._dataset._label_list)
            ):
                # Reorganize `_label_list` to match the ordering of the model.
                labels_are_int = isinstance(
                    self._dataset.train.features[self._label_col].feature,
                    int,
                )
                if labels_are_int:
                    self._dataset.label_to_id = {
                        i: int(self.model.config.label2id[l])
                        for i, l in enumerate(self._dataset._label_list)
                    }
                    self.__label_list = [
                        self.model.config.id2label[i]
                        for i in range(self._dataset._num_labels)
                    ]
                else:
                    self._dataset._label_list = [
                        self.model.config.id2label[i]
                        for i in range(self._dataset._num_labels)
                    ]
                    self._label_to_id = {
                        l: i for i, l in enumerate(self._dataset._label_list)
                    }
            else:
                logger.warning(
                    f"""Your model seems to have been trained with labels,
                    but they don't match the dataset; model labels:
                    {list(sorted(self.model.config.label2id.keys()))},
                    dataset labels:
                    {list(sorted(self._dataset._label_list))}.\n
                    Ignoring the model labels as a result.""",
                )

        # Set the correspondences label/ID inside the model sconfig
        self.model.config.label2id = {
            l: i for i, l in enumerate(self._dataset._label_list)
        }
        self.model.config.id2label = {
            i: l for i, l in enumerate(self._dataset._label_list)
        }
