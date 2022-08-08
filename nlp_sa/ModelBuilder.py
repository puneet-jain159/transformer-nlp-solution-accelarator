from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ModelBuilder:
    """
    Class for loading Models from Hugging Face including Tokenizer, Model and configuration file
    """

    def __init__(
            self,
            conf,
            Dataset=None):
        self.conf = conf
        self.Dataset = Dataset
        self.label_to_id = None
        self._loadModelConfig()
        self._loadTokenizer()
        self._loadModel()
        self.CorrectLabeltoId()

    def _loadModelConfig(self):
        # Load Config
        self.Dataset.get_num_class()

        self.config = AutoConfig.from_pretrained(
            self.conf.model_args.config_name if self.conf.model_args.config_name else self.conf.model_args.model_name_or_path,
            num_labels=self.Dataset.num_labels,
            finetuning_task=self.conf.data_args.task_name,
            cache_dir=self.conf.model_args.cache_dir,
            revision=self.conf.model_args.model_revision,
            use_auth_token=True if self.conf.model_args.use_auth_token else None)
    

    def _loadTokenizer(self):
        '''
        Function to load tokenizer by specifying location or path in the config
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.conf.model_args.tokenizer_name if self.conf.model_args.tokenizer_name else self.conf.model_args.model_name_or_path,
            cache_dir=self.conf.model_args.cache_dir,
            use_fast=self.conf.model_args.use_fast_tokenizer,
            revision=self.conf.model_args.model_revision,
            use_auth_token=True if self.conf.model_args.use_auth_token else None,
        )

        # Correct the sequence length incase of any mismatch
        if self.conf.data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}.")
            self.conf.max_seq_length = min(self.conf.training_args.max_seq_length, self.tokenizer.model_max_length)

    def _loadModel(self):
        '''
        Function to load the model architecture with the specific task name
        '''
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.conf.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.conf.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.conf.model_args.cache_dir,
            revision=self.conf.model_args.model_revision,
            use_auth_token=True if self.conf.model_args.use_auth_token else None,
            ignore_mismatched_sizes=self.conf.model_args.ignore_mismatched_sizes)
    


    def CorrectLabeltoId(self):
        '''
        Function to correct label_to_id
        '''
        if (self.Dataset is not None) and (self.conf.data_args.task_name is not None):
            non_label_column_names = [name for name in self.Dataset.train.column_names if name != "label"]
            sentence1_key, sentence2_key = non_label_column_names[0], None
            self.label_to_id = {v: i for i, v in enumerate(self.Dataset.label_list)}

        if self.label_to_id is not None:
            self.model.config.label2id = self.label_to_id
            self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}
        elif self.data_args.task_name is not None and not self.conf.model_args.is_regression:
            self.model.config.label2id = {l: i for i, l in enumerate(self.Dataset.label_list)}
            self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}