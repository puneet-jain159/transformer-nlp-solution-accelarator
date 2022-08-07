
import yaml
import pathlib
from typing import Dict, Any, Tuple, List, Union
import pandas as pd
from omegaconf import OmegaConf
import logging

from transformers import (
    HfArgumentParser,
    TrainingArguments
)

from transformers.utils import logging
from utils import get_config, DataTrainingArguments, ModelArguments

logger = logging.get_logger(__name__)

class ConfLoader:
    def __init__(
            self,
            conf: Union[str, Dict[str, Any]] = "conf/model.yaml"):
        self.conf = self._load_conf_file(conf)
        self.model_args, self.data_args, self.training_args = self.load_HfFArgurements()

    def _load_conf_file(self, conf):
        '''
        Function which conf loader to load Dict or read yaml file from specified directory
        '''
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        elif isinstance(conf, str):
            logger.debug(f"Reading the configuration from YAML file in location :{conf}")
            _yaml_conf = yaml.safe_load(
                pathlib.Path(f"./{conf}").read_text()
            )
            conf = OmegaConf.create(_yaml_conf)
        return conf

    def load_HfFArgurements(self):
        '''
        Function to load the Hugging Face Arguements returns Model,Data,Training 
        Arguements
        '''
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments))
        return parser.parse_dict(self.conf)


if __name__ == '__main__':
    conf = ConfLoader()
    print(conf.conf)
