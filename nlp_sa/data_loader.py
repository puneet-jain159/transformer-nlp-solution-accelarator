
from datasets import load_dataset, load_metric
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DataLoader:
    '''
    Helper class to download the data to Fine-Tune and Train the Data
    '''

    def __init__(
            self,
            conf):
        self.conf = conf
        self._LoadDataSet()

    def _LoadDataSet(self):
        """
        Function to load data from Delta table or HuggingFace DataSet
        """

        if self.conf.data_args.dataset_name:
            logger.debug(
                f"Loading the dataset from hugging face dataset: {self.conf.data_args.dataset_name}")
            self.raw_dataset = load_dataset(
                self.conf.data_args.dataset_name,
                self.conf.data_args.dataset_config_name,
                cache_dir=self.conf.model_args.cache_dir,
                use_auth_token=True if self.conf.model_args.use_auth_token else None)
