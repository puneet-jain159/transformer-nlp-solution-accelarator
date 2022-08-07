
from datasets import load_dataset, load_metric,Dataset
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DataLoader:
    '''
    Helper class to download the data to Fine-Tune and Train the Data
    '''

    def __init__(
            self,
            conf,
            spark = None):
        self.conf = conf
        self.spark = spark
        self._LoadDataSet()

    def _LoadDataSet(self):
        """
        Function to load data from Delta table or HuggingFace DataSet
        """

        if self.conf.data_args.dataset_name:
            logger.debug(
                f"Loading the dataset from hugging face dataset: {self.conf.data_args.dataset_name}")
            if self.conf.training_args.do_train:
                self.train = load_dataset(
                    self.conf.data_args.dataset_name,
                    self.conf.data_args.dataset_config_name,
                    split = "train",
                    cache_dir=self.conf.model_args.cache_dir,
                    use_auth_token=True if self.conf.model_args.use_auth_token else None)
            
            if self.conf.training_args.do_eval:
                self.test = load_dataset(
                    self.conf.data_args.dataset_name,
                    self.conf.data_args.dataset_config_name,
                    split = "test",
                    cache_dir=self.conf.model_args.cache_dir,
                    use_auth_token=True if self.conf.model_args.use_auth_token else None)

        elif self.conf.data_args.database_name:
            logger.debug(
                f"Loading the data from Database: {self.conf.data_args.database_name}")

            if self.conf.data_args.train_table:
                logger.debug(
                    f"Loading the train data from table: {self.conf.data_args.train_table}")
                train = self.spark.read.table(f"{self.conf.data_args.database_name}.{self.conf.data_args.train_table}")
                train = train.toPandas()
                self.train = Dataset.from_pandas(train)
           
            if self.conf.data_args.validation_table:
                logger.debug(
                    f"Loading the data validation table from: {self.conf.data_args.validation_table}")
                train = self.spark.read.table(f"{self.conf.data_args.database_name}.{self.conf.data_args.validation_table}")
                validation = train.toPandas()
                self.train = Dataset.from_pandas(validation)




