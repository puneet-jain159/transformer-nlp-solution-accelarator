
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
        self.num_labels = None
        self.label_list = None
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
                
            if self.conf.data_args.max_train_samples is not None:
                max_train_samples = min(len(self.traint), self.conf.data_args.max_train_samples)
                self.train = self.train.select(range(max_train_samples))
            
            if self.conf.training_args.do_eval:
                self.test = load_dataset(
                    self.conf.data_args.dataset_name,
                    self.conf.data_args.dataset_config_name,
                    split = "test",
                    cache_dir=self.conf.model_args.cache_dir,
                    use_auth_token=True if self.conf.model_args.use_auth_token else None)

            if self.conf.data_args.max_eval_samples is not None:
                max_eval_samples = min(len(self.test),  self.conf.data_args.max_eval_samples)
                self.test = self.test.select(range(max_eval_samples))

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
    
    def get_num_class(self):
        '''
        Based on the task type find the number_of_classes in the arguement
        '''
        if self.conf.data_args.task_name is not None:
            is_regression = self.conf.data_args.task_name == "stsb"
            #ToDo change the label_col to label_names
            if not is_regression:
                self.label_list = self.train.features[self.conf.data_args.label_col].names
                self.num_labels = len(self.label_list)
            else:
                self.num_labels = 1
        else:
            is_regression =  self.train.features[self.conf.data_args.label_col].names.dtype in ["float32", "float64"]
            if is_regression:
                self.num_labels = 1
            else:
                self.label_list = self.train.unique(self.conf.data_args.label_col)
                self.label_list.sort()  # Let's sort it for determinism
                self.num_labels = len(self.label_list)



