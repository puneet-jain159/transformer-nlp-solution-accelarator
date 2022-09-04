import logging

from datasets import load_dataset, Dataset, ClassLabel
from .utils import get_label_list

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Helper class to download the data to Fine-Tune and Train the Data
    """

    def __init__(
        self,
        dataset_name,
        dataset_config_name = None,
        cache_dir = "/tmp/",
        use_auth_token = None,
        max_train_samples = None,
        max_eval_samples = None,
        database_name = None,
        do_eval = True,
        do_train = True,
        train_split = "train",
        eval_split = "test",
        train_table = None,
        validation_table = None,
        spark = None
    ):
        self._dataset_name = dataset_name
        self._dataset_config_name = dataset_config_name
        self._num_labels = None
        self._label_list = None
        self._cache_dir = cache_dir
        self._use_auth_token = use_auth_token
        self._max_train_samples = max_train_samples
        self._max_eval_samples = max_eval_samples
        self._database_name = database_name
        self._do_eval = do_eval
        self._do_train = do_train
        self._train_split = train_split
        self._eval_split = eval_split
        self._train_table = train_table
        self._validation_table = validation_table
        self._spark = spark

        if ((self._database_name and not self._spark) or 
        (self._spark and not self._database_name)):
            raise ValueError(
                """Invalid arguments - make sure to pass a Spark session
                object if you want to read data from a Hive Table using Spark"""
            )

        self._load_dataset()

    def _get_label_col(self):

        feature_set = None
        if self._do_train:
            feature_set = self.train.features
        elif self._do_eval:
            feature_set = self.test.features

        for feature_id in feature_set.keys():
            if isinstance(
                feature_set[feature_id],
                ClassLabel
            ):
                return feature_id

    def _load_dataset(self):
        """
        Function to load data from Delta table or HuggingFace DataSet
        """

        if self._dataset_name:
            logger.debug(
                f"""Loading the dataset from hugging face dataset:
                {self._dataset_name}"""
            )
            if self._do_train:
                self.train = load_dataset(
                    path = self._dataset_name,
                    download_config = self._dataset_config_name,
                    split=self._train_split,
                    cache_dir=self._cache_dir,
                    use_auth_token=self._use_auth_token
                )

            if self._max_train_samples is not None:
                max_train_samples = min(
                    len(self.train),
                    self._max_train_samples,
                )
                self.train = self.train.select(
                    range(max_train_samples)
                )

            if self._do_eval:
                self.test = load_dataset(
                    path = self._dataset_name,
                    download_config = self._dataset_config_name,
                    split=self._eval_split,
                    cache_dir=self._cache_dir,
                    use_auth_token=self._use_auth_token
                )

            if self._max_eval_samples is not None:
                max_eval_samples = min(
                    len(self.test),
                    self._max_eval_samples,
                )
                self.test = self.test.select(range(max_eval_samples))

            self._label_col = self._get_label_col()


        elif self._database_name:
            logger.debug(
                f"""Loading the data from Database:
                {self._database_name}"""
            )

            if self._train_table:
                logger.debug(
                    f"""Loading the train data from table:
                    {self._train_table}"""
                )
                train = self.spark.read.table(
                    f"""{self._database_name}
                    .{self._train_table}"""
                )
                train = train.toPandas()
                self.train = Dataset.from_pandas(train)

            if self._validation_table:
                logger.debug(
                    f"""Loading the data validation table from:
                    {self._validation_table}"""
                )
                test = self.spark.read.table(
                    f"""{self._database_name}.
                    {self._validation_table}"""
                )
                test = test.toPandas()
                self.test = Dataset.from_pandas(test)

    def get_num_class(self):
        """
        Based on the task type find the number_of_classes in the arguement
        """
        if self.conf.data_args.task_name is not None:
            is_regression = self.conf.data_args.task_name == "stsb"
            # ToDo change the label_col to label_names
            if not is_regression:
                if self._task_name == "multi-class":
                    self._num_labels = (
                        self._get_num_class_multi_class()
                    )
                elif self.conf.data_args.task_name == "ner":
                    self._num_labels = self._get_num_class_ner()
            else:
                self.num_labels = 1
        else:
            is_regression = self.train.features[
                self.conf.data_args.label_col
            ].names.dtype in ["float32", "float64"]
            if is_regression:
                self.num_labels = 1
            else:
                self.label_list = self.train.unique(
                    self.conf.data_args.label_col
                )
                self.label_list.sort()  # Let's sort it for determinism
                self.num_labels = len(self.label_list)

    def _get_num_class_multi_class(self):
        """
        Function to get num of classes for mult-class
        """

        self._label_list = self.train.unique(
            self._label_col
        )
        # Let's sort it for determinism
        self._label_list.sort()
        return len(self._label_list)

    def _get_num_class_ner(self):
        """
        Function to get num of classes when task name is ner
        """

        """If the labels are of type ClassLabel, they are already integers
        and we have the map stored somewhere. Otherwise, we have to get
        the list of labels manually."""

        labels_are_int = isinstance(
            self.train.features[
                self._label_col
            ].feature,
            ClassLabel,
        )
        if labels_are_int:
            self._label_list = self.train.features[
                self._label_col
            ].feature.names
            self._label_to_id = {
                i: i for i in range(len(self._label_list))
            }
        else:
            self._label_list = get_label_list(
                self.train[self._label_col]
            )
            self._label_to_id = {
                l: i for i, l in enumerate(self._label_list)
            }

        return len(self._label_list)
