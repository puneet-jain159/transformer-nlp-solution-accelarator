import logging
from multiprocessing.sharedctypes import Value

from datasets import load_dataset, Dataset, ClassLabel
from src.utils import get_label_list
import inspect

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Helper class to download the data to Fine-Tune and Train the Data
    """

    def __init__(
        self,
        dataset_name=None,
        dataset_config_name=None,
        cache_dir=None,
        use_auth_token=None,
        max_train_samples=None,
        max_eval_samples=None,
        database_name=None,
        do_eval=None,
        do_train=None,
        train_split=None,
        eval_split=None,
        train_table=None,
        validation_table=None,
        task_name=None,
        label_col=None,
        spark=None,
        conf=None,
    ):
        self.conf = conf
        self._dataset_name = (
            self.conf.args.dataset_name if dataset_name is None else dataset_name
        )
        self._task_name = self.conf.args.task_name if task_name is None else task_name
        self._dataset_config_name = (
            self.conf.args.dataset_config_name
            if dataset_config_name is None
            else dataset_config_name
        )
        self._cache_dir = self.conf.args.cache_dir if cache_dir is None else cache_dir
        self._use_auth_token = (
            self.conf.args.use_auth_token if use_auth_token is None else use_auth_token
        )
        self._max_train_samples = (
            self.conf.args.max_train_samples
            if max_train_samples is None
            else max_train_samples
        )
        self._max_eval_samples = (
            self.conf.args.max_train_samples
            if max_eval_samples is None
            else max_eval_samples
        )
        self._database_name = (
            self.conf.args.database_name if database_name is None else database_name
        )
        self._do_eval = self.conf.training_args.do_eval if do_eval is None else do_eval
        self._do_train = (
            self.conf.training_args.do_train if do_train is None else do_train
        )
        self._train_split = (
            self.conf.args.train_split if train_split is None else train_split
        )
        self._eval_split = (
            self.conf.args.eval_split if eval_split is None else eval_split
        )
        self._train_table = (
            self.conf.args.train_table if train_table is None else train_table
        )
        self._validation_table = (
            self.conf.args.validation_table
            if validation_table is None
            else validation_table
        )
        self._label_col = self.conf.args.label_col if label_col is None else label_col
        self._spark = spark

        self._num_labels = None
        self._label_list = None

        # check if spark context passed
        if self._database_name and not self._spark:
            raise ValueError(
                """Invalid arguments - make sure to pass a Spark session
                object if you want to read data from a Hive Table using Spark"""
            )

        self._load_dataset()

    def _load_attribute_from_config(self):
        """
        Load parameter from config
        """

        args = inspect.getfullargspec(DataLoader.__dict__["__init__"])[0]
        for arg in args:
            if not getattr(self, arg, None):
                print(arg)
                print("config", getattr(self.config.args, arg, None))
                if not getattr(self.config.args, arg, None):
                    setattr(self, arg, getattr(self.config.args, arg, None))
                elif not getattr(self.config.args, arg, None):
                    setattr(self, arg, getattr(self.config.args, arg, None))
                elif not getattr(self.config.training_args, arg, None):
                    setattr(self, arg, getattr(self.config.training_args, arg, None))
                else:
                    if arg not in ["config", "self"]:
                        logger.warning(f"{arg} not defined in config or method")

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
                    path=self._dataset_name,
                    download_config=self._dataset_config_name,
                    split=self._train_split,
                    cache_dir=self._cache_dir,
                    use_auth_token=self._use_auth_token,
                )

            if self._max_train_samples is not None:
                max_train_samples = min(
                    len(self.train),
                    self._max_train_samples,
                )
                self.train = self.train.select(range(max_train_samples))

            if self._do_eval:
                self.test = load_dataset(
                    path=self._dataset_name,
                    download_config=self._dataset_config_name,
                    split=self._eval_split,
                    cache_dir=self._cache_dir,
                    use_auth_token=self._use_auth_token,
                )

            if self._max_eval_samples is not None:
                max_eval_samples = min(
                    len(self.test),
                    self._max_eval_samples,
                )
                self.test = self.test.select(range(max_eval_samples))

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
        if self._task_name is not None:
            is_regression = self._task_name == "stsb"
            # ToDo change the label_col to label_names
            if not is_regression:
                if self._task_name == "multi-class":
                    self._num_labels = self._get_num_class_multi_class()
                elif self._task_name == "ner":
                    self._num_labels = self._get_num_class_ner()
            else:
                self._num_labels = 1
        else:
            is_regression = self.train.features[self._label_col].names.dtype in [
                "float32",
                "float64",
            ]
            if is_regression:
                self._num_labels = 1
            else:
                self.label_list = self.train.unique(self._label_col)
                self.label_list.sort()  # Let's sort it for determinism
                self._num_labels = len(self.label_list)

    def _get_num_class_multi_class(self):
        """
        Function to get num of classes for mult-class
        """

        self._label_list = self.train.unique(self._label_col)
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
            self.train.features[self._label_col].feature,
            ClassLabel,
        )
        if labels_are_int:
            self._label_list = self.train.features[self._label_col].feature.names
            self._label_to_id = {i: i for i in range(len(self._label_list))}
        else:
            self._label_list = get_label_list(self.train[self._label_col])
            self._label_to_id = {l: i for i, l in enumerate(self._label_list)}

        return len(self._label_list)
