import os
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from logging import Logger

import dotenv
from pyspark.sql import SparkSession

from nlp_sa import ConfLoader


class Workload(ABC):
    """
    This is an abstract class that provides handy interfaces to implement workloads (e.g. pipelines or job tasks).
    Create a child from this class and implement the abstract launch method.
    Class provides access to the following useful objects:
    * self.spark is a SparkSession
    * self.logger provides access to the Spark-compatible logger
    * self.conf provides access to the parsed configuration of the job
    * self.env_vars provides access to the parsed environment variables of the job
    """
    def __init__(self, spark=None):
        self.spark = self._prepare_spark(spark)
        self.logger = self._prepare_logger()
        # todo: Adapt config loading after omegaconf refactoring
        self.conf = ConfLoader(self._get_conf_file())
        self.env_vars = self.get_env_vars_as_dict()
        self._log_env_vars()

    @staticmethod
    def _prepare_spark(spark) -> SparkSession:
        if not spark:
            return SparkSession.builder.getOrCreate()
        else:
            return spark

    @staticmethod
    def _get_base_env():
        p = ArgumentParser()
        p.add_argument('--base-env', required=False, type=str)
        namespace = p.parse_known_args(sys.argv[1:])[0]
        return namespace.base_env

    @staticmethod
    def _get_conf_file():
        p = ArgumentParser()
        p.add_argument('--conf-file', required=False, type=str)
        namespace = p.parse_known_args(sys.argv[1:])[0]
        return namespace.conf_file

    @staticmethod
    def _set_environ(env_vars):
        dotenv.load_dotenv(env_vars)

    def get_env_vars_as_dict(self):
        base_env = self._get_base_env()
        self._set_environ(base_env)
        return dict(os.environ)

    def _prepare_logger(self) -> Logger:
        log4j_logger = self.spark._jvm.org.apache.log4j  # noqa
        return log4j_logger.LogManager.getLogger(self.__class__.__name__)

    def _log_env_vars(self):
        # log parameters
        self.logger.info('Using environment variables:')
        for key, item in self.env_vars.items():
            self.logger.info('\t Parameter: %-30s with value => %-30s' % (key, item))

    @abstractmethod
    def launch(self):
        """
        Main method of the job.
        :return:
        """
        pass
