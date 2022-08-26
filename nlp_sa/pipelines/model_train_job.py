import os

import yaml
from transformers import (
    Trainer,
    set_seed)
from transformers.integrations import MLflowCallback

import mlflow
from nlp_sa.ModelBuilder import ModelBuilder
from nlp_sa.data_loader import DataLoader
from nlp_sa.utils.callbacks import CustomMLflowCallback
from nlp_sa.utils.logger_utils import get_logger
from nlp_sa.utils.train_utils import apply_preprocessing, detect_checkpoint, get_metric_callable, get_check_point, \
    combine_training_args, log_conf_as_yaml
from nlp_sa.workload import Workload

logger = get_logger()

yaml.SafeDumper.yaml_representers[None] = lambda self, data: \
    yaml.representer.SafeRepresenter.represent_str(
        self,
        str(data),
    )


class ModelTrainJob(Workload):

    def launch(self):
        logger.info('ModelTrainJob job started!')
        log_level = self.conf.training_args.log_level
        logger.setLevel(log_level)

        set_seed(self.conf.training_args.seed)
        data_loader = DataLoader(self.conf, self.spark)
        model_builder = ModelBuilder(self.conf, data_loader)

        apply_preprocessing(self.conf, data_loader, model_builder)
        last_checkpoint = detect_checkpoint(self.conf)

        # combine the arguements for trainig
        combine_training_args(self.conf, data_loader)

        # log the conf as conf.yaml

        log_conf_as_yaml(self.conf)

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        with mlflow.start_run():
            trainer = Trainer(
                model=model_builder.model,
                args=self.conf.training_args,
                train_dataset=data_loader.train if self.conf.training_args.do_train else None,
                eval_dataset=data_loader.test if self.conf.training_args.do_eval else None,
                compute_metrics=get_metric_callable(self.conf, data_loader),
                tokenizer=model_builder.tokenizer
            )

            trainer.remove_callback(MLflowCallback)
            trainer.add_callback(CustomMLflowCallback)

            checkpoint = get_check_point(self.conf, last_checkpoint)
            trainer.train(resume_from_checkpoint=checkpoint)

            logger.info('ModelTrainJob job finished!')

            # Evaluation
            if self.conf.training_args.do_eval:
                logger.info("*** Evaluate ***")
                trainer.evaluate(eval_dataset=data_loader.test)


if __name__ == '__main__':
    job = ModelTrainJob()
    job.launch()
