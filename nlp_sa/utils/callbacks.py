from transformers.trainer_callback import TrainerCallback
import functools
import importlib.util
import json
import numbers
import os
import sys
import tempfile
from pathlib import Path
from .custom_func import TransformerModel

from transformers.utils import flatten_dict,logging, ENV_VARS_TRUE_VALUES, is_torch_tpu_available


logger = logging.get_logger(__name__)


def is_mlflow_available():
    if os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return importlib.util.find_spec("mlflow") is not None


class CustomMLflowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting
    environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.
    """

    def __init__(self):
        if not is_mlflow_available():
            raise RuntimeError(
                "MLflowCallback requires mlflow to be installed. Run `pip install mlflow`.")
        import mlflow

        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._auto_end_run = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.
        Environment:
            HF_MLFLOW_LOG_ARTIFACTS (`str`, *optional*):
                Whether to use MLflow .log_artifact() facility to log artifacts. This only makes sense if logging to a
                remote server, e.g. s3 or GCS. If set to `True` or *1*, will copy whatever is in
                [`TrainingArguments`]'s `output_dir` to the local or remote artifact storage. Using it without a remote
                storage will just copy the files to your artifact location.
            MLFLOW_EXPERIMENT_NAME (`str`, *optional*):
                Whether to use an MLflow experiment_name under which to launch the run. Default to "None" which will
                point to the "Default" experiment in MLflow. Otherwise, it is a case sensitive name of the experiment
                to be activated. If an experiment with this name does not exist, a new experiment with this name is
                created.
            MLFLOW_TAGS (`str`, *optional*):
                A string dump of a dictionary of key/value pair to be added to the MLflow run as tags. Example:
                os.environ['MLFLOW_TAGS']='{"release.candidate": "RC1", "release.version": "2.2.0"}'
            MLFLOW_NESTED_RUN (`str`, *optional*):
                Whether to use MLflow nested runs. If set to `True` or *1*, will create a nested run inside the current
                run.
            MLFLOW_RUN_ID (`str`, *optional*):
                Allow to reattach to an existing run which can be usefull when resuming training from a checkpoint.
                When MLFLOW_RUN_ID environment variable is set, start_run attempts to resume a run with the specified
                run ID and other parameters are ignored.
            MLFLOW_FLATTEN_PARAMS (`str`, *optional*):
                Whether to flatten the parameters dictionary before logging. Default to `False`.
        """
        self._log_artifacts = os.getenv(
            "HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._nested_run = os.getenv(
            "MLFLOW_NESTED_RUN", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", None)
        self._flatten_params = os.getenv(
            "MLFLOW_FLATTEN_PARAMS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._create_model = os.getenv(
            "CREATE_MFLOW_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._run_id = os.getenv("MLFLOW_RUN_ID", None)
        self._parent_run_id = None
        self._tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
        if self._tracking_uri:
            self._ml_flow.set_tracking_uri(self._tracking_uri)
        logger.debug("setting_tracking_uri")
        logger.debug(
            f"MLflow experiment_name={self._experiment_name}, run_name={args.run_name}, nested={self._nested_run},"
            f" tags={self._nested_run}"
        )
        if state.is_world_process_zero:
            if self._ml_flow.active_run() is None or self._nested_run or self._run_id:
                if self._experiment_name:
                    # Use of set_experiment() ensure that Experiment is created if not exists
                    self._ml_flow.set_experiment(self._experiment_name)
                self._ml_flow.start_run(
                    run_name=args.run_name, nested=self._nested_run)
                logger.info(
                    f"MLflow run started with run_id={self._ml_flow.active_run().info.run_id}")
                if self._parent_run_id is None:
                    self._parent_run_id = self._ml_flow.active_run().info.run_id
                    logger.warning(f"setting parent run id {str(self._parent_run_id)}")
                self._auto_end_run = True
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            combined_dict = flatten_dict(
                combined_dict) if self._flatten_params else combined_dict
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f'trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow\'s'
                        " log_param() only accepts values no longer than 250 characters so we dropped this attribute."
                        " You can use `MLFLOW_FLATTEN_PARAMS` environment variable to flatten the parameters and"
                        " avoid this message."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_flow.log_params(
                    dict(combined_dict_items[i: i + self._MAX_PARAMS_TAGS_PER_BATCH]))
            mlflow_tags = os.getenv("MLFLOW_TAGS", None)
            if mlflow_tags:
                mlflow_tags = json.loads(mlflow_tags)
                self._ml_flow.set_tags(mlflow_tags)
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not self._initialized:
            logger.warning("logger started run")
            self.setup(args, state, model)
        if state.is_world_process_zero:
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
                else:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                        "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                    )
            self._ml_flow.log_metrics(metrics=metrics, step=state.global_step)

    def on_epoch_begin(self, args, state, control, model=None, tokenizer=None, train_dataloader=None, **kwargs):

        if (self._auto_end_run and self._ml_flow.active_run()
                and self._parent_run_id != self._ml_flow.active_run().info.run_id):
            logger.warning("terminate run")
            self._ml_flow.end_run()

        if state.is_world_process_zero:
            self._ml_flow.start_run(
                run_name=args.run_name, nested=self._nested_run)
            logger.debug(
                f"Epoch started MLflow run started with run_id={self._ml_flow.active_run().info.run_id}")
            self._auto_end_run = True
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            combined_dict = flatten_dict(
                combined_dict) if self._flatten_params else combined_dict
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f'trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow\'s'
                        " log_param() only accepts values no longer than 250 characters so we dropped this attribute."
                        " You can use `MLFLOW_FLATTEN_PARAMS` environment variable to flatten the parameters and"
                        " avoid this message."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_flow.log_params(
                    dict(combined_dict_items[i: i + self._MAX_PARAMS_TAGS_PER_BATCH]))
            mlflow_tags = os.getenv("MLFLOW_TAGS", None)
            if mlflow_tags:
                mlflow_tags = json.loads(mlflow_tags)
                self._ml_flow.set_tags(mlflow_tags)

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, train_dataloader=None, **kwargs):

        if self._create_model:
            logger.debug("Creating Custom Pyfunc Model")
            cpu_model = model.to('cpu')
            transformer_model = TransformerModel(tokenizer=tokenizer,
                                                 model=cpu_model,
                                                 max_token_length=args.max_token_length)

            # Create conda environment
            with open('requirements.txt', 'r') as additional_requirements:
                libraries = additional_requirements.readlines()
                libraries = [library.rstrip() for library in libraries]

            model_env = self._ml_flow.pyfunc.get_default_conda_env()
            model_env['dependencies'][-1]['pip'] += libraries

            input_example = train_dataloader.dataset.data[:5].to_pandas()

            self._ml_flow.pyfunc.log_model("mlflow_model",
                                           python_model=transformer_model,
                                           conda_env=model_env,
                                           input_example=input_example)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, train_dataloader=None, **kwargs):

        if (self._auto_end_run and self._ml_flow.active_run()
                and (self._parent_run_id != self._ml_flow.active_run().info.run_id)):
            logger.debug("terminating child run")
            self._ml_flow.end_run()

        if self._initialized and state.is_world_process_zero:
            if self._log_artifacts:
                logger.info("Logging artifacts. This may take time.")
                self._ml_flow.log_artifacts(args.output_dir)
                logger.info("Logging Tokenizer")
                tokenizer.save_pretrained(args.output_dir)

            if self._create_model:
                logger.info("Creating Custom Pyfunc Model")
                cpu_model = model.to('cpu')
                transformer_model = TransformerModel(tokenizer=tokenizer,
                                                     model=cpu_model,
                                                     max_token_length=args.max_token_length)

                # Create conda environment
                with open('requirements.txt', 'r') as additional_requirements:
                    libraries = additional_requirements.readlines()
                    libraries = [library.rstrip() for library in libraries]

                model_env = self._ml_flow.pyfunc.get_default_conda_env()
                model_env['dependencies'][-1]['pip'] += libraries

                input_example = train_dataloader.dataset.data[:5].to_pandas()

                self._ml_flow.pyfunc.log_model("mlflow_model",
                                               python_model=transformer_model,
                                               conda_env=model_env,
                                               input_example=input_example)


    def on_evaluate(self, args, state, control, **kwargs):
        """
        Event called after an evaluation phase.
        """
        logger.debug("Evaluate has been called")
        if (self._auto_end_run and self._ml_flow.active_run()
            and (self._parent_run_id == self._ml_flow.active_run().info.run_id)):
            logger.debug("terminating parent run")
            self._ml_flow.end_run()

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            self._auto_end_run
            and callable(getattr(self._ml_flow, "active_run", None))
            and self._ml_flow.active_run() is not None
        ):
            self._ml_flow.end_run()