import mlflow
import os
import yaml
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _get_root_uri_and_artifact_path
from mlflow.utils.model_utils import _get_flavor_configuration, _add_code_from_conf_to_system_path

from transformers import AutoModelForSequenceClassification,AutoModelForTokenClassification

class TransformerModel(mlflow.pyfunc.PythonModel):
    """
      Class to use HuggingFace Models
    """

    def __init__(self, model=None, tokenizer=None,conf = None):
        self.conf = conf
        self.tokenizer = tokenizer
        self.model = model
        if model is None:
            raise ValueError("model not initialized")
        self._define_pipeline()

    def predict(self, model_input):
        '''
        Function to predict 
        '''
        print(list(model_input['text']))
        return self.pipe(list(model_input['text']))

    def _define_pipeline(self):
        '''
        Function to define the pipeline
        '''
        from transformers import TextClassificationPipeline, TokenClassificationPipeline
        if self.conf['task_name']== 'multi-class':
            self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer,
                                                   padding='max_length', truncation=True, max_length=self.conf['max_seq_length'], framework="pt")

        elif self.conf['task_name'] == 'ner':
            self.pipe = TokenClassificationPipeline(
                model=self.model, tokenizer=self.tokenizer, framework="pt")

        else:
            raise ValueError("The task is not defined please define the task")


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.
    :param path: Local filesystem path to the MLflow Model with the ``xgboost`` flavor.
    """
    model, tokenizer,conf = load_model_tokenizer(path)
    return TransformerModel(model=model, tokenizer=tokenizer,conf= conf)


def load_model_tokenizer(model_uri, dst_path=None):
    """
    Load an HF model tokenizer and conf from a local file or a run.
    :param model_uri: The location, in URI format, of the MLflow model. For example:
                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.
    """
    from transformers import AutoTokenizer,AutoModel
    import os
    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path)
    local_model_path = "/".join(local_model_path.split("/")[:-2])
    flavor_conf = _get_flavor_configuration(
        local_model_path, "python_function")
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    #  Load the conf from the model
    code_path = os.path.join(local_model_path, flavor_conf.get('code'))
    conf = None
    for file in os.listdir(code_path):
        if "yaml" in file:
            with open(os.path.join(code_path,file), 'r') as stream:
                conf = yaml.unsafe_load(stream)
    
    if conf is None:
        raise ValueError("Conf file used to train the model the not found please log it in the artifact")

    model = get_model_from_configuration(conf,os.path.join(local_model_path, flavor_conf.get('data')))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(local_model_path, flavor_conf.get('data')))
    return model, tokenizer,conf


def get_model_from_configuration(conf,path):
    '''
    Function to load the model based on the task_name specified
    '''
    if conf['task_name'] == 'multi-class':
        return AutoModelForSequenceClassification.from_pretrained(path)
    elif conf['task_name'] == 'ner':
        return AutoModelForTokenClassification.from_pretrained(path)
    else:
        raise ValueError(f"task_name {conf['task_name']} not supported yet ")
