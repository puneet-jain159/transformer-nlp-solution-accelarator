import numpy as np
from transformers import (
    TextClassificationPipeline,
    TokenClassificationPipeline,
)
import mlflow


class TransformerModel(mlflow.pyfunc.PythonModel):
    """
    Class to use HuggingFace Models
    """

    def __init__(
        self, tokenizer, model, max_token_length=None, task_name=None
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.max_token_length = max_token_length
        self.task_name = task_name
        self.pipe = self._define_pipeline()

    def predict(self, context, model_input):
        """
        Function to predict
        """
        return self.pipe(list(model_input["text"]))

    def _define_pipeline(self):
        """
        Function to define the pipeline
        """
        if self.task_name == "multi-class":
            self.pipe = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                padding="max_length",
                truncation=True,
                max_length=self.max_token_length,
                framework="pt",
            )

        elif self.task_name == "ner":
            self.pipe = TokenClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                framework="pt",
            )

        else:
            raise ValueError(
                "The task is not defined please define the task"
            )
