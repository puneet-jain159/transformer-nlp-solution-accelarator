import numpy as np
from transformers import TextClassificationPipeline
import mlflow

     
class TransformerModel(mlflow.pyfunc.PythonModel):
  """
    Class to use HuggingFace Models
  """
  def __init__(self, tokenizer, model, max_token_length):
    self.tokenizer = tokenizer
    self.model = model
    self.max_token_length = max_token_length
    

  def predict(self, context, model_input):
    pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer,padding='max_length', truncation=True, max_length= self.max_token_length,framework = "pt")
    return pipe(list(model_input['text']))
