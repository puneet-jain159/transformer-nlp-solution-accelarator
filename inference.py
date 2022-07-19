import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import mlflow
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("sqlite:///mlflow.db")


client = MlflowClient()

  # Copy artifacts to driver
loaded_model = mlflow.pyfunc.load_model(model_uri=f"./mlruns/1/0025e4b0d72749a18992453ab49d03b5/artifacts/mlflow_model")

model = loaded_model._model_impl.python_model.model

tokenizer = loaded_model._model_impl.python_model.tokenizer
max_token_length = loaded_model._model_impl.python_model.max_token_length

apply_tokenizer = lambda x: tokenizer(str(x), padding='max_length', truncation=True, max_length= max_token_length, return_tensors='pt') 



with torch.no_grad():
    apply_model = lambda x: model(x['input_ids'], x['attention_mask']).logits
    softmax = torch.nn.Softmax(dim=1)
    
    apply_softmax = lambda x: softmax(x).numpy().flatten()
    apply_rounding = lambda x: np.around(x, decimals=4)
    arg_max = lambda x: np.argmax(x)

    model_input = d.iloc[:, 0].apply(apply_tokenizer)
    model_input = model_input.apply(apply_model)
    model_input = model_input.apply(apply_softmax)
    # model_input = model_input.apply(apply_rounding)
    model_input = model_input.apply(arg_max)

model_input
import pandas as pd

    d =pd.DataFrame([
    { "text": "Do you have info about the card on delivery?" },
    {"test" : "when do I get my card?"},
    {"test" : "What can I do if my card still hasn't arrived after 2 weeks?"}
])

d =pd.DataFrame(['Do you have info about the card on delivery?',
 'when do I get my card?',
 'What can I do if my card still hasnt arrived after 2 weeks?'],columns=['text'])

from transformers import TextClassificationPipeline

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer,padding='max_length', truncation=True, max_length= max_token_length,framework = "pt")

pipe(list(d['text']))

