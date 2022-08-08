import numpy as np
from transformers import EvalPrediction
    
    
# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.

def compute_metrics(p: EvalPrediction,conf,metric):
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if conf.model_args.is_regression else np.argmax(preds, axis=1)
    if conf.data_args.task_name is not None:
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    elif conf.model_args.is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}