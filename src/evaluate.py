"""
This module contains helper functions for computing model metrics.

You can define your custom compute_metrics function.
It takes an `EvalPrediction` object (a named tuple with a predictions
and label_ids field) and has to return a dictionary string to float.
"""

import numpy as np
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction, conf, metric, dataset):
    if conf.data_args.task_name is not None:
        if conf.data_args.task_name == "multi-class":
            result = _compute_metrics_multi_class(p, conf, metric)
        elif conf.data_args.task_name == "ner":
            result = _compute_metrics_ner(p, conf, metric, dataset)
        else:
            raise ValueError(
                "task not implemented please implement the task"
            )
    return result


def _compute_metrics_multi_class(p: EvalPrediction, conf, metric):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    predictions = predictions.reshape(len(predictions), -1)

    if conf.data_args.task_name is not None:
        result = metric.compute(
            predictions=predictions,
            references=labels.reshape(len(labels), -1),
        )
        if conf.data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        elif conf.model_args.is_regression:
            return {"mse": ((predictions - p.label_ids) ** 2).mean().item()}
        else:
            return {
                "precision": result["overall_precision"],
                "recall": result["overall_recall"],
                "f1": result["overall_f1"],
<<<<<<< HEAD
                "accuracy": result["overall_accuracy"]}
=======
                "accuracy": result["overall_accuracy"],
            }
        if len(result) > 1:
            result["combined_score"] = np.mean(
                list(result.values())
            ).item()
            return result
>>>>>>> parent of 512300e (lint)
    else:
        raise ValueError("task name not defined")


def _compute_metrics_ner(
    examples,
    conf,
    metric,
    dataset,
    special_token_ids=[-100]
):
    """
    Function to compute the metric of NER Task
    """
    predictions, labels = examples
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    for special_token in special_token_ids:
        predictions.remove(special_token)
        labels.remove(special_token)

    results = metric.compute(
        predictions=predictions, references=labels
    )

    if conf.data_args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
