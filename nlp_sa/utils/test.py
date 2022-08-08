
import torch

def over_fit_check(trainer)
'''
Function to overfit the model on a small batch of external data to make sure
the weights are getting updated return Either Error or no value

## Todo only being done for multi-class needs to be completed for other pipelines
'''
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    for batch in trainer.get_train_dataloader():
        break

    device = "cpu"
    batch = {k: v.to(device) for k, v in batch.items()}
    trainer.create_optimizer()

    for _ in range(100):
        outputs = trainer.model(**batch)
        loss = outputs.loss
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    with torch.no_grad():
    ts = trainer.model(**batch)
    preds = outputs.logits
    labels = batch["labels"]

    acc = metric.compute(predictions=np.argmax(preds.cpu().detach().numpy(), axis=1), references=labels)

    if acc['accuracy'] != 1:
        raise f"The model parameters need to be reconfigured"


