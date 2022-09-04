from src.model_builder import ModelBuilder
from tests.fixtures.sentiment import dataset

def test_model_builder_sentiment(dataset):

    builder = ModelBuilder(
        dataset = dataset,
        model_name_or_path = "cardiffnlp/twitter-roberta-base",
        task_name = "sentiment"
    )

    assert builder