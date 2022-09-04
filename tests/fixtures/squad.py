from datasets import load_dataset
import pytest

@pytest.fixture
def dataset():

    dataset = load_dataset(
        "lhoestq/custom_squad",
        revision="main"
    )

    return dataset