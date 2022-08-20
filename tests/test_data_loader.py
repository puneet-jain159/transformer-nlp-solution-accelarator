from src import data_loader


def test_load_dataset():

    loader = data_loader.DataLoader(
        dataset_name = "squad",
        do_eval = False
    )

    assert loader.train