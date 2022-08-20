from src import data_loader


def test_load_training_set():

    loader = data_loader.DataLoader(
        dataset_name = "squad",
        train_split = "train",
        do_eval = False
    )

    assert loader.train

def test_load_validation_set():

    loader = data_loader.DataLoader(
        dataset_name = "squad",
        eval_split = "validation",
        do_eval = True,
        do_train = False
    )

    assert loader.test