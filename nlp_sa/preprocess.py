def preprocess_function(examples, conf, Dataset, Model):
    """
    Function to Tokenize the data based on task name
    """
    # Tokenize the texts
    if conf.data_args.task_name is not None:
        if conf.data_args.task_name == "multi-class":
            result = _preprocess_function_multi_class(
                examples, conf, Dataset, Model
            )
        elif conf.data_args.task_name == "ner":
            result = _preprocess_function_ner(
                examples, conf, Dataset, Model
            )
        else:
            raise ValueError(
                "task not implemented please implement the task"
            )
    return result


def _preprocess_function_multi_class(examples, conf, Dataset, Model):
    """
    Function to Tokenize the data and perform any preprocessing required
    """
    # Tokenize the texts
    args = (examples[conf.data_args.feature_col],)
    result = Model.tokenizer(
        *args,
        max_length=conf.data_args.max_seq_length,
        truncation=True
    )

    return result


def _preprocess_function_ner(examples, conf, dataset, model):
    """
    Function to Tokenize the data and perform any preprocessing required
    """
    b_to_i_label = []
    for idx, label in enumerate(dataset.label_list):
        if (
            str(label).startswith("B-")
            and str(label).replace("B-", "I-") in dataset.label_list
        ):
            b_to_i_label.append(
                dataset.label_list.index(str(label).replace("B-", "I-"))
            )
        else:
            b_to_i_label.append(idx)
    padding = (
        "max_length" if conf.data_args.pad_to_max_length else False
    )

    """We use is_split_into_words because the texts in our dataset
    are lists of words (with a label for each word)."""
    tokenized_inputs = model.tokenizer(
        examples[conf.data_args.feature_col],
        padding=padding,
        truncation=True,
        max_length=conf.data_args.max_seq_length,
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[conf.data_args.label_col]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            """Special tokens have a word id that is None.
            We set the label to -100 so they are automatically
            ignored in the loss function."""
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(dataset.label_to_id[label[word_idx]])
            else:
                """For the other tokens in a word, we set the label
                to either the current label or -100, depending on
                the label_all_tokens flag."""
                if conf.data_args.label_all_tokens:
                    label_ids.append(
                        b_to_i_label[
                            dataset.label_to_id[label[word_idx]]
                        ]
                    )
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
