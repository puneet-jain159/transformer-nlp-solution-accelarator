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

    # # Map labels to IDs (not necessary for GLUE tasks)
    # if Model.model.config.label2id is not None and conf.data_args.feature_col in examples:
    #     result[conf.data_args.feature_col] = [(Model.model.config.label2id [l] if l != -1 else -1)
    #                        for l in examples[conf.data_args.feature_col]]
    return result


def _preprocess_function_ner(examples, conf, Dataset, Model):
    """
    Function to Tokenize the data and perform any preprocessing required
    """
    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(Dataset.label_list):
        if (
            str(label).startswith("B-")
            and str(label).replace("B-", "I-") in Dataset.label_list
        ):
            b_to_i_label.append(
                label_list.index(str(label).replace("B-", "I-"))
            )
        else:
            b_to_i_label.append(idx)

    padding = (
        "max_length" if conf.data_args.pad_to_max_length else False
    )

    tokenized_inputs = Model.tokenizer(
        examples[conf.data_args.feature_col],
        padding=padding,
        truncation=True,
        max_length=conf.data_args.max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[conf.data_args.label_col]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(Dataset.label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if conf.data_args.label_all_tokens:
                    label_ids.append(
                        b_to_i_label[
                            Dataset.label_to_id[label[word_idx]]
                        ]
                    )
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
