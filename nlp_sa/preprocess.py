def preprocess_function(examples, conf, Dataset, Model):
    '''
    Function to Tokenize the data and perform any preprocessing required
    '''
    # Tokenize the texts
    args = (examples[conf.data_args.feature_col],)
    result = Model.tokenizer(*args, max_length=conf.data_args.max_seq_length, truncation=True)

    # # Map labels to IDs (not necessary for GLUE tasks)
    # if Model.model.config.label2id is not None and conf.data_args.feature_col in examples:
    #     result[conf.data_args.feature_col] = [(Model.model.config.label2id [l] if l != -1 else -1)
    #                        for l in examples[conf.data_args.feature_col]]
    return result


