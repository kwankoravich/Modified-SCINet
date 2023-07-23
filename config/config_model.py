class Config:
    train_test_split = 0.8
    norm_method = 'standard' # 'minmax', 'standard'
    batch_size = 64
    seq_len = 30
    pred_len = 1
    hidden_size = 128
    n_layer = 20
    n_epochs = 500
    lr = 3e-4
    opt = "adamw"
    # for SciNet
    in_dim = 4
    hidden_size = 128
    stacks = 1
    levels = 3
    num_decoder_layer = 1
    concat_len = 0
    groups = 1
    kernel = 5 # 3, 5, 7
    dropout = 0.5
    single_step_output_One = 0
    positionalEcoding = False
    RIN = False