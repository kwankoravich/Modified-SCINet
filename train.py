import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchinfo import summary
import torch.nn as nn
from torcheval.metrics import R2Score
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import joblib

from config.config_model import Config
from data.CustomDataset import CustomDatasets
from model.SCINet_decompose import SCINet_decompose
from utils.tools import train_test_split, normalize
from utils.loss import custom_loss



def train(data):

    config = Config()
    # data = df.copy()

    ## train test split
    train_df, val_df = train_test_split(data, config.train_test_split)
    # display(train_df)
    # train_cp_df, val_cp_df = train_df.iloc[:,:3], val_df.iloc[:,:3]
    print(f"train_df.shape: {train_df.shape}")
    print(f"val_df.shape: {val_df.shape}")

    ## normalization
    train_df_norm, scaler = normalize(config.norm_method, train_df)
    # train_df_norm = train_df.copy()
    scaler_filename = "scaler.save"
    joblib.dump(scaler, scaler_filename)
    scaler = joblib.load(scaler_filename)

    val_df_norm = val_df.copy()
    cols = val_df.columns
    val_df_norm[cols] = scaler.transform(val_df[cols])

    # display(train_df_norm)
    # display(val_df_norm)

    while True:
        if config.seq_len % (np.power(2, config.levels)) != 0:
            config.seq_len +=1
        else:
            break
    ## dataloader
    train_ds = CustomDatasets(df=train_df_norm,
                            seq_len=config.seq_len,
                            pred_len=config.pred_len)
    val_ds = CustomDatasets(df=val_df_norm,
                            seq_len=config.seq_len,
                            pred_len=config.pred_len)
    train_loader = DataLoader(dataset=train_ds,
                            batch_size=config.batch_size,
                            shuffle=True,
                            drop_last=False)
    val_loader = DataLoader(dataset=val_ds,
                            batch_size=config.batch_size,
                            shuffle=False,
                            drop_last=False)


    # create model
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = SCINet_decompose(
                output_len=config.pred_len,
                input_len=config.seq_len,
                input_dim= config.in_dim,
                hid_size = config.hidden_size,
                num_stacks=config.stacks,
                num_levels=config.levels,
                concat_len = config.concat_len,
                groups = config.groups,
                kernel = config.kernel,
                dropout = config.dropout,
                single_step_output_One = config.single_step_output_One,
                positionalE = config.positionalEcoding,
                modified = True,
                RIN=config.RIN).to(device)

    print(summary(model,
            input_size=(config.batch_size, config.seq_len, len(train_df.columns)),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]))

    # optimizer
    n_epochs = config.n_epochs
    learning_rate = config.lr
    if config.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if config.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # train loop
    best_train_loss = 999
    best_epoch = -1
    train_loss_per_ep = []
    train_r2_per_ep = []
    val_loss_per_ep = []
    val_r2_per_ep = []

    # LOOP EPOCHS
    for epoch in tqdm(range(n_epochs)):
        # train
        train_loss = []
        train_r2 = []
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device)
            # print(input.shape)
            output = model(input)
            # output = output[:,:,-1].squeeze() #adjusted
            # print(output.shape)
            # criterion = nn.MSELoss()
            # loss = criterion(output,target)
            loss = custom_loss(output,target)
            # metric = R2Score()
            metric = R2Score().to(device) #adjusted
            # metric.update(output, target)
            # metric.update(output, target.squeeze())
            metric.update(output.reshape(-1, output.shape[2]), target.reshape(-1, target.shape[2]))
            r_squared = metric.compute().cpu()
            # r_squared = 0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            train_r2.append(r_squared)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        train_loss_per_ep.append(np.average(train_loss))
        train_r2_per_ep.append(np.average(train_r2))


        # LOOP EPOCHS
        val_loss = []
        val_r2 = []
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                # output = output[:,:,-1].squeeze() #adjusted
                # criterion = nn.MSELoss()
                # loss = criterion(output,target)
                loss = custom_loss(output,target)
                # metric = R2Score()
                metric = R2Score().to(device)
                # metric.update(output, target)
                metric.update(output.reshape(-1, output.shape[2]), target.reshape(-1, target.shape[2]))
                # metric.update(output.squeeze(dim = 1), target)
                # metric.update(output, target.squeeze())
                r_squared = metric.compute().cpu()
                # r_squared = 0
                val_loss.append(loss.item())
                val_r2.append(r_squared)

            val_loss_per_ep.append(np.average(val_loss))
            val_r2_per_ep.append(np.average(val_r2))

        print(f"[{epoch}/{n_epochs}] ==Train== loss: {np.average(train_loss):.4f}, average_r_squared: {np.average(train_r2):.4f}, lr: {before_lr:.6f} -> {after_lr:.6f} | ==Val== loss: {np.average(val_loss):.4f}, average_r_squared: {np.average(val_r2):.4f}")

    return model
