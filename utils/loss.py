import torch
import torch.nn as nn


def correlation_loss(output,target):

    # Convert the data to PyTorch tensors
    tensor_1 = torch.tensor(output, dtype=torch.float32)
    tensor_2 = torch.tensor(target, dtype=torch.float32)

    # Compute the correlation coefficient
    mean_1 = torch.mean(tensor_1, dim=1, keepdim=True)
    mean_2 = torch.mean(tensor_2, dim=1, keepdim=True)
    covariance = torch.mean((tensor_1 - mean_1) * (tensor_2 - mean_2), dim=1, keepdim=True)
    std_1 = torch.std(tensor_1, dim=1, keepdim=True)
    std_2 = torch.std(tensor_2, dim=1, keepdim=True)

    correlation = covariance / (std_1 * std_2)
    correlation = torch.mean(correlation)

    # print("Correlation coefficient:", correlation.item())

    return torch.exp(-correlation)

def custom_loss(output, target):
    """
    
    """

    # 1. Mean Square Error
    criterion = nn.MSELoss()
    mse_loss = criterion(output,target)
    # print('mse_loss',mse_loss.shape)
    # print(mse_loss)

    # 2. Correlation efficient
    if output.shape[1] > 1:
      corr_loss = correlation_loss(output, target)
    else:
      corr_loss = 0
    # print('corr_loss',corr_loss.shape)

    # 3. Total loss

    total_loss = mse_loss + corr_loss


    # take mean and exp to convert to scalar values of each batch
    return total_loss 