import torch
import joblib
from config.config_model import Config
def predict(model,val_df):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config()
    actual_pred = [val_df.iloc[:,-1][-1]]
    scaler = joblib.load('scaler.save')
    val_df_norm = val_df.copy()
    cols = val_df.columns
    val_df_norm[cols] = scaler.transform(val_df[cols])

    with torch.no_grad():
        input = torch.tensor(val_df_norm.iloc[-config.seq_len-1:-1].values).float().unsqueeze(dim=0).to(device)
        output = model(input)
        output = scaler.inverse_transform(output.squeeze(dim = 0).cpu()) # aggregate the target column
        actual_pred.append(output[:,-1].item())
    return output[:,-1].item()