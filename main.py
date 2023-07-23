import pandas as pd
from utils.tools import train_test_split
from config.config_model import Config
from train import train
from prediction import predict

df = pd.read_csv('stock_alldays.csv')
company_name = df.name.unique()
# df = df[df.name == 'KTC']
df = df[['time', 'name','open','high','low','close']]
# df = df.drop(columns=['lastSequence', 'open', 'high', 'low', 'close', 'volume', 'value', 'name', 'open_previous', 'close_previous', 'high_previous', 'low_previous', 'volume_previous', 'open_future', 'close_future', 'high_future', 'low_future', 'volume_future', 'first_day', 'last_day', 'period'])
df = df.rename(columns = {'time': 'date'})
df = df.set_index('date')
# df = df.pct_change()
# df = df.dropna()

company_name = df.name.unique()

predict_pct_change = {}
config = Config()

for idx, name in enumerate(company_name):
    
    data = df[df.name == name].copy()
    data = data.drop(columns = ['name'])
    data = data.pct_change()
    data = data.dropna()
    # display(data)
    _, val_data = train_test_split(data, config.train_test_split)

    if len(data) < config.seq_len or len(val_data) < config.seq_len:
        continue
    print(name)
    model = train(data)
    # print('test')
    pred = predict(model, val_data)
    predict_pct_change[name] = pred
    pd.DataFrame(predict_pct_change.items(), columns = ['Stock', 'Percentage Change']).to_csv('stock_ranking.csv', index = False)

    # if idx == 3:
    #    break

# pd.DataFrame(predict_pct_change.items(), columns = ['Stock', 'Percentage Change']).to_csv('stock_ranking.csv', index = False)
