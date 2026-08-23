import pandas as pd

train_ds = pd.read_pickle("data/pkls/crisp/rb_db600train.pkl")
print(train_ds[(train_ds['label'] != 0) & (train_ds['label'] != 1)])