import math
import pandas as pd

# Baraja el dataframe
def barajar(df, seed=0):
    return df.sample(frac=1,random_state=seed).reset_index(drop=True)

# Hace subsample, la clase con menos elementos marcará el máximo de elementos por clase.
def subsample(df):
    uns = [0,1] # posibles clases
    df_classes = []
    c_min = math.inf

    for l in uns:
        c = df[df['label'].round() == l]
        c_min = min(c_min,c.shape[0])
        df_classes.append(c)

    for i,c in enumerate(df_classes):
        df_classes[i] = c.sample(n=c_min)

    df = pd.concat(df_classes).reset_index(drop=True)
    return df