import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from LR import *

df_voice = pd.read_csv("voice.csv")

# replacing male -> 0 and female -> 1
df_voice["label"] = df_voice["label"].replace("male", 0)
df_voice["label"] = df_voice["label"].replace("female", 1)

# get all columns except "label" (gender)
features = df_voice.keys()
features = features.drop("label") # remove label

# splitting dataset
df1 = df_voice.iloc[: int(len(df_voice)*0.66)] # 66% of data
df2 = df_voice.iloc[int(len(df_voice)*0.66)+1 :] # 34% of data

# training set
x_train = df1.loc[:, features].values
y_train = df1.loc[:, ['label']].values

# test set
x_test = df2.loc[:, features].values
y_test = df2.loc[:, ['label']].values

# standardizing the dataset
x_train = StandardScaler().fit_transform(x_train)

print(PCA_decomposition(x_train, y_train))

