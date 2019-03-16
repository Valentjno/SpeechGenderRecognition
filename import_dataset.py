import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from LR import *

df_voice = pd.read_csv("voice.csv")
df_voice = shuffle(df_voice)

# replacing male -> 0 and female -> 1
df_voice["label"] = df_voice["label"].replace("male", 0)
df_voice["label"] = df_voice["label"].replace("female", 1)

# get all columns except "label" (gender)
features = df_voice.keys()
features = features.drop("label") # remove label

# remove features less significant
# features = features.drop(["dfrange", "mindom", "centroid", "mode", "sfm", "IQR", "median", "sd"])

# splitting dataset
df1 = df_voice.iloc[: int(len(df_voice)*0.66)] # 66% of data
df2 = df_voice.iloc[int(len(df_voice)*0.66)+1 :] # 34% of data

# training set
x_train = df1.loc[:, features].values
y_train = df1.loc[:, ['label']].values

# test set
x_test = df2.loc[:, features].values
y_test = df2.loc[:, ['label']].values

x_train, x_test = normalize_L2(x_train, x_test)
x_train, x_test = PCA_decomposition(x_train, x_test)

'''
lr = fit_LR(x_train, y_train)
print("LR")
predict_and_score(lr, x_test, y_test)

lr = fit_Bernoulli_NB(x_train, y_train)
print("NB")
predict_and_score(lr, x_test, y_test)

lr = fit_SVC(x_train, y_train, _gamma="scale")
print("SVC")
predict_and_score(lr, x_test, y_test)

lr = fit_2NN(x_train, y_train, _algorithm="ball_tree", _weights="distance")
print("2NN")
predict_and_score(lr, x_test, y_test)
'''