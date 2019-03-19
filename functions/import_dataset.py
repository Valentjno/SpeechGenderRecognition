import pandas as pd
from sklearn.utils import shuffle

def imp_dataset(dataset_path):

  df_voice = pd.read_csv(dataset_path)
  df_voice = shuffle(df_voice)

  # replacing male -> 0 and female -> 1
  df_voice["label"] = df_voice["label"].replace("male", 0)
  df_voice["label"] = df_voice["label"].replace("female", 1)

  # get all columns except "label" (gender)
  features = df_voice.keys()
  features = features.drop("label") # remove label

  # remove features less significant
  # features = features.drop(["dfrange", "mindom", "centroid", "mode", "sfm", "IQR", "median", "sd"])
  # features = features.drop(["meanfun", "maxfun", "minfun", "meandom", "mindom", "maxdom", "dfrange", "modindx"])

  # splitting dataset
  df1 = df_voice.iloc[: int(len(df_voice)*0.66)] # 66% of data
  df2 = df_voice.iloc[int(len(df_voice)*0.66)+1 :] # 34% of data

  # training set
  x_train = df1.loc[:, features].values
  y_train = df1.loc[:, ['label']].values

  # test set
  x_test = df2.loc[:, features].values
  y_test = df2.loc[:, ['label']].values

  return x_train, y_train, x_test, y_test
