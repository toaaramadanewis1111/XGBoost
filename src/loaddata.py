import pandas as pd
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
train_df["text"] = train_df["Title"] + " " + train_df["Description"]
test_df["text"] = test_df["Title"] + " " + test_df["Description"]
X_train =  train_df["text"]
y_train = train_df["Class Index"]-1
X_test = test_df["text"]
y_test = test_df["Class Index"]-1
