import pandas as pd
synthetic_data = pd.read_csv("/Users/toaaramadan/Desktop/machine projects/AIDataInbreeding/data/synthetic/G1/synthetic_G1.csv")
real_data = pd.read_csv("data/train.csv")
g1_data = pd.concat(
    [real_data, synthetic_data],
    ignore_index=True
)
g1_data.to_csv("data/G1_train.csv", index=False)
print("G1 shape:", g1_data.shape)
print(g1_data["Class Index"].value_counts().sort_index())