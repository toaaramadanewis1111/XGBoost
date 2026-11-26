from xgboost import XGBClassifier
from tfidf import x_traintfidf, x_testtfidf, y_train, y_test
from loaddata import train_df, test_df
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix
from evaluation import evaluate_model
import pandas as pd
model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42
)
model.fit(x_traintfidf,y_train)
ypred= model.predict(x_testtfidf)
accuracy, report, matrix = evaluate_model(y_test, ypred)
print(accuracy * 100)
print(report)
print(matrix)
results = pd.DataFrame({
    "Generation": ["G0"],
    "Synthetic_Percentage": [0],
    "Accuracy": [accuracy]
})
results.to_csv("experiment_results.csv", mode="a", index=False, header=False)