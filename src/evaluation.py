from xgboost import XGBClassifier
from tfidf import x_traintfidf, x_testtfidf, y_train, y_test
from loaddata import train_df, test_df
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix
def evaluate_model(y_test,ypred):
    accuracy=accuracy_score(y_test,ypred)
    report = classification_report(y_test,ypred)        
    matrix = confusion_matrix(y_test,ypred)
    return accuracy, report, matrix