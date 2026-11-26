from sklearn.feature_extraction.text import TfidfVectorizer
from loaddata import X_train, y_train, X_test, y_test       
vectorizer = TfidfVectorizer()
x_traintfidf= vectorizer.fit_transform(X_train)
x_testtfidf = vectorizer.transform(X_test)
