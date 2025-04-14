# evaluate.py
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    print(classification_report(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, predictions[:, 1]))
