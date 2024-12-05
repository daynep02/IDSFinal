from sklearn.metrics import confusion_matrix
from src.utils.utils import print_metrics
def test_best_model(model, X_test, y_test):
    y_final_pred = model.predict(X_test)
    confusion = confusion_matrix(y_test, y_final_pred)
    print_metrics(model, y_test, y_final_pred, confusion) 