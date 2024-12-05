
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def specificity(confusion_matrix):
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    if tn == fp and tn == 0:
        return "Undefined"
    return tn / (tn + fp)

def print_metrics(model, y_val, y_val_pred, confusion_matrix):
    print(f"{model}: ")
    print(f"\tAccuracy Score: {accuracy_score(y_val, y_val_pred)}\n\tPrecision Score: {precision_score(y_val, y_val_pred, zero_division=0)}")
    print(f"\tRecall Score: {recall_score(y_val, y_val_pred)}\n\tSpecificity: {specificity(confusion_matrix)}")
    print(f"\tF1 Score: {f1_score(y_val, y_val_pred, zero_division=0)}")