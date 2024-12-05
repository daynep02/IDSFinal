from ..utils import utils
from sklearn.metrics import confusion_matrix
def evaluate_models_roc(knn, decision_tree, naive_bayes_model, X_validate):
    knn_probs = knn.predict_proba(X_validate)[:, 1]
    gaussian_probs = naive_bayes_model.predict_proba(X_validate)[:, 1]
    decision = decision_tree.predict_proba(X_validate)[:, 1]
    return {"KNN": knn_probs, "Gaussian": gaussian_probs, "Decision Tree": decision}

def evaluate_models(models, X_validate, y_validate):
    for model in models:
        y_pred = model.predict(X_validate)
        _confusion_matrix = confusion_matrix(y_validate, y_pred)
        utils.print_metrics(model, y_validate, y_pred, _confusion_matrix)

def get_best_model(roc_auc_scores: dict, knn_model, gaussian_model, decision_tree_model):
    match max(roc_auc_scores, key=roc_auc_scores.get):
        case 'KNN':
            return knn_model
        case 'Gaussian':
            return gaussian_model
        case 'Decision Tree':
            return decision_tree_model