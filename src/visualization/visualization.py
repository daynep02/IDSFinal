import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree
from sklearn.metrics import roc_auc_score, roc_curve

def plot_knn_with_variable_neighbors(neighbors: np.array, train_accuracies: dict, test_accuracies: dict):
    plt.figure(figsize=(8, 6))
    plt.title("KNN: Varying Number of Neighbors")
    plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
    plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
    plt.legend()
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.show()

def plot_decision_tree(decision_tree):
    plt.figure(figsize=(25,25))
    plot_tree(decision_tree=decision_tree, filled=True, class_names=["Empty", "Occupied"],fontsize=20)
    plt.title(f'Decision Tree')
    plt.show()

def plot_roc_curve(y_val, y_pred, color, name, roc_auc_scores):
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    plt.plot(fpr, tpr, color=color, label=name)
    roc_auc_scores[name] = roc_auc_score(y_val, y_pred)

def plot_roc_curves(y_validate, rocs: dict) -> dict:
    
    plt.plot([0, 1], [0, 1], 'k--')
    color_index = 0
    colors = ['blue', 'red', 'green']
    roc_auc_scores = {}
    for name in rocs:
        plot_roc_curve(y_validate, rocs[name], colors[color_index], name, roc_auc_scores)
        color_index += 1
        roc_auc_scores[name] = roc_auc_score(y_validate, rocs[name])
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    return roc_auc_scores