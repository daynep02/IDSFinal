from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from ..visualization import visualization

def get_best_knn(X_train, y_train, X_valid, y_valid) -> KNeighborsClassifier:
    train_accuracies = {}
    test_accuracies = {}
    neighbors = np.arange(1,26, 2)
    print(neighbors)
    for neighbor in neighbors:
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        knn.fit(X_train, y_train)
        train_accuracies[neighbor] = knn.score(X_train, y_train)
        test_accuracies[neighbor] = knn.score(X_valid, y_valid)
    optimal_neighbor = max(test_accuracies, key=test_accuracies.get)
    visualization.plot_knn_with_variable_neighbors(neighbors, train_accuracies, test_accuracies)
    return KNeighborsClassifier(n_neighbors=optimal_neighbor)

def train_decision_tree(X_train, y_train, depth_limit:int = 4) -> DecisionTreeClassifier:
    decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth_limit)
    decision_tree.fit(X_train, y_train)
    return decision_tree

def train_gaussian(X_train, y_train) -> GaussianNB:

    naive_bayes_model = GaussianNB()

    naive_bayes_model.fit(X_train, y_train)
    return naive_bayes_model