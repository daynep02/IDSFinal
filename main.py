from src.features import build_features
from src.data import load_data
from src.data import preprocess
from src.features import build_features
from src.models import train_models
from src.visualization import visualization
from src.models import validate_models
from tests.tests import test_best_model
import numpy as np

def main():
    np.random.seed(15789034)
    data1, data2, data3 = load_data.load_data("data/raw/datatest.csv", "data/raw/datatest2.csv", "data/raw/datatraining.csv")
    data = preprocess.preprocess(data1, data2, data3)
    print(data.describe)

    data.to_csv("data/processed/full_data.csv")

    X_train, X_validate, X_test, y_train, y_validate, y_test = build_features.build_features(data)

    knn_model = train_models.get_best_knn(X_train, y_train, X_validate, y_validate)
    knn_model.fit(X_train, y_train)

    gaussian_model = train_models.train_gaussian(X_train, y_train)

    decision_tree = train_models.train_decision_tree(X_train, y_train, 4)

    visualization.plot_decision_tree(decision_tree)

    validate_models.evaluate_models([ knn_model,  gaussian_model, decision_tree],
                                     X_validate, y_validate)
    pred_probs = validate_models.evaluate_models_roc(knn_model, decision_tree, gaussian_model, X_validate)
    roc_auc_scores = visualization.plot_roc_curves(y_validate, pred_probs)
    print(roc_auc_scores)
    best_model = validate_models.get_best_model(roc_auc_scores, knn_model, gaussian_model, decision_tree)

    print(f'Best model is {best_model}')
    test_best_model(best_model, X_test, y_test)


if __name__ == '__main__':
    main()