
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score, 
    f1_score, fbeta_score, recall_score, precision_score, 
    confusion_matrix, make_scorer, average_precision_score
)
import numpy as np
import pandas as pd

class nrCV:
    def __init__(self, dataset, estimators, hyperparameters, rounds=10, N=5, K=3, 
                 inner_metric="f1_macro", scoring_strategy="macro"):
        """
        Repeated nested cross-validation for classification, supporting imbalanced datasets.
        
        Parameters:
            dataset: Tuple (X, y)
            estimators: list of (name, estimator) tuples
            hyperparameters: dict of {estimator_name: param_grid}
            rounds: number of repetitions of nested CV
            N: number of outer folds
            K: number of inner folds
            inner_metric: metric used for GridSearchCV (e.g., 'f1_macro')
            scoring_strategy: 'macro', 'weighted', or 'binary' (used in metrics)
        """
        self.X, self.y = dataset
        self.estimators = estimators
        self.hyperparameters = hyperparameters
        self.rounds = rounds
        self.N = N
        self.K = K
        self.inner_metric = inner_metric
        self.scoring_strategy = scoring_strategy
        self.results = {}

    def compute_outer_metrics(self, y_true, y_pred, y_proba):
        """Compute metrics on outer test set."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) == 2 else [0,0,0,0]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        return {
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, average=self.scoring_strategy),
            "F2": fbeta_score(y_true, y_pred, beta=2, average=self.scoring_strategy),
            "Recall": recall_score(y_true, y_pred, average=self.scoring_strategy),
            "Precision": precision_score(y_true, y_pred, average=self.scoring_strategy),
            "Specificity": specificity,
            "NPV": npv,
            "AUC": roc_auc_score(y_true, y_proba, multi_class='ovr' if len(np.unique(y_true)) > 2 else 'raise', average=self.scoring_strategy),
            "PR-AUC": average_precision_score(y_true, y_proba, average=self.scoring_strategy),
        }

    def train(self):
        for name, estimator in self.estimators:
            param_grid = self.hyperparameters.get(name, {})
            all_metrics = []

            for rnd in range(self.rounds):
                outer_cv = KFold(n_splits=self.N, shuffle=True, random_state=rnd+42)
                inner_cv = KFold(n_splits=self.K, shuffle=True, random_state=rnd+42)

                for train_idx, test_idx in outer_cv.split(self.X, self.y):
                    X_train, y_train = self.X[train_idx], self.y[train_idx]
                    X_test, y_test = self.X[test_idx], self.y[test_idx]

                    # Build pipeline: imputer → scaler → estimator
                    pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("clf", estimator)
                    ])

                    # Update param grid for the pipeline: prefix with "clf__"
                    param_grid_pipe = {f"clf__{k}": v for k, v in param_grid.items()}

                    clf = GridSearchCV(estimator=pipe, param_grid=param_grid_pipe,
                                    cv=inner_cv, scoring=self.inner_metric)
                    clf.fit(X_train, y_train)

                    best_model = clf.best_estimator_
                    y_pred = best_model.predict(X_test)
                    y_proba = (best_model.predict_proba(X_test)[:, 1]
                            if hasattr(best_model, "predict_proba") else y_pred)

                    metrics = self.compute_outer_metrics(y_test, y_pred, y_proba)
                    all_metrics.append(metrics)

            # Aggregate metrics
            final_scores = {}
            for key in all_metrics[0]:
                values = [m[key] for m in all_metrics]
                final_scores[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "all": values
                }

            self.results[name] = final_scores

    def summary(self):
        for name, metrics in self.results.items():
            print(f"Model: {name}")
            for metric, values in metrics.items():
                print(f"  {metric}: Mean = {values['mean']:.4f} | Std = {values['std']:.4f}")
            print("-" * 50)



def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['diagnosis']).apply(pd.to_numeric, errors='coerce')
    y = LabelEncoder().fit_transform(df['diagnosis'])
    return X.values, y

X, y = load_data("../data/breast_cancer_final.csv")

estimators = [
    ("SVM", SVC(probability=True)),
    ("RF", RandomForestClassifier(random_state=42))
]

hyperparameters = {
    "SVM": {"C": [1, 10], "gamma": [0.01, 0.1]},
    "RF": {"n_estimators": [50, 100]}
}

cv = nrCV(dataset=(X, y), estimators=estimators, hyperparameters=hyperparameters,
           rounds=5, N=3, K=2, inner_metric='f1_macro')
cv.train()
cv.summary()
