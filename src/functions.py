
# Import necessary libraries
import numpy as np
import pandas as pd
import warnings
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

# Function for loading data and label encoding
def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['diagnosis']).apply(pd.to_numeric, errors='coerce')
    y = LabelEncoder().fit_transform(df['diagnosis'])
    return X.values, y

# Function to compute bootstrap confidence intervals for the median 
def bootstrap_median_CI(data, n_bootstrap=1000, ci=95, random_state=42):
    """
    Compute the bootstrap confidence interval around the median.
    """
    np.random.seed(random_state)
    medians = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        medians.append(np.median(sample))
    
    lower_bound = np.percentile(medians, (100 - ci) / 2)
    upper_bound = np.percentile(medians, 100 - (100 - ci) / 2)
    median = np.median(data)
    
    return median, lower_bound, upper_bound

# Function to generate a table of median and confidence intervals
def generate_median_CI_table(model_runner, n_bootstrap=1000):
    """
    Generate a table of Model - Metric - Median - Lower CI - Upper CI
    """
    rows = []
    
    for model_name, metrics in model_runner.results.items():
        for metric_name, metric_values in metrics.items():
            all_values = metric_values['all']
            median, lower_ci, upper_ci = bootstrap_median_CI(all_values, n_bootstrap=n_bootstrap)
            
            row = {
                "Model": model_name,
                "Metric": metric_name,
                "Median": median,
                "95% CI Lower": lower_ci,
                "95% CI Upper": upper_ci
            }
            rows.append(row)
    
    df_median_ci = pd.DataFrame(rows)
    return df_median_ci

# Class for Nested Repeated Cross-Validation
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
    
    # Function to compute metrics on outer test set
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
    # Function to train models using nested cross-validation
    def train(self):
        for name, estimator in self.estimators:
            param_grid = self.hyperparameters.get(name, {})
            all_metrics = []

            for rnd in range(self.rounds):
                outer_cv = KFold(n_splits=self.N, shuffle=True, random_state=rnd+42) # random_state for reproducibility
                # Inner CV for hyperparameter tuning 
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
                    # Suppress warnings from GridSearchCV 
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
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

    # Function to print summary of results
    def summary(self):
        for name, metrics in self.results.items():
            print(f"Model: {name}")
            for metric, values in metrics.items():
                print(f"  {metric}: Mean = {values['mean']:.4f} | Std = {values['std']:.4f}")
            print("-" * 50)

# Function to select the best model based on metrics
def winner_model_selection(csv_path, primary_metric='MCC', secondary_metric='AUC'):
    
    # Load metrics summary CSV
    df = pd.read_csv(csv_path)

    # Select primary metric (MCC)
    df_primary = df[df['Metric'] == primary_metric]
    df_primary = df_primary.sort_values(by='Median', ascending=False).reset_index(drop=True)

    top1 = df_primary.iloc[0]
    top2 = df_primary.iloc[1]

    print(f"Checking {primary_metric} first...")
    print(f"Top 1: {top1['Model']} - Median: {top1['Median']}, 95% CI: [{top1['95% CI Lower']}, {top1['95% CI Upper']}]")
    print(f"Top 2: {top2['Model']} - Median: {top2['Median']}, 95% CI: [{top2['95% CI Lower']}, {top2['95% CI Upper']}]")

    # Decision based on primary metric
    if top1['95% CI Lower'] > top2['95% CI Upper']:
        print(f"\n Winner: {top1['Model']} based on {primary_metric} (clear statistical difference)")
        return top1['Model']
    else:
        print("\n MCC CIs overlap — checking secondary metric (AUC)...")

        # Select secondary metric (AUC)
        df_secondary = df[df['Metric'] == secondary_metric]
        df_secondary = df_secondary[df_secondary['Model'].isin([top1['Model'], top2['Model']])]
        df_secondary = df_secondary.sort_values(by='Median', ascending=False).reset_index(drop=True)

        secondary_top = df_secondary.iloc[0]

        print(f"\n{secondary_metric} comparison:")
        for idx, row in df_secondary.iterrows():
            print(f"{row['Model']} - Median {secondary_metric}: {row['Median']}")

        print(f"\n Winner: {secondary_top['Model']} based on {secondary_metric} (better separability)")
        return secondary_top['Model']

