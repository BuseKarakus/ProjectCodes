import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.special import expit
from pyswarm import pso


data_path = '/Users/b/Desktop/heart_statlog_cleveland_hungary_final.csv'
data = pd.read_csv(data_path)


X = data.drop(columns='target')
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def logistic_regression_loss(params, X, y):
    intercept = params[0]
    coefficients = params[1:]
    linear_combination = np.dot(X, coefficients) + intercept
    predictions = expit(linear_combination)

    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return loss


num_features = X_train.shape[1]
bounds = [(-5, 5)] * (num_features + 1)


optimal_params, _ = pso(logistic_regression_loss, [-5]*(num_features+1), [5]*(num_features+1), args=(X_train, y_train), swarmsize=50, maxiter=100)


intercept_optimal = optimal_params[0]
coefficients_optimal = optimal_params[1:]


linear_combination_test = np.dot(X_test, coefficients_optimal) + intercept_optimal
y_pred_prob = expit(linear_combination_test)
y_pred = (y_pred_prob >= 0.5).astype(int)


accuracy = accuracy_score(y_test, y_pred)
sensitivity = confusion_matrix(y_test, y_pred)[1, 1] / sum(y_test)
specificity = confusion_matrix(y_test, y_pred)[0, 0] / sum(1 - y_test)
auc = roc_auc_score(y_test, y_pred_prob)


print(f'Accuracy: {accuracy:.4f}')
print(f'Sensitivity: {sensitivity:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'AUC: {auc:.4f}')


benchmarks = {
    "Gradient Descent": {"Accuracy": 0.8303, "Sensitivity": 0.8283, "Specificity": 0.8324, "AUC": 0.9053},
    "Genetic Algorithm": {"Accuracy": 0.8328, "Sensitivity": 0.8410, "Specificity": 0.8235, "AUC": 0.9062},
    "Simulated Annealing": {"Accuracy": 0.8328, "Sensitivity": 0.8410, "Specificity": 0.8235, "AUC": 0.9062},
    "Randomized Hill Climbing": {"Accuracy": 0.8336, "Sensitivity": 0.8362, "Specificity": 0.8307, "AUC": 0.9057}
}


print("\nComparison with provided benchmarks:")
for method, metrics in benchmarks.items():
    print(f"{method}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")



benchmarks = {
    "Gradient Descent": {"Accuracy": 0.8303, "Sensitivity": 0.8283, "Specificity": 0.8324, "AUC": 0.9053},
    "Genetic Algorithm": {"Accuracy": 0.8328, "Sensitivity": 0.8410, "Specificity": 0.8235, "AUC": 0.9062},
    "Simulated Annealing": {"Accuracy": 0.8328, "Sensitivity": 0.8410, "Specificity": 0.8235, "AUC": 0.9062},
    "Randomized Hill Climbing": {"Accuracy": 0.8336, "Sensitivity": 0.8362, "Specificity": 0.8307, "AUC": 0.9057}
}


pso_results = {"Accuracy": 0.8543, "Sensitivity": 0.8818, "Specificity": 0.8182, "AUC": 0.9147}


comparison_df = pd.DataFrame(benchmarks).T
comparison_df.loc['PSO'] = pso_results


print(comparison_df)
