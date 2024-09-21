import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing
import random
from sklearn.metrics import roc_auc_score, confusion_matrix


file_path = "/Users/b/Desktop/heart_statlog_cleveland_hungary_final.csv"
data = pd.read_csv(file_path)


feature_columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'resting ecg',
                   'max heart rate', 'exercise angina', 'oldpeak', 'ST slope']
target_column = 'target'


X = data[feature_columns].values
y = data[target_column].values


X = (X - X.mean(axis=0)) / X.std(axis=0)


X = np.hstack((np.ones((X.shape[0], 1)), X))



def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-5
    cost = -(1 / m) * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon)))
    return cost



def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        theta = theta - (learning_rate / m) * np.dot(X.T, sigmoid(np.dot(X, theta)) - y)
        cost_history.append(cost_function(theta, X, y))

    return theta, cost_history



def performance_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = y_pred_proba >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_pred_proba)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy, sensitivity, specificity, auc



theta = np.zeros(X.shape[1])
learning_rate = 0.01
iterations = 1000
theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)
predictions_proba = sigmoid(np.dot(X, theta))
accuracy, sensitivity, specificity, auc = performance_metrics(y, predictions_proba)
print(f"Gradient Descent - Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}, AUC: {auc}")
print("Gradient Descent Coefficients:")
print(theta)



def genetic_algorithm_cost(theta):
    return cost_function(theta, X, y)


bounds = [(-1, 1) for _ in range(X.shape[1])]
result = differential_evolution(genetic_algorithm_cost, bounds)
theta_genetic = result.x
predictions_genetic_proba = sigmoid(np.dot(X, theta_genetic))
accuracy_genetic, sensitivity_genetic, specificity_genetic, auc_genetic = performance_metrics(y,
                                                                                              predictions_genetic_proba)
print(
    f"Genetic Algorithm - Accuracy: {accuracy_genetic}, Sensitivity: {sensitivity_genetic}, Specificity: {specificity_genetic}, AUC: {auc_genetic}")
print("Genetic Algorithm Coefficients:")
print(theta_genetic)



def simulated_annealing_cost(theta):
    return cost_function(theta, X, y)


bounds = [(-1, 1) for _ in range(X.shape[1])]
result = dual_annealing(simulated_annealing_cost, bounds)
theta_annealing = result.x
predictions_annealing_proba = sigmoid(np.dot(X, theta_annealing))
accuracy_annealing, sensitivity_annealing, specificity_annealing, auc_annealing = performance_metrics(y,
                                                                                                      predictions_annealing_proba)
print(
    f"Simulated Annealing - Accuracy: {accuracy_annealing}, Sensitivity: {sensitivity_annealing}, Specificity: {specificity_annealing}, AUC: {auc_annealing}")
print("Simulated Annealing Coefficients:")
print(theta_annealing)



def hill_climbing_cost(theta):
    return cost_function(theta, X, y)


def randomized_hill_climbing(initial_theta, iterations):
    best_theta = initial_theta
    best_cost = hill_climbing_cost(best_theta)

    for _ in range(iterations):
        candidate_theta = best_theta + np.random.normal(0, 0.1, size=best_theta.shape)
        candidate_cost = hill_climbing_cost(candidate_theta)

        if candidate_cost < best_cost:
            best_theta, best_cost = candidate_theta, candidate_cost

    return best_theta


initial_theta = np.zeros(X.shape[1])
iterations = 1000
theta_hill_climbing = randomized_hill_climbing(initial_theta, iterations)
predictions_hill_climbing_proba = sigmoid(np.dot(X, theta_hill_climbing))
accuracy_hill_climbing, sensitivity_hill_climbing, specificity_hill_climbing, auc_hill_climbing = performance_metrics(y,
                                                                                                                      predictions_hill_climbing_proba)
print(
    f"Randomized Hill Climbing - Accuracy: {accuracy_hill_climbing}, Sensitivity: {sensitivity_hill_climbing}, Specificity: {specificity_hill_climbing}, AUC: {auc_hill_climbing}")
print("Randomized Hill Climbing Coefficients:")
print(theta_hill_climbing)
