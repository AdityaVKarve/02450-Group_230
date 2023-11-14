import nicotine_df
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB

df = nicotine_df.df
X = df[nicotine_df.predictors].to_numpy()  # personal data
y = df[nicotine_df.predicates].to_numpy().squeeze()  # nicotine

# define models
models = {
    'Baseline': DummyClassifier(strategy='most_frequent', random_state=42),
    'Multinomial Logistic Regression': LogisticRegression(multi_class='multinomial', max_iter=5000, random_state=42),
    'Multinomial Naive Bayes': MultinomialNB()
}

# Parameter grids for hyperparameter tuning
param_grid_nb = {'alpha': np.arange(10, 30, 0.5)}
param_grid_lr = {'C': np.arange(0.01, 0.3, 0.01)}
print(param_grid_nb)
print(param_grid_lr)

# Outer cross-validation
K = 10
outer_cv = KFold(n_splits=K, shuffle=True, random_state=42)
outer_fold = 1

# Initialize variables
results = []
nb_train_error = np.empty((K+1, 1))
nb_test_error = np.empty((K+1, 1))
lr_train_error = np.empty((K+1, 1))
lr_test_error = np.empty((K+1, 1))
b_train_error = np.empty((K+1, 1))
b_test_error = np.empty((K+1, 1))

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    internal_cross_validation = 10

    # Inner cross-validation for Naive Bayes
    inner_cv_nb = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search_nb = GridSearchCV(estimator=models['Multinomial Naive Bayes'], param_grid=param_grid_nb, cv=inner_cv_nb)
    grid_search_nb.fit(X_train, y_train)
    best_nb = grid_search_nb.best_estimator_
    alpha_nb = grid_search_nb.best_params_['alpha']
    print(f"The best alpha found is: {alpha_nb}")
    # compute squared error with all features selected (no feature selection)
    nb_train_error[outer_fold] = np.square(y_train - grid_search_nb.predict(X_train)).sum() / y_train.shape[0]
    nb_test_error[outer_fold] = np.square(y_test - grid_search_nb.predict(X_test)).sum() / y_test.shape[0]

    # Inner cross-validation for Logistic Regression
    inner_cv_lr = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search_lr = GridSearchCV(estimator=models['Multinomial Logistic Regression'], param_grid=param_grid_lr, cv=inner_cv_lr)
    grid_search_lr.fit(X_train, y_train)
    best_lr = grid_search_lr.best_estimator_
    lambda_lr = grid_search_lr.best_params_['C']
    print(f"The best C found is: {lambda_lr}")
    # compute squared error with all features selected (no feature selection)
    lr_train_error[outer_fold] = np.square(y_train - grid_search_lr.predict(X_train)).sum() / y_train.shape[0]
    lr_test_error[outer_fold] = np.square(y_test - grid_search_lr.predict(X_test)).sum() / y_test.shape[0]

    baseline = models['Baseline'].fit(X_train, y_train)
    # compute squared error with all features selected (no feature selection)
    b_train_error[outer_fold] = np.square(y_train - baseline.predict(X_train)).sum() / y_train.shape[0]
    b_test_error[outer_fold] = np.square(y_test - baseline.predict(X_test)).sum() / y_test.shape[0]

    # Model fitting and evaluation
    best_models = [best_nb, best_lr, models['Baseline']]
    error_rates = []

    results.append([outer_fold, alpha_nb, nb_test_error[outer_fold], lambda_lr, lr_test_error[outer_fold], b_test_error[outer_fold]])
    outer_fold += 1

# Creating a table
print("Outer fold | Naive Bayes | Logistic Regression | Baseline")
print("i | alpha i & Test error i | C i & Test error i | Test error i")
for row in results:
    print(f"{row[0]} | {row[1]} & {row[2][0]:.1f}% | {row[3]**(-1):.1f} & {row[4][0]:.1f}% | {row[5][0]:.1f}%")
