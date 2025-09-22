# config_models.py
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

models = {
    ########################################################################################################################################
    ########################################################################################################################################
    # 🟢 LOGISTIC REGRESSION
    # "LogisticRegression": {
    #     "model": LogisticRegression(max_iter=10, random_state=42),
    #     "params_grid": {
    #         "model__C": [0.01],
    #         "model__penalty": ["l2"],
    #     },
    #     "params_random": {
    #         "model__C": [0.01, 0.1, 1, 10, 100],
    #         "model__penalty": ["l2"],
    #     },
    #     "params_bayes": {
    #         "model__C": (1e-3, 1e3, "log-uniform"),
    #     },
    # },
    ########################################################################################################################################
    ########################################################################################################################################
    # 🟢 RANDOM FOREST REGRESSOR
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params_grid": {
            "model__n_estimators": [50],
            "model__max_depth": [5],
            "model__min_samples_split": [2],
        },
        "params_random": {
            "model__n_estimators": [50, 100, 200, 300],
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
        },
        "params_bayes": {
            "model__n_estimators": (50, 300),
            "model__max_depth": (2, 20),
            "model__min_samples_split": (2, 10),
        },
    },
    ########################################################################################################################################
    ########################################################################################################################################
    # 🔴 OTHER MODELS
    # ...
}

###########################################################################
# 🟢 DEBUG MODE MODELS
###########################################################################

# Debug classification models (fast & simple)
models_debug_classification = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=100, random_state=42),
        "params_grid": {
            "model__C": [1.0],
            "model__penalty": ["l2"],
        },
        "params_random": {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
        },
        "params_bayes": {
            "model__C": (1e-3, 1e3, "log-uniform"),
        },
    }
}

# Debug regression models (fast & simple)
models_debug_regression = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params_grid": {
            "model__fit_intercept": [True, False],
            "model__positive": [True, False],
        },
        "params_random": {
            "model__fit_intercept": [True, False],
            "model__positive": [True, False],
        },
        "params_bayes": {
            # Bayesian search usually expects continuous ranges,
            # but LinearRegression has very few tunable params.
            # We'll still keep discrete options for consistency.
            "model__fit_intercept": [True, False],
            "model__positive": [True, False],
        },
    }
}