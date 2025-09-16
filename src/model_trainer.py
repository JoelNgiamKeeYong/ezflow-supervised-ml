# ===========================================================================================================================================
# 🤖 MODEL TRAINER
# ===========================================================================================================================================

import os
import time
import numpy as np
import joblib
from typing import Any, Dict, Optional
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from rich import print as rprint

# ===========================================================================================================================================
# 🤖 MAIN CLASS
# ===========================================================================================================================================
class ModelTrainer:
    """
    A unified model trainer for classification and regression tasks with support
    for GridSearchCV, RandomizedSearchCV, and Bayesian optimization.

    Parameters
    ----------
    task_type : str
        Type of task: "classification" or "regression".
    resampling_method : Optional[str], default=None
        Resampling strategy: "null", "SMOTE", "SMOTE-ENN".
    hyperparameter_search_type : str
        Search strategy: "grid", "random", "bayes".
    cv_folds : int
        Number of cross-validation folds.
    scoring_metric : str
        Metric to optimize (e.g., "accuracy", "r2").
    n_iter : int
        Number of iterations for "random" or "bayes" search.
    n_jobs : int
        Number of parallel jobs.
    random_state : int
        Random seed.
    model_dir : str
        Directory to save trained models.
    """

    ########################################################################################################################################
    ########################################################################################################################################
    # 🏗️ CLASS CONSTRUCTOR
    def __init__(
        self,
        task_type: str = "classification",
        resampling_method: Optional[str] = None,
        hyperparameter_search_type: str = "grid",
        cv_folds: int = 5,
        scoring_metric: str = "accuracy",
        n_iter: int = 10,
        n_jobs: int = -1,
        random_state: int = 42,
        model_dir: str = "models",
    ):
        self.task_type = task_type
        self.resampling_method = resampling_method
        self.hyperparameter_search_type = hyperparameter_search_type
        self.cv_folds = cv_folds
        self.scoring_metric = scoring_metric
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model_dir = model_dir

    ########################################################################################################################################
    ########################################################################################################################################
    # 🤖 TRAIN
    def train(self, models: dict, X_train, y_train):
        """
        Train models with hyperparameter tuning.

        Parameters
        ----------
        models : dict
            {
              "RandomForest": {
                  "model": RandomForestClassifier(),
                  "params_gscv": {...},
                  "params_rscv": {...},
                  "params_bayes": {...}
              }
            }
            
        Returns
        -------
        list of [model_name, best_model, training_time, model_size_kb]
        """
        print(f"   └── Initalizing the model training framework...")

        # Detecting task type
        rprint(f"   └── Task type: [yellow]{self.task_type.title()}[/yellow]")
        
        # Detect binary vs multiclass
        if self.task_type == "classification":
            n_classes = len(np.unique(y_train))
            is_binary = (n_classes == 2)
            print(f"   └── Detected {n_classes} classes → {'Binary' if is_binary else 'Multiclass'} classification")
        
        # Validate models and task type
        self._detect_task_type_and_validate_models(models=models)

        # Setting hyoperparameter search config
        rprint(f"   └── Hyperparameter search strategy: [yellow]{self.hyperparameter_search_type.title()}SearchCV[/yellow]")
        rprint(f"   └── Cross-validation folds: [yellow]{self.cv_folds}[/yellow]")
        rprint(f"   └── Scoring metric: [yellow]{self.scoring_metric}[/yellow]")
        rprint(f"   └── Maximum search iterations: [yellow]{self.n_iter if self.hyperparameter_search_type != 'grid' else 'N/A (GridSearch)'}[/yellow]")
        print(f"   └── Parallel jobs: {self.n_jobs}")
        print(f"   └── Random seed: {self.random_state}")
        print("   └── Framework for training is set — proceeding to train models...")

        # Store trained models
        trained_models = []

        # Train each model
        for model_name, model_info in models.items():
            try:
                print(f"\n   ⛏️  Training \033[1;38;5;214m{model_name}\033[0m model...")
                start_time = time.time()

                # Build pipeline
                steps = []
                # Add resampling method to pipeline if defined
                if self.task_type == "classification" and self.resampling_method:
                    steps.append(("sampler", self._choose_sampler()))
                # Add model to pipeline
                steps.append(("model", model_info["model"]))
                pipeline = Pipeline(steps)
                print(f"      └── Pipeline steps: {[name for name, _ in pipeline.steps]}")

                # Pick hyperparam config
                param_key = f"params_{self.hyperparameter_search_type}"
                if param_key not in model_info:
                    raise KeyError(f"Missing {param_key} in model config for {model_name}")
                param_config = model_info[param_key]

                # Create search CV
                search, n_candidates = self._create_search_cv(
                    model_pipeline=pipeline,
                    param_config=param_config
                )

                # Fit search
                print(f"      └── Starting hyperparameter search for {model_name}...")
                print(f"      └── Number of hyperparameter combinations: {n_candidates}")
                print(f"      └── Fitting {self.cv_folds} folds for each of {n_candidates} candidates, totalling {self.cv_folds * n_candidates} fits")  
                search.fit(X_train, y_train)

                # Training time
                training_time = time.time() - start_time
                print(f"      └── Completed in {training_time:.2f}s")

                # Extract best model
                best_model = search.best_estimator_
                best_params = {
                    (k.replace("model__", "")): (round(v, 3) if isinstance(v, float) else v)
                    for k, v in search.best_params_.items()
                }
                rprint(f"      └── Best params: [yellow]{best_params}[/yellow]")

                # Save model
                model_path = f"{self.model_dir}/{model_name.replace(' ', '_').lower()}_model.joblib"
                joblib.dump(best_model, model_path)
                model_size_kb = round(os.path.getsize(model_path) / 1024, 2)
                rprint(f"      └── Saved to '{model_path}' ({model_size_kb} KB)")

                trained_models.append([model_name, best_model, training_time, model_size_kb])

            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")

        return trained_models
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🤖 DETECT TASK TYPE AND VALIDATE MODELS
    def _detect_task_type_and_validate_models(self, models):
        """
        Validate that all models are appropriate for the specified task type.

        Args:
            models (dict): Dict of models to validate.
            
        Returns:
            None.

        Raises:
            ValueError: If model types don't match the task or task_type is invalid.
        """
        print(f"   └── Validating candidate models...")

        # Detect task type from target variable
        if self.task_type not in ["classification", "regression"]:
            raise ValueError(f"❌  task_type must be 'classification' or 'regression', got '{self.task_type}'")

        # Validate models
        for name, model_dict in models.items():
            model = model_dict["model"]
            if self.task_type == "classification" and not is_classifier(model):
                raise ValueError(f"    └── ❌  Model '{name}' is not a classifier but task is classification.")
            elif self.task_type == "regression" and not is_regressor(model):
                raise ValueError(f"    └── ❌  Model '{name}' is not a regressor but task is regression.")

        print(f"   └── Candidate models provided are valid for {self.task_type} task")

    ########################################################################################################################################
    ########################################################################################################################################
    # 🤖 CREATE SEARCH CV
    def _create_search_cv(self, model_pipeline, param_config):
        """
        Factory method to create a hyperparameter search object: GridSearchCV, RandomizedSearchCV, or BayesSearchCV.

        This method also calculates the total number of hyperparameter combinations for logging purposes.

        Parameters
        ----------
        model_pipeline : sklearn.pipeline.Pipeline
            The pipeline containing preprocessing steps and the estimator to tune.
        param_config : dict
            Dictionary of hyperparameters:
            - For GridSearchCV: keys map to lists of values.
            - For RandomizedSearchCV: keys map to lists of values (sampling is random).
            - For BayesSearchCV: keys map to continuous ranges or distributions.

        Returns
        -------
        search_cv : GridSearchCV, RandomizedSearchCV, or BayesSearchCV object
            Configured hyperparameter search object ready to fit.
        n_candidates : int
            Number of hyperparameter combinations that will be evaluated (useful for logging).
        """

        # Compute number of candidates for logging
        if self.hyperparameter_search_type == "grid":
            # Product of lengths of lists in param_config
            n_candidates = 1
            for values in param_config.values():
                n_candidates *= len(values) if isinstance(values, (list, tuple)) else 1
        elif self.hyperparameter_search_type == "random":
            n_candidates = min(self.n_iter, 1 if not param_config else self.n_iter)  # Random samples up to n_iter
        elif self.hyperparameter_search_type == "bayes":
            n_candidates = self.n_iter
        else:
            raise ValueError("hyperparameter_search_type must be one of {'grid', 'random', 'bayes'}")

        # Create the appropriate search object
        if self.hyperparameter_search_type == "grid":
            search_cv = GridSearchCV(
                estimator=model_pipeline,
                param_grid=param_config,
                cv=self.cv_folds,
                scoring=self.scoring_metric,
                n_jobs=self.n_jobs,
                verbose=0
            )
        elif self.hyperparameter_search_type == "random":
            search_cv = RandomizedSearchCV(
                estimator=model_pipeline,
                param_distributions=param_config,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring=self.scoring_metric,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=0
            )
        elif self.hyperparameter_search_type == "bayes":
            search_cv = BayesSearchCV(
                estimator=model_pipeline,
                search_spaces=param_config,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring=self.scoring_metric,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=0
            )

        return search_cv, n_candidates
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🤖 CHOOOSE SAMPLER
    def _choose_sampler(self):
        """
        Select the resampling strategy for imbalanced classification tasks.

        This method configures and returns the appropriate sampler based on
        the `resampling_method` string in the config.

        Strategies
        ----------
        - null:
            No resampling (returns None).
        - SMOTE:
            Synthetic Minority Over-sampling Technique.
            Creates synthetic samples for the minority class to balance the dataset.
        - SMOTE-ENN:
            A hybrid method combining SMOTE with Edited Nearest Neighbours (ENN).
            SMOTE oversamples the minority class, while ENN cleans noisy or
            ambiguous samples from the majority class.

        Returns
        -------
        sampler : imblearn BaseSampler or None
            The configured sampler instance, or None if resampling_method is null.
        """

        if self.resampling_method is None or str(self.resampling_method).lower() == "null":
            return None

        if str(self.resampling_method).upper() == "SMOTE":
            return SMOTE(
                sampling_strategy="auto",
                k_neighbors=5,
                random_state=self.random_state
            )

        if str(self.resampling_method).upper() == "SMOTE-ENN":
            return SMOTEENN(
                smote=SMOTE(
                    sampling_strategy="auto",
                    k_neighbors=5,
                    random_state=self.random_state
                ),
                enn=EditedNearestNeighbours(
                    sampling_strategy="majority",
                    n_neighbors=5,
                    kind_sel="mode"
                )
            )

        raise ValueError(f"    └── ❌  Unknown resampling_method: {self.resampling_method}")
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🤖 PARAM COMBINATIONS
    def _param_combinations(self, param_config):
        """
        Generate all possible hyperparameter combinations from a parameter grid.

        This method is useful for:
        - Counting the number of candidate models that will be evaluated 
            in GridSearchCV.
        - Inspecting the actual parameter sets before running a search.

        Notes
        -----
        - Works only when `param_config` is a dictionary with parameter names as keys
        and **iterable values** (e.g., lists, tuples, or ranges).
        - If a value is not iterable, it will be wrapped into a single-element list.
        - For empty configs, returns an empty list.

        Parameters
        ----------
        param_config : dict
            Dictionary mapping parameter names (str) to a list/tuple/range of values.

            Example
            -------
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [3, 5, 10]
            }

        Returns
        -------
        List[dict]
            A list of dictionaries, each representing one unique hyperparameter combination.

            Example output
            --------------
            [
                {"model__n_estimators": 50, "model__max_depth": 3},
                {"model__n_estimators": 50, "model__max_depth": 5},
                {"model__n_estimators": 50, "model__max_depth": 10},
                {"model__n_estimators": 100, "model__max_depth": 3},
                {"model__n_estimators": 100, "model__max_depth": 5},
                {"model__n_estimators": 100, "model__max_depth": 10}
            ]
        """
        from itertools import product

        if not param_config:
            return []

        keys = list(param_config.keys())
        values = [
            v if isinstance(v, (list, tuple, range)) else [v] 
            for v in (param_config[k] for k in keys)
        ]
        return [dict(zip(keys, combination)) for combination in product(*values)]