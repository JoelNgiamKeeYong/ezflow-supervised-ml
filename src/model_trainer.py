# ===========================================================================================================================================
# 🤖 MODEL TRAINER
# ===========================================================================================================================================

import os
import time
import joblib
from typing import Any, Dict
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

from sklearn.utils.multiclass import type_of_target
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from rich import print as rprint

# ===========================================================================================================================================
# 🤖 MAIN CLASS
# ===========================================================================================================================================
class ModelTrainer:
    """
    A unified model trainer for classification and regression tasks with support for GridSearchCV, RandomizedSearchCV, and Bayesian optimization.

    Attributes:
    -----------
    config : Dict[str, Any], optional
            Configuration dictionary containing parameters for cleaning rules, by default None
    """

    ########################################################################################################################################
    ########################################################################################################################################
    # 🏗️ CLASS CONSTRUCTOR
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the ModelTrainer class.

        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration dictionary containing parameters for model training rules, by default None
        """
        self.config = config or {}

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
        rprint(f"   └── Task type: [magenta]{self.config["task_type"].title()}[/magenta]")
        
        # Validate models and task type
        self._detect_task_type_and_validate_models(models=models)

        # Setting hyoperparameter search config
        rprint(f"   └── Hyperparameter search strategy: [magenta]{self.config["search_type"].title()}SearchCV[/magenta]")
        rprint(f"   └── Cross-validation folds: [magenta]{self.config["cv_folds"]}[/magenta]")
        rprint(f"   └── Scoring metric: [magenta]{self.config["scoring_metric"]}[/magenta]")
        rprint(f"   └── Maximum search iterations: [magenta]{self.config["n_iter"] if self.config["search_type"] != 'grid' else 'N/A (GridSearch)'}[/magenta]")
        print(f"   └── Parallel jobs: {self.config["n_jobs"]}")
        print(f"   └── Random seed: {self.config["random_state"]}")
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
                if self.config["task_type"] == "classification":
                    steps.append(("sampler", self._choose_sampler()))
                steps.append(("model", model_info["model"]))
                pipeline = Pipeline(steps)
                print(f"      └── Pipeline steps: {[name for name, _ in pipeline.steps]}")

                # Pick hyperparam config
                param_key = f"params_{self.config["search_type"]}"
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
                print(f"      └── Fitting {self.config["cv_folds"]} folds for each of {n_candidates} candidates, totalling {self.config["cv_folds"] * n_candidates} fits")  
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
                print(f"      └── Best params: {best_params}")

                # Save model
                model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
                joblib.dump(best_model, model_path)
                model_size_kb = round(os.path.getsize(model_path) / 1024, 2)
                print(f"      └── Saved to {model_path} ({model_size_kb} KB)")

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
        if self.config["task_type"] not in ["classification", "regression"]:
            raise ValueError(f"❌  task_type must be 'classification' or 'regression', got '{self.config["task_type"]}'")

        # Validate models
        for name, model_dict in models.items():
            model = model_dict["model"]
            if self.config["task_type"] == "classification" and not is_classifier(model):
                raise ValueError(f"    └── ❌  Model '{name}' is not a classifier but task is classification.")
            elif self.config["task_type"] == "regression" and not is_regressor(model):
                raise ValueError(f"    └── ❌  Model '{name}' is not a regressor but task is regression.")

        print(f"   └── Candidate models provided are valid for {self.config["task_type"]} task")

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
        if self.config["search_type"] == "grid":
            # Product of lengths of lists in param_config
            n_candidates = 1
            for values in param_config.values():
                n_candidates *= len(values) if isinstance(values, (list, tuple)) else 1
        elif self.config["search_type"] == "random":
            n_candidates = min(self.config["n_iter"], 1 if not param_config else self.config["n_iter"])  # Random samples up to n_iter
        elif self.config["search_type"] == "bayes":
            n_candidates = self.config["n_iter"]
        else:
            raise ValueError("search_type must be one of {'grid', 'random', 'bayes'}")

        # Create the appropriate search object
        if self.config["search_type"] == "grid":
            search_cv = GridSearchCV(
                estimator=model_pipeline,
                param_grid=param_config,
                cv=self.config["cv_folds"],
                scoring=self.config["scoring_metric"],
                n_jobs=self.config["n_jobs"],
                verbose=0
            )
        elif self.config["search_type"] == "random":
            search_cv = RandomizedSearchCV(
                estimator=model_pipeline,
                param_distributions=param_config,
                n_iter=self.config["n_iter"],
                cv=self.config["cv_folds"],
                scoring=self.config["scoring_metric"],
                n_jobs=self.config["n_jobs"],
                random_state=self.config["random_state"],
                verbose=0
            )
        elif self.search_type == "bayes":
            search_cv = BayesSearchCV(
                estimator=model_pipeline,
                search_spaces=param_config,
                n_iter=self.config["n_iter"],
                cv=self.config["cv_folds"],
                scoring=self.config["scoring_metric"],
                n_jobs=self.config["n_jobs"],
                random_state=self.config["random_state"],
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
        whether `use_smote_enn` is enabled.

        Strategies
        ----------
        - SMOTE:
            Synthetic Minority Over-sampling Technique.
            Creates synthetic samples for the minority class to balance the dataset.
        - SMOTEENN:
            A hybrid method combining SMOTE with Edited Nearest Neighbours (ENN).
            SMOTE oversamples the minority class, while ENN cleans noisy or
            ambiguous samples from the majority class.

        Notes
        -----
        - Both methods rely on nearest neighbors to generate or clean samples.
        - The `random_state` ensures reproducibility.
        - Currently tailored for binary or multiclass imbalanced classification.

        Returns
        -------
        sampler : imblearn BaseSampler
            The configured sampler instance:
            - `SMOTE(...)` if `use_smote_enn` is False.
            - `SMOTEENN(...)` if `use_smote_enn` is True.

        Examples
        --------
        >>> self.use_smote_enn = False
        >>> sampler = self._choose_sampler()
        >>> type(sampler)
        <class 'imblearn.over_sampling._smote.base.SMOTE'>

        >>> self.use_smote_enn = True
        >>> sampler = self._choose_sampler()
        >>> type(sampler)
        <class 'imblearn.combine.SMOTEENN'>
        """
        if not self.use_smote_enn:
            return SMOTE(
                sampling_strategy="auto",
                k_neighbors=5,
                random_state=self.config["random_state"]
            )

        return SMOTEENN(
            smote=SMOTE(
                sampling_strategy="auto",
                k_neighbors=5,
                random_state=self.config["random_state"]
            ),
            enn=EditedNearestNeighbours(
                sampling_strategy="majority",
                n_neighbors=5,
                kind_sel="mode"
            )
        )
    
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