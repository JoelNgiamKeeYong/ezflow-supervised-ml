# ===========================================================================================================================================
# 🤖 MODEL TRAINER
# ===========================================================================================================================================

import os
import time
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

# ===========================================================================================================================================
# 🤖 MAIN CLASS
# ===========================================================================================================================================
class ModelTrainer:
    """
    A unified model trainer for classification and regression tasks with
    support for GridSearchCV, RandomizedSearchCV, and Bayesian optimization.

    Parameters
    ----------
    task_type : str
        "classification" or "regression".
    search_type : str, default="random"
        One of {"grid", "random", "bayes"}.
    use_smote_enn : bool, default=True
        Only applies if task_type="classification".
    cv_folds : int, default=5
        Number of cross-validation folds.
    scoring_metric : str, default="f1" (classification) or "neg_root_mean_squared_error" (regression).
    n_iter : int, default=50
        Iterations for randomized/bayes search.
    n_jobs : int, default=-1
        Number of parallel jobs.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    ########################################################################################################################################
    ########################################################################################################################################
    # 🏗️ CLASS CONSTRUCTOR
    def __init__(
        self,
        task_type: str,
        search_type: str = "grid",
        use_smote_enn: bool = False,
        cv_folds: int = 5,
        scoring_metric: str = None,
        n_iter: int = 50,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.search_type = search_type
        self.use_smote_enn = use_smote_enn
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Default scoring depending on task type
        if scoring_metric is None:
            if self.task_type == "classification":
                self.scoring_metric = "f1"
            elif self.task_type == "regression":
                self.scoring_metric = "neg_root_mean_squared_error"
            else:
                raise ValueError("task_type must be 'classification' or 'regression'")
        else:
            self.scoring_metric = scoring_metric

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
        trained_models = []
        print(f"   └── Initalizing the model training framework...")
        print(f"   └── Task type: {self.task_type.title()}")
        print(f"   └── Hyperparameter search strategy: {self.search_type.title()}SearchCV")
        print(f"   └── Cross-validation folds: {self.cv_folds}")
        print(f"   └── Scoring metric: {self.scoring_metric}")
        print(f"   └── Maximum search iterations: {self.n_iter if self.search_type != 'grid' else 'N/A (GridSearch)'}")
        print(f"   └── Parallel jobs: {self.n_jobs}")
        print(f"   └── Random seed: {self.random_state}")
        print("   └── Framework for training is set — proceeding to train models...")

        for model_name, model_info in models.items():
            try:
                print(f"\n   ⛏️  Training {model_name}...")
                start_time = time.time()

                # Build pipeline
                steps = []
                if self.task_type == "classification":
                    steps.append(("sampler", self._choose_sampler()))
                steps.append(("model", model_info["model"]))
                pipeline = Pipeline(steps)
                print(f"      └── Pipeline steps: {[name for name, _ in pipeline.steps]}")

                # Pick hyperparam config
                param_key = f"params_{self.search_type}"
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
        if self.search_type == "grid":
            # Product of lengths of lists in param_config
            n_candidates = 1
            for values in param_config.values():
                n_candidates *= len(values) if isinstance(values, (list, tuple)) else 1
        elif self.search_type == "random":
            n_candidates = min(self.n_iter, 
                            1 if not param_config else
                            self.n_iter)  # Random samples up to n_iter
        elif self.search_type == "bayes":
            n_candidates = self.n_iter
        else:
            raise ValueError("search_type must be one of {'grid', 'random', 'bayes'}")

        # Create the appropriate search object
        if self.search_type == "grid":
            search_cv = GridSearchCV(
                estimator=model_pipeline,
                param_grid=param_config,
                cv=self.cv_folds,
                scoring=self.scoring_metric,
                n_jobs=self.n_jobs,
                verbose=0
            )
        elif self.search_type == "random":
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
        elif self.search_type == "bayes":
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
                random_state=self.random_state
            )

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