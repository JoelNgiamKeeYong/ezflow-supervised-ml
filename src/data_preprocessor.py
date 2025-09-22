# ===========================================================================================================================================
# ⚙️ DATA PREPROCESSOR CLASS
# ===========================================================================================================================================

# Standard library imports
import re
from typing import Any, Dict, List, Optional, Tuple

# Related third-party imports
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from kmodes.kmodes import KModes
from rich import print as rprint
import inspect

# ===========================================================================================================================================
# ⚙️ MAIN CLASS
# ===========================================================================================================================================
class DataPreprocessor:
    """
    A class used to preprocess data: handle missing values, scale numerical features,
    and one-hot encode categorical features.

    Attributes:
    -----------
    config : Dict[str, Any], optional
        Configuration dictionary containing lists of numerical and categorical features.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline object for transforming the data.
    """

    ########################################################################################################################################
    ########################################################################################################################################
    # 🏗️ CLASS CONSTRUCTOR
    def __init__(self):
        """
        Initializes the DataPreprocessor class.
        """
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        self.preprocessor: Optional[ColumnTransformer] = None

    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ CREATE PREPROCESSOR PIPELINE
    def _create_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Automatically detects numerical and categorical columns and creates a ColumnTransformer
        with pipelines for missing value imputation, scaling, and encoding.

        Allows per-column custom imputers/scalers/encoders, falling back to defaults if not defined.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        ColumnTransformer
            A preprocessor for the DataFrame.
        """
        print("   └── Detecting numerical and categorical columns...")

        self.numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"   └── Numerical columns ({len(self.numerical_features)}): {self.numerical_features}")
        print(f"   └── Categorical columns ({len(self.categorical_features)}): {self.categorical_features}")

        # ================================================================================================================================
        # ✍️ Define your custom transformers here (manual config section)
        # ================================================================================================================================
        print("   └── Defining preprocessing pipeline...")
        # ================================================================================================================================
        # ================================================================================================================================
        # 🔵 Custom numerical imputers:
        custom_num_imputers = {
            # "age": SimpleImputer(missing_values=np.nan, strategy="median")
            # "salary": KNNImputer(missing_values=np.nan, n_neighbors=5, weights="uniform", metric="nan_euclidean"),
            # "revenue": IterativeImputer(max_iter=10, random_state=42),
            # --- ⬇️ INSERT BELOW ---
            "n_tokens_title": SimpleImputer(strategy="median")
        }
        # ================================================================================================================================
        # ================================================================================================================================
        # 🔴 Custom numerical scalers:
        custom_num_scalers = {
            # "salary": MinMaxScaler()
            # --- ⬇️ INSERT BELOW ---

        }
        # ================================================================================================================================
        # ================================================================================================================================
        # 🔵 Custom categorical imputers:
        custom_cat_imputers = {
            # "gender": SimpleImputer(strategy="constant", fill_value="missing").
            # "weekday": KModesImputer(columns_for_clustering=['data_channel'], n_clusters=5, random_state=42),
            # --- ⬇️ INSERT BELOW ---
            "weekday": KModesImputer(columns_for_clustering=['data_channel'], n_clusters=5, random_state=42)
        }
        # ================================================================================================================================
        # ================================================================================================================================
        # 🟢 Custom categorical encoders:
        custom_cat_encoders = {
            # "city": OrdinalEncoder()
            # --- ⬇️ INSERT BELOW ---

        }

        # Defaults if column not listed above
        default_num_imputer = SimpleImputer(strategy="mean")
        default_num_scaler = StandardScaler()
        default_cat_imputer = SimpleImputer(strategy="most_frequent")
        default_cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        transformers = []

        # Numeric transformers
        num_default_cols = []
        for col in self.numerical_features:
            imputer = custom_num_imputers.get(col, default_num_imputer)
            scaler = custom_num_scalers.get(col, default_num_scaler)

            if col in custom_num_imputers or col in custom_num_scalers:
                transformers.append((
                    f"num_pipe_{col.lower().replace(' ', '_')}",
                    Pipeline([
                        ("imputer", imputer),
                        ("scaler", scaler)
                    ]),
                    [col]
                ))
            else:
                num_default_cols.append(col)

        if num_default_cols:
            transformers.append((
                "num_pipe_default",
                Pipeline([
                    ("imputer", default_num_imputer),
                    ("scaler", default_num_scaler)
                ]),
                num_default_cols
            ))

        # Categorical transformers
        cat_default_cols = []
        for col in self.categorical_features:
            imputer = custom_cat_imputers.get(col, default_cat_imputer)
            encoder = custom_cat_encoders.get(col, default_cat_encoder)

            input_cols = [col]

            # if imputer has clustering columns, include them
            if hasattr(imputer, "columns_for_clustering"):
                input_cols += [c for c in imputer.columns_for_clustering if c != col]

            if col in custom_cat_imputers or col in custom_cat_encoders:
                transformers.append((
                    f"cat_pipe_{col.lower().replace(' ', '_')}",
                    Pipeline([
                        ("imputer", imputer),
                        ("encoder", encoder)
                    ]),
                    input_cols
                ))
            else:
                cat_default_cols.append(col)

        if cat_default_cols:
            transformers.append((
                "cat_pipe_default",
                Pipeline([
                    ("imputer", default_cat_imputer),
                    ("encoder", default_cat_encoder)
                ]),
                cat_default_cols
            ))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            # n_jobs=-1
        )

        # # Print ASCII-style pipeline diagram in CLI
        for name, transformer, cols in preprocessor.transformers:
            if name == "remainder" and transformer == "passthrough":
                cols_formatted = "[" + ", ".join(f"\033[38;5;214m{col}\033[0m" for col in cols) + "]"
                print(f"\n   🔧 \033[1;38;5;214m{name}\033[0m (passthrough) → {cols_formatted}")
            else:
                cols_formatted = "[" + ", ".join(f"\033[38;5;214m{col}\033[0m" for col in cols) + "]"
                print(f"\n   🔧 \033[1;38;5;214m{name}\033[0m → {cols_formatted}")
                
                if hasattr(transformer, "named_steps"):
                    for step_name, step in transformer.named_steps.items():
                        manual_args = {}
                        
                        # Detect manually-set arguments only
                        if hasattr(step, "get_params"):
                            sig = inspect.signature(step.__class__.__init__)
                            for param_name, param in sig.parameters.items():
                                if param_name == "self":
                                    continue
                                val = getattr(step, param_name, param.default)
                                if val != param.default:
                                    manual_args[param_name] = val

                        if manual_args:
                            rprint(f"      └── {step_name}: [magenta]{step.__class__.__name__}[/magenta] [cyan]{manual_args}[/cyan]")
                        else:
                            rprint(f"      └── {step_name}: [magenta]{step.__class__.__name__}[/magenta]")

                else:
                    # Standalone transformer (not in a pipeline)
                    manual_args = {}
                    if hasattr(transformer, "get_params"):
                        sig = inspect.signature(transformer.__class__.__init__)
                        for param_name, param in sig.parameters.items():
                            if param_name == "self":
                                continue
                            val = getattr(transformer, param_name, param.default)
                            if val != param.default:
                                manual_args[param_name] = val
                    if manual_args:
                        rprint(f"      └── [magenta]{transformer.__class__.__name__}[/magenta] [cyan]{manual_args}[/cyan]")
                    else:
                        rprint(f"      └── [magenta]{transformer.__class__.__name__}[/magenta]")

        return preprocessor

    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ FIT & TRANSFORM
    def fit_transform(self, df: pd.DataFrame, show: bool = False) -> pd.DataFrame:
        """
        Detect columns, fit the preprocessor, and transform the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to preprocess.
        show : bool, optional
            If True, displays the transformed DataFrame, by default False.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.
        """
        self.preprocessor = self._create_preprocessor(df)

        print("\n   └── Fitting and transforming the Training DataFrame...")
        transformed_array = self.preprocessor.fit_transform(df)
        feature_names = self._get_feature_names()
        transformed_df = pd.DataFrame(transformed_array, columns=feature_names, index=df.index)

        if show:
            print("\n🫧 Preprocessed DataFrame:")
            display(transformed_df.head())

        return transformed_df

    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ TRANSFORM ONLY
    def transform(self, df: pd.DataFrame, show: bool = False) -> pd.DataFrame:
        """
        Transform a DataFrame using an already-fitted preprocessor.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to preprocess.
        show : bool, optional
            If True, displays the transformed DataFrame, by default False.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        if not self.preprocessor:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        print("   └── Transforming the Test DataFrame using the fitted preprocessor...")
        transformed_array = self.preprocessor.transform(df)
        feature_names = self._get_feature_names()
        transformed_df = pd.DataFrame(transformed_array, columns=feature_names, index=df.index)

        print("   └── Data successfully preprocessed")

        if show:
            print("\n🫧 Transformed DataFrame:")
            display(transformed_df.head())

        return transformed_df

    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ GET FEATURE NAMES
    def _get_feature_names(self) -> List[str]:
        """
        Get column names after preprocessing, including expanded one-hot encoded columns,
        and respecting the order of the ColumnTransformer.

        Returns
        -------
        List[str]
            List of feature names after transformation.
        """
        feature_names: List[str] = []

        for name, trans, cols in self.preprocessor.transformers_:
            if name == "remainder" and trans == "passthrough":
                # passthrough columns keep original names
                feature_names.extend(cols)
            elif hasattr(trans, "get_feature_names_out"):
                # scikit-learn >= 1.0
                feature_names.extend(trans.get_feature_names_out(cols))
            elif hasattr(trans, "named_steps"):
                # Pipelines inside ColumnTransformer
                last_step = list(trans.named_steps.values())[-1]
                if hasattr(last_step, "get_feature_names_out"):
                    feature_names.extend(last_step.get_feature_names_out(cols))
                else:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)

        return feature_names
    
    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ GET MANUAL ARGS
    def get_manual_args(obj):
        """
        Returns a dict of the arguments that were manually set during initialization,
        excluding default values.
        """
        cls = obj.__class__
        sig = inspect.signature(cls.__init__)
        manual_args = {}
        
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            current_val = getattr(obj, name, param.default)
            if current_val != param.default:
                manual_args[name] = current_val
        
        return manual_args
    
    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ REMOVE MISSING VALUES
    def remove_missing_values(
        self,
        df: pd.DataFrame,
        how: str = "any",
        subset: Optional[List[str]] = None,
        placeholders: Optional[Dict[str, List[Any]]] = None
    ) -> pd.DataFrame:
        """
        Removes rows with missing values (NaN or placeholders) from the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to clean.
        how : {"any", "all"}, default="any"
            - "any": drop rows with at least one NaN/placeholder
            - "all": drop rows only if all values are NaN/placeholder
        subset : list of str, optional
            Columns to check. By default, checks all columns.
        placeholders : dict, optional
            Dictionary mapping columns to placeholder values.
            Example: {"age": ["unknown", "999"], "income": ["N/A"], "gender": ["unknown"]}

        Returns
        -------
        pd.DataFrame
            DataFrame with rows removed.
        """
        df_cleaned = df.copy()
        total_before = len(df)

        print(f"\n   └── Removing rows with missing values...")
        print(f"       └── Strategy: how='{how}', subset={subset if subset else 'ALL columns'}")

        # Step 1: Replace placeholders with NaN
        if placeholders:
            for col, vals in placeholders.items():
                if col in df_cleaned.columns:
                    print(f"       └── Converting placeholders {vals} → NaN in column '{col}'")
                    df_cleaned[col] = df_cleaned[col].replace(vals, pd.NA)

        # Step 2: Drop rows with missing values
        df_cleaned = df_cleaned.dropna(how=how, subset=subset)

        total_after = len(df_cleaned)
        total_removed = total_before - total_after
        perc_removed = (total_removed / total_before * 100) if total_before > 0 else 0

        print(f"       └── Rows before: {total_before:,}")
        print(f"       └── Rows after:  {total_after:,}")
        print(f"       └── Removed {total_removed:,} rows ({perc_removed:.2f}%)")

        return df_cleaned   

    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ SPLIT DATASET
    def split_dataset(
        self,
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.2,
        stratify: bool = False,
        random_state: Optional[int] = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits a dataset into training and test sets, with optional stratified sampling.

        Parameters
        ----------
        df : pd.DataFrame
            The full dataset containing features and target.
        target : str
            The name of the target column.
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split.
        stratify : bool, default=False
            Whether to perform stratified sampling based on the target variable.
            Useful for classification tasks to maintain class distribution.
        random_state : int, optional, default=42
            Random seed for reproducibility.

        Returns
        -------
        X_train : pd.DataFrame
            Training feature set.
        X_test : pd.DataFrame
            Test feature set.
        y_train : pd.Series
            Training target values.
        y_test : pd.Series
            Test target values.
        """
        print("   └── Splitting the dataset...")

        if target not in df.columns:
            raise ValueError(f"   └── ❌ Target column '{target}' not found in DataFrame.")

        X = df.drop(columns=[target])
        y = df[target]

        stratify_col = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state
        )

        train_rows, train_cols = X_train.shape
        test_rows, test_cols = X_test.shape
        rprint(f"   └── Dataset split completed: {len(X_train):,} train samples, {len(X_test):,} test samples.")
        print(f"   └── Training set shape: ({train_rows:,}, {train_cols:,})")
        print(f"   └── Test set shape:     ({test_rows:,}, {test_cols:,})")

        if stratify:
            print(f"   └── Stratified on target='{target}'")

        return X_train, X_test, y_train, y_test

# ===========================================================================================================================================
# ⚙️ OUTLIER CLIPPER
# ===========================================================================================================================================

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer to handle outliers
    by capping values using different statistical rules.

    Supported Methods
    -----------------
    - "zscore"   : Clip values beyond mean ± threshold*std.
    - "iqr"      : Clip using Tukey’s rule (Q1 - k*IQR, Q3 + k*IQR).
    - "quantile" : One-sided clipping at a chosen quantile.
                   (threshold > 0.5 clips the upper tail,
                    threshold < 0.5 clips the lower tail).
    - "mad"      : Clip based on Modified Z-Score with Median Absolute Deviation.
    - "domain"   : Clip to user-provided domain limits.

    Parameters
    ----------
    method : str, default="zscore"
        Outlier detection/clipping method.
    threshold : float, default=3.0
        Threshold value depending on the method:
        - zscore  : # of standard deviations from the mean
        - iqr     : IQR multiplier (e.g. 1.5, 3.0)
        - quantile: quantile cutoff (0 < threshold < 1)
        - mad     : modified z-score cutoff
    domain_limits : dict, optional
        Required if method="domain".
        Format: {"col_name": (lower, upper)}
    """

    def __init__(self, method="zscore", threshold=3.0, domain_limits=None):
        self.method = method
        self.threshold = threshold
        self.domain_limits = domain_limits

    def fit(self, X, y=None):
        """
        Compute statistics needed for clipping based on the chosen method.
        """
        X = pd.DataFrame(X).copy()

        if self.method == "zscore":
            # Store mean and std for each column
            self.means_ = X.mean()
            self.stds_ = X.std()

        elif self.method == "iqr":
            # Store quartiles and IQR for each column
            self.q1_ = X.quantile(0.25)
            self.q3_ = X.quantile(0.75)
            self.iqr_ = self.q3_ - self.q1_

        elif self.method == "quantile":
            # Store chosen quantile values for each column
            if not (0 < self.threshold < 1):
                raise ValueError("For quantile method, threshold must be between 0 and 1.")
            self.q_ = X.quantile(self.threshold)

        elif self.method == "mad":
            # Store median and Median Absolute Deviation
            self.median_ = X.median()
            self.mad_ = (X - self.median_).abs().median()

        elif self.method == "domain":
            # For domain clipping, user must provide explicit limits
            if self.domain_limits is None:
                raise ValueError("`domain_limits` must be provided for domain method.")

        return self

    def transform(self, X):
        """
        Apply clipping to the data based on the fitted statistics.
        """
        X = pd.DataFrame(X).copy()

        if self.method == "zscore":
            # Cap values outside ± threshold * std
            for col in X.columns:
                z_scores = (X[col] - self.means_[col]) / self.stds_[col]
                X[col] = np.where(
                    z_scores > self.threshold,
                    self.means_[col] + self.threshold * self.stds_[col],
                    np.where(
                        z_scores < -self.threshold,
                        self.means_[col] - self.threshold * self.stds_[col],
                        X[col],
                    ),
                )

        elif self.method == "iqr":
            # Cap values outside Tukey’s fences
            for col in X.columns:
                lower = self.q1_[col] - self.threshold * self.iqr_[col]
                upper = self.q3_[col] + self.threshold * self.iqr_[col]
                X[col] = X[col].clip(lower, upper)

        elif self.method == "quantile":
            # Clip one side only (upper or lower depending on threshold)
            for col in X.columns:
                q_val = self.q_[col]
                if self.threshold > 0.5:
                    # Cap upper tail
                    X[col] = np.minimum(X[col], q_val)
                else:
                    # Cap lower tail
                    X[col] = np.maximum(X[col], q_val)

        elif self.method == "mad":
            # Cap values outside median ± threshold * MAD
            for col in X.columns:
                mad = self.mad_[col]
                median = self.median_[col]
                if mad == 0:
                    continue  # avoid division by zero if column is constant
                lower = median - self.threshold * mad
                upper = median + self.threshold * mad
                X[col] = X[col].clip(lower, upper)

        elif self.method == "domain":
            # Apply user-defined limits directly
            for col, (lower, upper) in self.domain_limits.items():
                if col in X.columns:
                    X[col] = X[col].clip(lower, upper)

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Return column names after transformation (for sklearn pipelines).
        """
        return np.array(input_features)

# ===========================================================================================================================================
# ⚙️ KMODES IMPUTER CLASS
# ===========================================================================================================================================
class KModesImputer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that imputes missing categorical values
    using KModes clustering.

    Parameters
    ----------
    columns_for_clustering : List[str]
        Categorical columns used to define similarity for clustering.
    n_clusters : int, default=5
        Number of clusters for KModes.
    init : str, default='Huang'
        Initialization method for KModes ('Huang' or 'Cao').
    n_init : int, default=5
        Number of runs for KModes with different centroid seeds.
    placeholder : str, default='unknown'
        Placeholder representing missing values in the column to impute.
    random_state : Optional[int], default=42
        Random state for KModes clustering.

    Attributes
    ----------
    cluster_modes_ : pd.Series
        Mode (most frequent value) of target column per cluster.
    kmodes_ : KModes
        Fitted KModes instance.
    """

    ########################################################################################################################################
    ########################################################################################################################################
    # 🏗️ CLASS CONSTRUCTOR
    def __init__(
        self,
        columns_for_clustering: List[str],
        n_clusters: int = 5,
        init: str = "Huang",
        n_init: int = 5,
        placeholder: str = "unknown",
        random_state: Optional[int] = 42,
    ):
        self.columns_for_clustering = columns_for_clustering
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.placeholder = placeholder
        self.random_state = random_state
        self._missing_token = "__missing__"

    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ FIT
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            # The target column is the first (and only) column passed by the pipeline
            self._target_col_name = X.columns[0]
        else:
            raise ValueError("KModesImputer expects a pandas DataFrame as input")

        # Ensure clustering columns exist
        missing_cols = [c for c in self.columns_for_clustering if c not in X.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in input DataFrame: {missing_cols}")

        # Replace placeholder in clustering columns
        cluster_data = X[self.columns_for_clustering].replace(self.placeholder, np.nan).fillna(self._missing_token).astype(str)

        # Fit KModes
        self.kmodes_ = KModes(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state
        )
        self.kmodes_.fit(cluster_data)
        cluster_labels = self.kmodes_.predict(cluster_data)

        # Compute per-cluster mode for the target column
        target_series = X.iloc[:, 0].replace(self.placeholder, np.nan)
        self.cluster_modes_ = pd.Series([
            target_series[cluster_labels == k].mode().iat[0] 
            if not target_series[cluster_labels == k].dropna().empty 
            else self.placeholder
            for k in range(self.n_clusters)
        ], index=range(self.n_clusters))

        return self

    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ TRANSFORM
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cluster_data = X[self.columns_for_clustering].replace(self.placeholder, np.nan).fillna(self._missing_token).astype(str)
        cluster_labels = self.kmodes_.predict(cluster_data)

        target_col = X.columns[0]
        imputed_col = [
            self.cluster_modes_[cluster] if pd.isna(val) or val == self.placeholder else val
            for val, cluster in zip(X[target_col], cluster_labels)
        ]

        # Replace only the target column, leave other columns unchanged
        X[target_col] = imputed_col
        return X 

    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ GET FEATURE NAMES OUT
    def get_feature_names_out(self, input_features=None):
        # Return the same column names as input_features (default: all columns passed)
        if input_features is None:
            input_features = [self._target_col_name] + [c for c in self.columns_for_clustering if c != self._target_col_name]
        return np.array(input_features)
    
    ########################################################################################################################################
    ########################################################################################################################################
    # ⚙️ SCIKIT-LEARN COMPATIBILITY TAGS
    def _more_tags(self):
        return {"preserves_dtype": True, "requires_y": False}