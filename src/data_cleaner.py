# ===========================================================================================================================================
# 📦 DATA CLEANER CLASS
# ===========================================================================================================================================

# Standard library imports
import re
import time
from typing import Any, Dict, List, Optional, Union

# Related third-party imports
import pandas as pd
from IPython.display import display 

class DataCleaner:
    """
    A class used to clean the data.
    """

    ########################################################################################################################################
    ########################################################################################################################################
    # 🏗️ CLASS CONSTRUCTOR
    def __init__(self):
        """
        Initializes the DataCleaner class.
        """
        pass

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 FULL CLEANING FUNCTION
    def clean_all(self, df: pd.DataFrame, mode: str = "all") -> pd.DataFrame:
        """
        Runs the cleaning pipeline selectively or fully depending on `mode`.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset
        mode : str, default="all"
            Cleaning mode:
            - "all"    : Run all cleaning functions
            - (extend with other keywords as needed)

        Returns
        -------
        pd.DataFrame
            Cleaned dataset
        """
        start_time = time.time()  
        df_cleaned = df.copy()

        # ================================================================================================================================
        # ✍️ Define your custom cleaning pipelines here (manual config section)
        # ================================================================================================================================    
        # Define cleaning pipelines
        cleaning_steps = {
            "snake_case_columns": [
                lambda df: self.convert_column_names_to_snake_case(df=df)
            ],
            # "rearrange_columns": [
            #     lambda df: self.move_column_before(df=df, col_to_move="shares", before_col="id")
            # ],
            "irrelevant": [
                lambda df: self.drop_irrelevant_features(df=df, columns_to_drop=["unnamed_0"])
            ],
            "clean_target": [
                lambda df: self.drop_missing_values(df=df, columns=["risk_category"]),
                lambda df: self.map_values(df=df, mapping_dict={"risk_category": {"Low-risk": 0, "High-risk": 1}})
            ],
            "clean_age": [
                lambda df: self.drop_missing_values(df=df, columns=["age"]),
                lambda df: self.remove_out_of_range(df=df, column="age", min_value=0, max_value=105)
            ],
            "clean_gender": [
                lambda df: self.drop_missing_values(df=df, columns=['gender'])
            ],
            "clean_comorbidities": [
                lambda df: self.drop_missing_values(df=df, columns=["comorbidities"]),
                lambda df: self.remove_out_of_range(df=df, column="comorbidities", min_value=0, max_value=20)
            ],
            "clean_length_of_stay": [
                lambda df: self.drop_missing_values(df=df, columns=["length_of_stay"]),
                lambda df: self.replace_character(df=df, columns=["length_of_stay"], to_replace=" days", replacement=""),
                lambda df: self.convert_columns_dtype(df=df, columns_types={"length_of_stay": int})
            ],
            "clean_medication_adherence": [
                lambda df: self.drop_missing_values(df=df, columns=["medication_adherence"]),
                lambda df: self.map_values(df=df, mapping_dict={"medication_adherence": {"Mediu": "Medium", "Lo": "Low", "Hig": "High"}})
            ],
            "clean_number_of_previous_admissions": [
                lambda df: self.drop_missing_values(df=df, columns=["number_of_previous_admissions"]),
                lambda df: self.remove_out_of_range(df=df, column="number_of_previous_admissions", min_value=0, max_value=15)
            ],
        }

        # Build cleaning pipeline
        if mode == "all":
            # Flatten all pipelines in order
            steps_to_run = [func for funcs in cleaning_steps.values() for func in funcs]
        elif mode in cleaning_steps:
            # Run selected function
            steps_to_run = cleaning_steps[mode]
        else:
            raise ValueError(f"   └── ❌ Unsupported cleaning mode: '{mode}' - use 'all' or one of {list(cleaning_steps.keys())}")

        # Run the selected steps
        for step_func in steps_to_run:
            df_cleaned = step_func(df_cleaned)

        elapsed_time = time.time() - start_time
        print(f"   └── Cleaning mode='{mode}' completed in {elapsed_time:.2f} seconds")

        return df_cleaned
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 GENERAL CLEANING FUNCTIONS

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 CONVERT COLUMN NAMES TO SNAKE CASE
    @staticmethod
    def convert_column_names_to_snake_case(df: pd.DataFrame, show: bool = False, verbose: int = 1) -> pd.DataFrame:
        """
        Convert all column names in a DataFrame to snake_case.
        
        This function:
        - Converts all letters to lowercase.
        - Strips leading/trailing whitespace.
        - Replaces spaces with underscores.
        - Removes all non-alphanumeric characters except underscores.
        - Collapses multiple underscores into a single underscore.
        
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame whose columns will be converted.
        show : bool, optional
            If True, prints/displays the cleaned DataFrame after processing, by default False.
        
        Returns
        -------
        pd.DataFrame
            A new DataFrame with snake_case column names.
        """
        if verbose == 0:
            print("   └── Converting column names to snake_case...")

        def to_snake_case(text: str) -> str:
            text = text.lower().strip()
            text = re.sub(r'\s+', '_', text)          # Replace spaces with underscores
            text = re.sub(r'[^a-z0-9_]', '', text)    # Remove special characters
            text = re.sub(r'_+', '_', text)           # Collapse multiple underscores
            return text.strip('_')                    # Remove leading/trailing underscores

        df_copy = df.copy()
        df_copy.columns = [to_snake_case(col) for col in df_copy.columns]

        if show:
            print("\n🫧 Cleaned DataFrame after applying snake_case to column names:")
            display(df_copy.head(1))

        return df_copy
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 CONVERT COLUMNS DATA TYPE
    @staticmethod
    def convert_columns_dtype(df: pd.DataFrame, columns_types: Dict[str, Any], show: bool = False) -> pd.DataFrame:
        """
        Convert specified columns in a DataFrame to the desired data types.

        Supported dtypes:
        - 'numeric'  : converts to numeric values (coerces errors to NaN)
        - 'category' : converts to categorical type
        - 'string'   : converts to string type
        - 'datetime' : converts to datetime (coerces errors to NaT)
        - Any valid pandas dtype (e.g., 'int', 'float', 'bool') is also supported.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to convert.
        columns_types : Dict[str, Any]
            A dictionary mapping column names to target dtypes.
        show : bool, optional
            If True, displays the converted columns and their dtypes, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with columns converted to specified data types.
        """
        print("   └── Converting columns to specified data types...")

        df_copy = df.copy()
        converted_columns = []

        for col, dtype in columns_types.items():
            if col not in df_copy.columns:
                print(f"       └── ⚠️ Column '{col}' not found. Skipping.")
                continue

            print(f"       └── Converting '{col}' to '{dtype}'")
            try:
                if dtype == 'numeric':
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                elif dtype == 'category':
                    df_copy[col] = df_copy[col].astype('category')
                elif dtype == 'string':
                    df_copy[col] = df_copy[col].astype('string')
                elif dtype == 'datetime':
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                else:
                    df_copy[col] = df_copy[col].astype(dtype)

                converted_columns.append(col)
            except Exception as e:
                print(f"       └── ⚠️ Could not convert '{col}' to '{dtype}': {e}")

        if show and converted_columns:
            print("\n🫧 Converted columns and their dtypes:")
            display(df_copy[converted_columns].dtypes)

        return df_copy
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 CONVERT COLUMNS BY DATA TYPE
    @staticmethod
    def convert_columns_by_dtype(
        df: pd.DataFrame,
        source_dtype: str,
        target_dtype: str,
        show: bool = False
    ) -> pd.DataFrame:
        """
        Convert all columns of a specified source data type to a target data type.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to convert.
        source_dtype : str
            The data type of columns to convert (e.g., 'object', 'int', 'float').
        target_dtype : str
            The target data type (e.g., 'category', 'string', 'numeric', 'datetime').
        show : bool, optional
            If True, displays which columns were converted and their new dtypes, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with converted columns.
        """
        print(f"   └── Converting all '{source_dtype}' columns to '{target_dtype}'...")

        df_copy = df.copy()
        converted_columns = []

        # Identify columns with the source dtype
        cols_to_convert = df_copy.select_dtypes(include=[source_dtype]).columns.tolist()

        if not cols_to_convert:
            print(f"   └── ⚠️ No columns of type '{source_dtype}' found.")
            return df_copy

        for col in cols_to_convert:
            try:
                print(f"       └── Converting '{col}' to '{target_dtype}'")
                if target_dtype == 'numeric':
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                elif target_dtype == 'category':
                    df_copy[col] = df_copy[col].astype('category')
                elif target_dtype == 'string':
                    df_copy[col] = df_copy[col].astype('string')
                elif target_dtype == 'datetime':
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                else:
                    df_copy[col] = df_copy[col].astype(target_dtype)
                converted_columns.append(col)
            except Exception as e:
                print(f"       └── ⚠️ Could not convert '{col}': {e}")

        if show and converted_columns:
            print("\n🫧 Converted columns and their new dtypes:")
            display(df_copy[converted_columns].dtypes)

        return df_copy
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 MOVE COLUMN BEFORE
    @staticmethod
    def move_column_before(df: pd.DataFrame, col_to_move: str, before_col: str, show: bool = False) -> pd.DataFrame:
        """
        Rearrange the order of columns in a DataFrame by moving a specific column
        before another specified column.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        col_to_move : str
            The name of the column to move.
        before_col : str
            The name of the column before which `col_to_move` should be placed.
        show : bool, optional
            If True, displays the reordered columns, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with columns rearranged.
        """
        print(f"   └── Moving column '{col_to_move}' before '{before_col}'...")

        df_copy = df.copy()

        if col_to_move not in df_copy.columns:
            raise ValueError(f"Column '{col_to_move}' not found in DataFrame.")
        if before_col not in df_copy.columns:
            raise ValueError(f"Column '{before_col}' not found in DataFrame.")
        if col_to_move == before_col:
            return df_copy  # nothing to do

        cols = list(df_copy.columns)
        cols.remove(col_to_move)
        insert_at = cols.index(before_col)
        cols.insert(insert_at, col_to_move)

        df_reordered = df_copy[cols]

        if show:
            print("\n🫧 Cleaned DataFrame after rearranging columns:")
            display(df_reordered.head(1))

        return df_reordered
    
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 REMOVING UNNECESSARY DATA

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 DROP IRRELEVANT FEATURES
    @staticmethod
    def drop_irrelevant_features(df: pd.DataFrame, columns_to_drop: list, show: bool = False) -> pd.DataFrame:
        """
        Drops specified irrelevant columns from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        columns_to_drop : list
            List of column names to drop.
        show : bool, optional
            If True, displays the columns dropped and remaining columns, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with specified columns removed.
        """
        df_copy = df.copy()
        
        if not columns_to_drop:
            print("   └── No columns specified to drop.")
            return df_copy

        existing_cols_to_drop = [col for col in columns_to_drop if col in df_copy.columns]

        if existing_cols_to_drop:
            df_copy.drop(columns=existing_cols_to_drop, inplace=True)
            cols = ", ".join(existing_cols_to_drop)
            print(f"   └── Dropping irrelevant columns: {cols}")
        else:
            print(f"   └── None of the specified columns exist in DataFrame. Nothing dropped.")

        if show:
            print("\n🫧 Cleaned DataFrame after dropping irrelevant features:\n")
            display(df_copy.info())

        return df_copy
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 DROP DUPLICATES
    @staticmethod
    def drop_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None, show: bool = False) -> pd.DataFrame:
        """
        Remove duplicate rows from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        subset : list of str, optional
            Columns to consider when identifying duplicates.
            - If None, all columns are used.
        show : bool, optional
            If True, displays the cleaned DataFrame after duplicate removal, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with duplicates removed.
        """
        df_copy = df.copy()

        # Identify duplicates before dropping
        duplicates = df_copy[df_copy.duplicated(subset=subset, keep=False)]

        if subset:
            print(f"   └── Dropping duplicates based on subset columns: {subset}")
        else:
            print("   └── Dropping duplicates based on all columns...")

        if duplicates.empty:
            print("       └── ⚠️  No duplicates found. Nothing dropped.")
        else:
            print(f"       └── ⚠️  Found {len(duplicates)} duplicate rows.")
            if show:
                display(duplicates)

        # Drop duplicates
        df_copy.drop_duplicates(subset=subset, inplace=True)

        return df_copy
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 DROP DUPLICATE IDS
    @staticmethod
    def drop_duplicate_ids(df: pd.DataFrame, id_column: str, show: bool = False) -> pd.DataFrame:
        """
        Remove rows with duplicate IDs from a DataFrame based on a specified ID column.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        id_column : str
            The column name to check for duplicate IDs.
        show : bool, optional
            If True, displays the duplicate rows (before removal) and 
            the cleaned DataFrame after duplicate removal, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with duplicate IDs removed.
        """
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found in DataFrame.")

        df_copy = df.copy()

        # Identify duplicates before dropping
        duplicates = df_copy[df_copy.duplicated(subset=[id_column], keep=False)]

        print(f"   └── Dropping duplicate IDs based on column: '{id_column}'")

        if duplicates.empty:
            print("       └── ⚠️ No duplicate IDs found. Nothing dropped.")
        else:
            print(f"       └── ⚠️ Found {len(duplicates)} rows with duplicate IDs.")
            if show:
                print("       └── Showing duplicate ID rows before removal:")
                if show:
                    display(duplicates)

        # Drop duplicates
        df_copy.drop_duplicates(subset=[id_column], inplace=True)

        return df_copy
    
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 HANDLING MISSING VALUES

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 DROP MISSING VALUES
    def drop_missing_values(df: pd.DataFrame, columns: list = None, show: bool = False) -> pd.DataFrame:
        """
        Drops rows with missing values in the specified columns.
        If no columns are specified, drops rows with missing values in ANY column.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        columns : list, optional
            List of column names to check for missing values. 
            If None or empty, rows with NaNs in any column will be dropped.
        show : bool, optional
            If True, displays the number of rows removed and remaining rows, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with rows containing missing values removed.
        """
        df_copy = df.copy()

        # If no columns specified → drop across all columns
        if not columns:
            before_count = len(df_copy)
            df_copy = df_copy.dropna()
            after_count = len(df_copy)
            dropped_count = before_count - after_count
            dropped_pct = (dropped_count / before_count * 100) if before_count > 0 else 0

            if dropped_count > 0:
                print(f"   └── Dropped {dropped_count:,} rows ({dropped_pct:.2f}%) with missing values in ANY column.")
            else:
                print("   └── No missing values found in DataFrame. Nothing dropped.")

            if show:
                print("\n🫧 Cleaned DataFrame after dropping missing values:\n")
                display(df_copy.info())

            return df_copy

        # If columns are specified
        missing_cols = [col for col in columns if col in df_copy.columns]
        if not missing_cols:
            print("   └── None of the specified columns exist in DataFrame. Nothing dropped.")
            return df_copy

        before_count = len(df_copy)
        df_copy = df_copy.dropna(subset=missing_cols)
        after_count = len(df_copy)
        dropped_count = before_count - after_count
        dropped_pct = (dropped_count / before_count * 100) if before_count > 0 else 0

        if dropped_count > 0:
            cols_str = ", ".join(missing_cols)
            print(f"   └── Dropped {dropped_count:,} rows ({dropped_pct:.2f}%) with missing values in column(s): {cols_str}")
        else:
            print(f"   └── No missing values found in specified column(s). Nothing dropped.")

        if show:
            print("\n🫧 Cleaned DataFrame after dropping missing values:\n")
            display(df_copy.info())

        return df_copy

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 MARK MISSING VALUES
    @staticmethod
    def mark_missing_values(
        df: pd.DataFrame,
        numerical_columns: List[str] = [],
        categorical_columns: List[str] = [],
        numerical_placeholder: Union[int, float] = -1,
        categorical_placeholder: str = 'missing',
        show: bool = False
    ) -> pd.DataFrame:
        """
        Impute missing values in numerical and categorical columns with specified placeholders.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        numerical_columns : list of str, optional
            Numerical columns to impute.
        categorical_columns : list of str, optional
            Categorical columns to impute.
        numerical_placeholder : int or float, default=-1
            Value to replace missing numerical data.
        categorical_placeholder : str, default='missing'
            Value to replace missing categorical data.
        show : bool, optional
            If True, displays summary of imputed columns and cleaned DataFrame, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with missing values imputed.
        """
        df_copy = df.copy()
        num_summary, cat_summary = {}, {}

        # Handle numerical columns
        if numerical_columns:
            print(f"   └── Checking numerical columns for missing values: {numerical_columns}")
            for col in numerical_columns:
                if col not in df_copy.columns:
                    print(f"       └── ⚠️ Column '{col}' not found in DataFrame. Skipping.")
                    continue
                missing_count = df_copy[col].isna().sum()
                if missing_count > 0:
                    df_copy[col] = df_copy[col].fillna(numerical_placeholder)
                    num_summary[col] = (missing_count, numerical_placeholder)

        # Handle categorical columns
        if categorical_columns:
            print(f"   └── Checking categorical columns for missing values: {categorical_columns}")
            for col in categorical_columns:
                if col not in df_copy.columns:
                    print(f"       └── ⚠️ Column '{col}' not found in DataFrame. Skipping.")
                    continue
                missing_count = df_copy[col].isna().sum()
                if missing_count > 0:
                    df_copy[col] = df_copy[col].astype("object")
                    df_copy[col] = df_copy[col].fillna(categorical_placeholder)
                    cat_summary[col] = (missing_count, categorical_placeholder)

        if not num_summary and not cat_summary:
            print("   └── ⚠️ No missing values found. Nothing imputed.")

        else:
            if num_summary:
                print("   └── Imputing numerical columns:")
                for col, (count, placeholder) in num_summary.items():
                    print(f"       └── Column '{col}': {count} missing values replaced with {placeholder}")
            if cat_summary:
                print("   └── Imputing categorical columns:")
                for col, (count, placeholder) in cat_summary.items():
                    print(f"       └── Column '{col}': {count} missing values replaced with '{placeholder}'")
            if show:
                print("\n🫧 Cleaned DataFrame after marking missing values:")
                display(df_copy)

        return df_copy

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 HANDLING NUMERICAL ISSUES

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 REMOVE OUT OF RANGE
    @staticmethod
    def remove_out_of_range(df: pd.DataFrame, column: str, min_value: Optional[float] = None, max_value: Optional[float] = None, show: bool = False) -> pd.DataFrame:
        """
        Remove rows where a column's values fall outside a specified range.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        column : str
            The column to check for out-of-range values.
        min_value : float, optional
            Minimum allowed value. Rows below this will be removed.
        max_value : float, optional
            Maximum allowed value. Rows above this will be removed.
        show : bool, optional
            If True, displays the removed rows, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with out-of-range rows removed.
        """
        df_copy = df.copy()
        
        # Boolean mask for out-of-range values
        mask = pd.Series(False, index=df_copy.index)
        if min_value is not None:
            mask |= df_copy[column] < min_value
        if max_value is not None:
            mask |= df_copy[column] > max_value
        
        out_of_range_rows = df_copy[mask]
        
        if out_of_range_rows.empty:
            print(f"   └── ⚠️ No out-of-range values found in '{column}'. Nothing removed.")
        else:
            print(f"   └── Removed {len(out_of_range_rows)} out-of-range rows in '{column}'.")
            if show:
                display(out_of_range_rows)
        
        # Remove out-of-range rows
        df_copy = df_copy[~mask].copy()
        
        return df_copy

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 GET ABSOLUTE VALUES
    @staticmethod
    def get_absolute_values(
        df: pd.DataFrame,
        columns: List[str] = [],
        show: bool = False
    ) -> pd.DataFrame:
        """
        Convert values in specified numerical columns to their absolute values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : list of str, optional
            Columns to apply absolute value transformation to.
        show : bool, optional
            If True, displays summary of transformed columns and the cleaned DataFrame, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with absolute values applied to specified columns.
        """
        df_copy = df.copy()
        changed_cols = {}

        if columns:
            print(f"   └── Taking absolute values for columns: {columns}")
            for col in columns:
                if col not in df_copy.columns:
                    print(f"       └── ⚠️ Column '{col}' not found in DataFrame. Skipping.")
                    continue
                if not pd.api.types.is_numeric_dtype(df_copy[col]):
                    print(f"       └── ⚠️ Column '{col}' is not numeric. Skipping.")
                    continue

                # Check how many values will be changed
                negative_count = (df_copy[col] < 0).sum()
                if negative_count > 0:
                    df_copy[col] = df_copy[col].abs()
                    changed_cols[col] = negative_count

        else:
            print("   └── No columns specified. Nothing to transform.")

        if not changed_cols:
            print("   └── ⚠️ No values were changed.")

        else:
            for col, neg_count in changed_cols.items():
                print(f"       └── Column '{col}': {neg_count} negative values converted to absolute values")
            if show:
                print("\n🫧 Cleaned DataFrame after absolute value transformation:")
                display(df_copy)

        return df_copy

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 HANDLING CATEGORICAL ISSUES

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 MAP VALUES
    @staticmethod
    def map_values(
        df: pd.DataFrame,
        mapping_dict: Dict[str, Dict],
        new_col: Optional[str] = None,
        default: Optional[str] = None,
        show: bool = False
    ) -> pd.DataFrame:
        """
        Map values in specified columns according to provided mapping dictionaries.
        Optionally, create a new column instead of overwriting, and assign a default
        value to any unmapped categories.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        mapping_dict : dict
            Dictionary of {column: {old_value: new_value}} mappings.
        new_col : str, optional
            If provided, creates a new column instead of overwriting the original.
        default : str, optional
            If provided, assigns this value to any unmapped categories.
        show : bool, optional
            If True, displays summary of value mappings and the cleaned DataFrame, 
            by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with mapped values applied.
        """
        df_copy = df.copy()
        changed_summary = {}

        if mapping_dict:
            print(f"   └── Mapping values for columns: {list(mapping_dict.keys())}")
            for col, mapping in mapping_dict.items():
                if col not in df_copy.columns:
                    print(f"       └── ⚠️ Column '{col}' not found in DataFrame. Skipping.")
                    continue

                before = df_copy[col].copy()
                target_col = new_col if new_col else col

                # Apply mapping
                df_copy[target_col] = df_copy[col].map(mapping)

                # If default provided → fill all unmapped with default
                if default is not None:
                    df_copy[target_col] = df_copy[target_col].fillna(default)
                else:
                    # Else fallback to original values
                    df_copy[target_col] = df_copy[target_col].fillna(df_copy[col])

                # Count changes
                changes = (before != df_copy[target_col]).sum()
                if changes > 0:
                    unmapped = set(before.unique()) - set(mapping.keys())
                    changed_summary[target_col] = {
                        "changes": changes,
                        "mapping": mapping,
                        "unmapped": sorted(unmapped) if unmapped else []
                    }
        else:
            print("   └── No mappings provided. Nothing to transform.")

        if not changed_summary:
            print("   └── ⚠️ No values were changed.")

        if changed_summary:
            for col, details in changed_summary.items():
                print(f"   └── Column '{col}': {details['changes']} values updated")
                inline_map = ", ".join([f"{old!r} → {new!r}" for old, new in details['mapping'].items()])
                print(f"       └── Mapping: {inline_map}")
                if details["unmapped"]:
                    print(f"       └── ⚠️ Unmapped categories in '{col}': {details['unmapped']}")

            if show:
                print("\n🫧 Cleaned DataFrame after value mapping:")
                display(df_copy)

        return df_copy

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 CONVERT TEXT TO LOWERCASE
    @staticmethod
    def convert_text_to_lowercase(
        df: pd.DataFrame,
        columns: List[str] = [],
        show: bool = False
    ) -> pd.DataFrame:
        """
        Convert text columns to lowercase in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : list of str, optional
            Columns to convert to lowercase.
        show : bool, optional
            If True, displays a summary of changes and the cleaned DataFrame, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with text columns converted to lowercase.
        """
        df_copy = df.copy()
        changed_columns = {}

        if not columns:
            print("   └── No columns specified for lowercase conversion.")
            return df_copy

        print(f"   └── Converting text to lowercase for columns: {columns}")

        for col in columns:
            if col not in df_copy.columns:
                print(f"       └── ⚠️ Column '{col}' not found in DataFrame. Skipping.")
                continue
            if df_copy[col].dtype != 'object':
                print(f"       └── ⚠️ Column '{col}' is not text type. Skipping.")
                continue

            before = df_copy[col].copy()
            df_copy[col] = df_copy[col].str.lower()
            changes = (before != df_copy[col]).sum()
            if changes > 0:
                changed_columns[col] = changes

        if not changed_columns:
            print("       └── ⚠️ No changes made. All text was already lowercase or columns skipped.")
        else:
            for col, count in changed_columns.items():
                print(f"       └── Column '{col}': {count} values converted to lowercase")

        if show:
            print("\n🫧 Cleaned DataFrame after lowercase conversion:")
            display(df_copy)

        return df_copy
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 REPLACE CHARACTER
    @staticmethod
    def replace_character(
        df: pd.DataFrame,
        to_replace: str,
        replacement: str,
        columns: List[str] = [],
        show: bool = False
    ) -> pd.DataFrame:
        """
        Replace a specified character or substring with another in text columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        to_replace : str
            Character or substring to replace.
        replacement : str
            Value to replace with.
        columns : list of str, optional
            Columns in which to perform the replacement.
        show : bool, optional
            If True, displays a summary of replacements and the cleaned DataFrame, by default False.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with replacements applied in specified columns.
        """
        df_copy = df.copy()
        changes_summary = {}

        if not columns:
            print("   └── No columns specified for character replacement.")
            return df_copy

        print(f"   └── Replacing '{to_replace}' with '{replacement}' in columns: {columns}")

        for col in columns:
            if col not in df_copy.columns:
                print(f"       └── ⚠️ Column '{col}' not found in DataFrame. Skipping.")
                continue
            if df_copy[col].dtype != 'object':
                print(f"       └── ⚠️ Column '{col}' is not text type. Skipping.")
                continue

            before = df_copy[col].copy()
            df_copy[col] = df_copy[col].str.replace(to_replace, replacement, regex=False)
            changes = (before != df_copy[col]).sum()
            if changes > 0:
                changes_summary[col] = changes

        if not changes_summary:
            print("       └── ⚠️ No changes made. Either characters not found or columns skipped.")
        else:
            for col, count in changes_summary.items():
                print(f"       └── Column '{col}': {count} values updated")

        if show:
            print("\n🫧 Cleaned DataFrame after replacements:")
            display(df_copy)

        return df_copy