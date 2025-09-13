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
from rich import print as rprint

class DataCleaner:
    """
    A class used to clean the data.

    Attributes:
    -----------
    config : Dict[str, Any], optional
            Configuration dictionary containing parameters for cleaning rules, by default None
    """

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 🏗️ CLASS CONSTRUCTOR
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the DataCleaner class.

        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration dictionary containing parameters for cleaning rules, by default None
        """
        self.config = config or {}

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 FULL CLEANING FUNCTION
    def clean_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full cleaning pipeline using all individual cleaning methods.

        Parameters
        ----------
        df : pd.DataFrame
            Raw HDB res dataset

        Returns
        -------
        pd.DataFrame
            Fully cleaned dataset
        """
        start_time = time.time()  

        df_cleaned = df.copy()
        df_cleaned = self.convert_column_names_to_snake_case(df=df_cleaned)
        df_cleaned = self.drop_irrelevant_features(df=df_cleaned, columns_to_drop=["url", 'n_non_stop_words', 'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos', 'n_comments', 'average_token_length', 'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_shares', 'num_keywords', 'kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg'])
        # df_cleaned = self.drop_duplicate_rows(df=df_cleaned)
        # df_cleaned = self.clean_flat_type(df_cleaned)
        # df_cleaned = self.clean_lease_commence_date(df_cleaned)
        # df_cleaned = self.clean_storey_range(df_cleaned)
        # df_cleaned = self.clean_missing_names(df_cleaned)
        # df_cleaned = self.drop_irrelevant_columns(df_cleaned)
        # df_cleaned = self.extract_year_month(df_cleaned)
        # df_cleaned = self.process_remaining_lease(df_cleaned)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"   └── Cleaning completed in {elapsed_time:.2f} seconds")
    
        return df_cleaned
    
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 GENERAL CLEANING FUNCTIONS

    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 CONVERT COLUMN NAMES TO SNAKE CASE
    @staticmethod
    def convert_column_names_to_snake_case(df: pd.DataFrame, show: bool = False) -> pd.DataFrame:
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
            display(df_copy.head())

        return df_copy
    
    ########################################################################################################################################
    ########################################################################################################################################
    # 🧼 CONVERT COLUMN NAMES TO SNAKE CASE
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
            display(df_reordered)

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
            print("\n🫧 Cleaned DataFrame after dropping irrelevant features:")
            display(df_copy)

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
    # 🧼 MARK MISSING VALUES
    @staticmethod
    def mark_missing_values(
        df: pd.DataFrame,
        numerical_columns: List[str] = [],
        categorical_columns: List[str] = [],
        numerical_placeholder: Union[int, float] = 999,
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
        numerical_placeholder : int or float, default=999
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
        show: bool = False
    ) -> pd.DataFrame:
        """
        Map values in specified columns according to provided mapping dictionaries.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        mapping_dict : dict
            Dictionary of {column: {old_value: new_value}} mappings.
        show : bool, optional
            If True, displays summary of value mappings and the cleaned DataFrame, by default False.

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
                df_copy[col] = df_copy[col].map(mapping).fillna(df_copy[col])
                changes = (before != df_copy[col]).sum()

                if changes > 0:
                    # Find unmapped categories (still in original values but not in mapping)
                    unmapped = set(before.unique()) - set(mapping.keys())
                    changed_summary[col] = {
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
                # Inline mapping
                inline_map = ", ".join([f"{old!r} → {new!r}" for old, new in details["mapping"].items()])
                print(f"       └── Mapping: {inline_map}")

                # Warn if there are unmapped categories
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