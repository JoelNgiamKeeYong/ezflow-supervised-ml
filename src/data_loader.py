# ===========================================================================================================================================
# 📥 DATA LOADER CLASS
# ===========================================================================================================================================

# Standard library imports
import os
import sqlite3

# Third-party imports
import pandas as pd
from IPython.display import display

class DataLoader:
    """
    Utility class for loading datasets from multiple sources
    (CSV, Excel, Parquet, JSON, Feather, ORC, HDF, SQL).

    Each loader is a static method and can be called independently.
    """

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 📥 LOAD CSV
    @staticmethod
    def load_csv(filepath: str, show: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load a dataset from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        show : bool, optional (default=False)
            If True, displays the head of the DataFrame.
        **kwargs : dict
            Additional arguments for `pd.read_csv`.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"   └── ❌ File not found: {filepath}")

        print(f"   └── Loading CSV file from {filepath}...")
        df = pd.read_csv(filepath, **kwargs)
        DataLoader._report(df, show)
        return df
    
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 📥 LOAD EXCEL
    @staticmethod
    def load_excel(filepath: str, show: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load a dataset from an Excel file.

        Parameters
        ----------
        filepath : str
            Path to the Excel file (.xls, .xlsx).
        show : bool, optional (default=False)
            If True, displays the head of the DataFrame.
        **kwargs : dict
            Additional arguments for `pd.read_excel`.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"   └── ❌ File not found: {filepath}")

        print(f"   └── Loading Excel file from {filepath}...")
        df = pd.read_excel(filepath, **kwargs)
        DataLoader._report(df, show)
        return df

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 📥 LOAD PARQUET
    @staticmethod
    def load_parquet(filepath: str, show: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load a dataset from a Parquet file.

        Parameters
        ----------
        filepath : str
            Path to the Parquet file.
        show : bool, optional (default=False)
            If True, displays the head of the DataFrame.
        **kwargs : dict
            Additional arguments for `pd.read_parquet`.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"   └── ❌ File not found: {filepath}")

        print(f"   └── Loading Parquet file from {filepath}...")
        df = pd.read_parquet(filepath, **kwargs)
        DataLoader._report(df, show)
        return df
    
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 📥 LOAD JSON
    @staticmethod
    def load_json(filepath: str, show: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load a dataset from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.
        show : bool, optional (default=False)
            If True, displays the head of the DataFrame.
        **kwargs : dict
            Additional arguments for `pd.read_json`.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"   └── ❌ File not found: {filepath}")

        print(f"   └── Loading JSON file from {filepath}...")
        df = pd.read_json(filepath, **kwargs)
        DataLoader._report(df, show)
        return df

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 📥 LOAD SQL
    @staticmethod
    def load_sqlite(
        db_path: str,
        query: str,
        show: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a dataset from a SQLite database using either a direct SQL query
        or an `.sql` file stored on disk.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.
        query : str
            SQL query string OR path to a `.sql` file containing the query.
        show : bool, optional (default=False)
            If True, displays the head of the DataFrame.
        **kwargs : dict
            Additional arguments for `pd.read_sql`.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"❌ Database file not found: {db_path}")

        # Load SQL query from file if necessary
        if os.path.exists(query) and query.lower().endswith(".sql"):
            print(f"📄 Reading SQL file: {query}")
            with open(query, "r", encoding="utf-8") as f:
                query_text = f.read()
        else:
            query_text = query

        print(f"🗄️ Connecting to SQLite database: {db_path}")
        conn = sqlite3.connect(db_path)

        try:
            print("🗄️ Running SQL query...")
            df = pd.read_sql(query_text, conn, **kwargs)
        finally:
            conn.close()

        DataLoader._report(df, show)
        return df

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 📥 LOAD FEATHER
    @staticmethod
    def load_feather(filepath: str, show: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load a dataset from a Feather file.

        Parameters
        ----------
        filepath : str
            Path to the Feather file.
        show : bool, optional (default=False)
            If True, displays the head of the DataFrame.
        **kwargs : dict
            Additional arguments for `pd.read_feather`.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"   └── ❌ File not found: {filepath}")

        print(f"   └── Loading Feather file from {filepath}...")
        df = pd.read_feather(filepath, **kwargs)
        DataLoader._report(df, show)
        return df

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 📥 LOAD ORC
    @staticmethod
    def load_orc(filepath: str, show: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load a dataset from an ORC file.

        Parameters
        ----------
        filepath : str
            Path to the ORC file.
        show : bool, optional (default=False)
            If True, displays the head of the DataFrame.
        **kwargs : dict
            Additional arguments for `pd.read_orc`.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"   └── ❌ File not found: {filepath}")

        print(f"   └── Loading ORC file from {filepath}...")
        df = pd.read_orc(filepath, **kwargs)
        DataLoader._report(df, show)
        return df

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 📥 LOAD HDF
    @staticmethod
    def load_hdf(filepath: str, key: str, show: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load a dataset from an HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to the HDF5 file.
        key : str
            Key identifying the group inside the HDF5 file.
        show : bool, optional (default=False)
            If True, displays the head of the DataFrame.
        **kwargs : dict
            Additional arguments for `pd.read_hdf`.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"   └── ❌ File not found: {filepath}")

        print(f"   └── Loading HDF5 file from {filepath}...")
        df = pd.read_hdf(filepath, key=key, **kwargs)
        DataLoader._report(df, show)
        return df

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    # 🗒️ _REPORT
    @staticmethod
    def _report(df: pd.DataFrame, show: bool = False) -> None:
        """
        Print summary information about a loaded dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to report on.
        show : bool, optional (default=False)
            If True, display the first 5 rows of the dataset.
        """
        print(f"   └── Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns")

        if df.empty:
            print("   └── ⚠️ Warning: Loaded DataFrame is empty!")

        if show:
            print("\n🫧 Preview of loaded DataFrame:")
            display(df.head())