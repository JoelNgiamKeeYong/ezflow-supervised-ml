import logging
from pathlib import Path
from typing import Optional
import pandas as pd


def load_excel_dataset(
    file_path: str,
    sheet_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Load an Excel dataset safely into a Pandas DataFrame.
    
    This function reads an Excel file (.xlsx, .xls) from the given path and 
    returns its content as a Pandas DataFrame. It supports selecting a specific 
    sheet if provided.
    
    Parameters
    ----------
    file_path : str
        Path to the Excel dataset file.
    sheet_name : str, optional
        Name or index of the sheet to load. If None, loads the first sheet.
    logger : logging.Logger, optional
        Logger instance to use for messages. If None, uses the root logger.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded dataset.
    
    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If there is an error while reading the Excel file.
    RuntimeError
        If any other unexpected error occurs during file reading.
    """
    log = logger or logging.getLogger(__name__)
    dataset_path = Path(file_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Excel file not found: {dataset_path}")

    try:
        df = pd.read_excel(dataset_path, sheet_name=sheet_name)
    except ValueError as e:
        raise ValueError(f"Error reading Excel file {dataset_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading Excel dataset from {dataset_path}: {e}")

    log.info("✅ Excel dataset loaded successfully from %s | Shape: %s", dataset_path, df.shape)
    return df
