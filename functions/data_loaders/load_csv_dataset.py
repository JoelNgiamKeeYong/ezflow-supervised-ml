import logging
from pathlib import Path
from typing import Optional
import pandas as pd


def load_csv_dataset(
    file_path: str,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Load a CSV dataset safely into a Pandas DataFrame.
    
    This function reads a CSV file from the given path and returns its content
    as a Pandas DataFrame. It includes robust error handling for file existence,
    encoding issues, and other unexpected exceptions.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV dataset file.
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
        If there is an encoding issue while reading the CSV.
    RuntimeError
        If any other error occurs during file reading.
    """
    log = logger or logging.getLogger(__name__)
    dataset_path = Path(file_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Data file not found: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error while reading {dataset_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading dataset from {dataset_path}: {e}")

    log.info("✅ Dataset loaded successfully from %s | Shape: %s", dataset_path, df.shape)
    return df
