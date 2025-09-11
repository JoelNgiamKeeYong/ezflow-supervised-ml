import logging
from pathlib import Path
from typing import Optional
import pandas as pd


def load_json_dataset(
    file_path: str,
    orient: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Load a JSON dataset safely into a Pandas DataFrame.
    
    This function reads a JSON file from the given path and returns its content 
    as a Pandas DataFrame. Supports different orientations for parsing JSON.
    
    Parameters
    ----------
    file_path : str
        Path to the JSON dataset file.
    orient : str, optional
        Indication of expected JSON string format. 
        Examples: 'records', 'index', 'split', etc.
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
        If there is an error while parsing the JSON file.
    RuntimeError
        If any other unexpected error occurs during file reading.
    """
    log = logger or logging.getLogger(__name__)
    dataset_path = Path(file_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {dataset_path}")

    try:
        df = pd.read_json(dataset_path, orient=orient)
    except ValueError as e:
        raise ValueError(f"Error parsing JSON file {dataset_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading JSON dataset from {dataset_path}: {e}")

    log.info("✅ JSON dataset loaded successfully from %s | Shape: %s", dataset_path, df.shape)
    return df
