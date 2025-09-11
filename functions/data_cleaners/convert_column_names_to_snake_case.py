import logging
import re
import pandas as pd
from typing import Optional


def convert_column_names_to_snake_case(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
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
    logger : logging.Logger, optional
        Logger instance to use for messages. If None, uses the root logger.
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame with snake_case column names.
    """
    log = logger or logging.getLogger(__name__)
    log.info("Converting column names to snake_case...")

    def to_snake_case(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', '_', text)         # Replace spaces with underscores
        text = re.sub(r'[^a-z0-9_]', '', text)   # Remove special characters
        text = re.sub(r'_+', '_', text)          # Collapse multiple underscores
        return text.strip('_')                    # Remove leading/trailing underscores

    df_copy = df.copy()
    df_copy.columns = [to_snake_case(col) for col in df_copy.columns]

    log.info("Column names converted successfully.")
    return df_copy
