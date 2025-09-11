import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


def load_config(
    config_file: str,
    required_keys: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Load a YAML configuration file and optionally validate required keys.
    
    This function reads a YAML file from the given path and returns its content
    as a dictionary. It can also check for the presence of required keys to 
    ensure the configuration meets expected standards.
    
    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    required_keys : List[str], optional
        List of keys that must be present in the configuration. Raises KeyError 
        if any are missing. Default is None (no validation).
    logger : logging.Logger, optional
        Logger instance to use for messages. If None, uses the root logger.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the parsed configuration.
    
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the YAML file cannot be parsed.
    KeyError
        If any of the required keys are missing from the configuration.
    """
    log = logger or logging.getLogger(__name__)
    config_path = Path(config_file)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")

    if required_keys:
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"Missing required config keys: {missing_keys}")

    log.info("✅ Config loaded successfully from %s", config_path)
    return config
