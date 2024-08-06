import json
from typing import Any, Optional

class Config:
    """
    A class to handle configuration settings from a JSON file.

    Attributes:
    -----------
    config : dict
        A dictionary to store the configuration settings.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the Config object by loading the JSON configuration file.

        Parameters:
        -----------
        config_path : str
            The path to the JSON configuration file.
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> dict:
        """
        Loads the JSON configuration file.

        Parameters:
        -----------
        config_path : str
            The path to the JSON configuration file.

        Returns:
        --------
        dict
            The configuration settings as a dictionary.

        Raises:
        -------
        FileNotFoundError
            If the configuration file does not exist.
        json.JSONDecodeError
            If the configuration file contains invalid JSON.
        """
        try:
            with open(config_path, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Configuration file {config_path} contains invalid JSON.")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves the value for the specified key from the configuration.

        Parameters:
        -----------
        key : str
            The key for the configuration setting.
        default : Any, optional
            The default value to return if the key is not found (default is None).

        Returns:
        --------
        Any
            The value associated with the key, or the default value if the key is not found.
        """
        return self.config.get(key, default)
