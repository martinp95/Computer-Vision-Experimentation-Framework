import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class DataLoader:
    """
    A class to handle loading and splitting image data for machine learning tasks.

    Attributes:
    -----------
    data_path : str
        The base directory where the images are stored.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, data_path: str, seed: int) -> None:
        """
        Initializes the DataLoader with the given parameters.

        Parameters:
        -----------
        data_path : str
            The base directory where the images are stored.
        seed : int
            Random seed for reproducibility.
        """
        self.data_path = data_path
        self.seed = seed
        
    def create_dataframe(self, classes: List[str]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from image data stored in directories.

        Parameters:
        -----------
        classes : List[str]
            List of class names, each corresponding to a subdirectory within the base directory.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing image paths and corresponding class labels.
        """
        data = []

        for label in classes:
            class_dir = os.path.join(self.data_path, label)
            if not os.path.exists(class_dir):
                raise ValueError(f"Class directory {class_dir} does not exist.")
            for image in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image)
                if os.path.isfile(image_path):
                    relative_path = os.path.relpath(image_path, self.data_path)
                    data.append({"path": relative_path, "class": label})

        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("No images found. Please check the directories and class names.")
        return df

    def split_data(self, df: pd.DataFrame, train_size: float, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into training, validation, and test sets.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing image paths and corresponding class labels.
        train_size : float
            Proportion of the data to use for training.
        val_size : float
            Proportion of the remaining data to use for validation.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple containing training, validation, and test DataFrames.
        """
        if not (0 < train_size < 1) or not (0 < val_size < 1 - train_size):
            raise ValueError("train_size and val_size should be between 0 and 1, and their sum should be less than 1.")
        
        train_df, temp_df = train_test_split(df, train_size=train_size, stratify=df['class'],
                                             shuffle=True, random_state=self.seed)
        val_df, test_df = train_test_split(temp_df, train_size=val_size / (1 - train_size),
                                           stratify=temp_df['class'], shuffle=True, random_state=self.seed)
        
        return train_df, val_df, test_df
