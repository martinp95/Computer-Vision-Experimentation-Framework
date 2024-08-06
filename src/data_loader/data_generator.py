from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from typing import Callable, Tuple, Optional

class DataGenerator:
    """
    A class to handle the creation of data generators with data augmentation for training, validation, and testing datasets.

    Attributes:
    -----------
    target_size : Tuple[int, int]
        The desired dimensions to which all images found will be resized.
    batch_size : int
        The number of samples per gradient update.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, target_size: Tuple[int, int], batch_size: int, seed: int) -> None:
        """
        Initializes the DataGenerator with the given parameters.

        Parameters:
        -----------
        target_size : Tuple[int, int]
            The desired dimensions to which all images found will be resized.
        batch_size : int
            The number of samples per gradient update.
        seed : int
            Random seed for reproducibility.
        """
        self.target_size = target_size
        self.batch_size = batch_size
        self.seed = seed

    def apply_data_augmentation(self, preprocessing_function: Optional[Callable]) -> ImageDataGenerator:
        """
        Applies data augmentation techniques and a preprocessing function.

        Parameters:
        -----------
        preprocessing_function : Optional[Callable]
            A preprocessing function to apply to each image.

        Returns:
        --------
        ImageDataGenerator
            An ImageDataGenerator object configured with the specified data augmentation techniques.
        """
        return ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def create_generators(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, base_dir: str,
                          preprocessing_function: Optional[Callable]) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
        """
        Creates data generators for training, validation, and test datasets.

        Parameters:
        -----------
        train_df : pd.DataFrame
            DataFrame containing the training data.
        val_df : pd.DataFrame
            DataFrame containing the validation data.
        test_df : pd.DataFrame
            DataFrame containing the test data.
        base_dir : str
            The base directory where the images are stored.
        preprocessing_function : Optional[Callable]
            A preprocessing function to apply to each image.

        Returns:
        --------
        Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]
            A tuple containing the training, validation, and test data generators.
        """
        train_datagen = self.apply_data_augmentation(preprocessing_function)
        valid_test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        
        train_data = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=base_dir,
            x_col='path',
            y_col='class',
            target_size=self.target_size,
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=True,
            class_mode='categorical'
        )
        
        val_data = valid_test_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=base_dir,
            x_col='path',
            y_col='class',
            target_size=self.target_size,
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=False,
            class_mode='categorical'
        )
        
        test_data = valid_test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=base_dir,
            x_col='path',
            y_col='class',
            target_size=self.target_size,
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=False,
            class_mode='categorical'
        )
        
        return train_data, val_data, test_data
