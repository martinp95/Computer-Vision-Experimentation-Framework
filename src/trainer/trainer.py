from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Any, Dict

class Trainer:
    """
    A class to handle the training of machine learning models with early stopping.

    Methods:
    --------
    train_model(model: Model, train_data: ImageDataGenerator, val_data: ImageDataGenerator, epochs: int, patience: int) -> Dict[str, Any]:
        Train the model with early stopping.
    """

    def __init__(self) -> None:
        pass

    def train_model(self, model: Model, train_data: ImageDataGenerator, val_data: ImageDataGenerator, epochs: int, patience: int) -> Dict[str, Any]:
        """
        Train the model with early stopping.

        Parameters:
        -----------
        model : Model
            The model to train.
        train_data : ImageDataGenerator
            Data generator for training data.
        val_data : ImageDataGenerator
            Data generator for validation data.
        epochs : int
            Number of epochs to train.
        patience : int
            Number of epochs to wait for improvement before stopping training.

        Returns:
        --------
        Dict[str, Any]
            Training history.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[early_stopping]
        )
        
        return history.history
