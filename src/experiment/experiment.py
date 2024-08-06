import os
import json
import numpy as np
import random
import tensorflow as tf
from typing import List, Optional, Any, Dict

from config import Config
from data_loader import DataLoader, ImagePreprocessing, DataGenerator
from custom_model import CustomModel
from trainer import Trainer

class Experiment:
    """
    A class to manage the setup and execution of machine learning experiments.

    Methods:
    --------
    run_experiment() -> List[dict]:
        Runs the experiment by training and evaluating models on the dataset.
    save_model(all_results: List[Dict[str, Any]], model_name: str, segmentation: str, distribution: str, save_dir: str) -> None:
        Save the model that matches the given criteria.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the Experiment with configuration settings.

        Parameters:
        -----------
        config_path : str
            The path to the configuration JSON file.
        """
        self.config = Config(config_path)
        self.batch_size = self.config.get('batch_size')
        self.epochs = self.config.get('epochs')
        self.seed = self.config.get('seed')
        self.target_size = tuple(self.config.get('target_size'))
        self.data_path = self.config.get('data_path')
        self.distributions = self.config.get('distributions')
        self.segmentations = self._import_dynamic_segmentation_methods()
        self.models = self.config.get('models')
        self.learning_rate = self.config.get('learning_rate')
        self.patience = self.config.get('patience')
        
        self.data_loader = DataLoader(self.data_path, self.seed)
        self.data_generator = DataGenerator(self.target_size, self.batch_size, self.seed)
        self.model = CustomModel()
        self.trainer = Trainer()
        
        # Set seed for reproducibility
        self._set_seed(self.seed)
        
    def _set_seed(self, seed: int) -> None:
        """
        Sets the random seed for reproducibility.

        Parameters:
        -----------
        seed : int
            The seed value to set.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def _import_dynamic_segmentation_methods(self) -> List[Optional[Any]]:
        """
        Dynamically imports segmentation methods specified in the configuration.

        Returns:
        --------
        List[Optional[Callable]]:
            A list of segmentation methods or None if no method is specified.
        """
        segmentation_methods = self.config.get('segmentations')
        segmentations = []
        for method_name in segmentation_methods:
            if method_name is None:
                segmentations.append(None)
            else:
                method = getattr(ImagePreprocessing, method_name)
                segmentations.append(method)
        return segmentations
        
    def run_experiment(self) -> List[dict]:
        """
        Runs the experiment by training and evaluating models on the dataset.

        Returns:
        --------
        List[dict]:
            A list of dictionaries containing model performance metrics and histories.
        """
        classes = os.listdir(self.data_path)
        
        df = self.data_loader.create_dataframe(classes)
        
        input_shape = (self.target_size[0], self.target_size[1], 3)
        
        all_results = []
        
        for train_size, val_size in self.distributions:
            train_df, val_df, test_df = self.data_loader.split_data(df, train_size, val_size)
            
            for segmentation_method in self.segmentations:
                train_data, val_data, test_data = self.data_generator.create_generators(
                    train_df, val_df, test_df, self.data_path, segmentation_method
                )
                
                for model_name in self.models:
                    # Create and train the model
                    model = self.model.create_model(model_name, self.learning_rate, input_shape, len(classes))
                    history = self.trainer.train_model(model, train_data, val_data, epochs=self.epochs, patience=self.patience)

                    # Evaluate the model on test data
                    test_loss, test_acc = model.evaluate(test_data)
                    print(f"Model: {model_name}, Segmentation: {segmentation_method}, Distribution: {train_size}/{val_size}")
                    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

                    # Predict the labels for test data
                    test_labels = test_data.classes
                    test_predictions = np.argmax(model.predict(test_data), axis=-1)

                    # Save the history and evaluation metrics
                    all_results.append({
                        'model_name': model_name,
                        'model': model,
                        'segmentation': segmentation_method,
                        'distribution': f"{train_size}/{val_size}",
                        'history': history,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'test_labels': test_labels,
                        'test_predictions': test_predictions,
                        'classes': list(test_data.class_indices.keys()),
                        'test_data': test_data
                    })
        
        return all_results
    
    def save_model(self, all_results: List[Dict[str, Any]], model_name: str, segmentation: str, distribution: str, save_dir: str) -> None:
        """
        Save the model that matches the given criteria and also save the classes in a JSON file.

        Parameters:
        -----------
        all_results : List[Dict[str, Any]]
            A list of dictionaries containing model performance metrics and histories.
        model_name : str
            The name of the model to search for.
        segmentation : str
            The segmentation method to search for.
        distribution : str
            The distribution to search for.
        save_dir : str
            The directory where the model should be saved.

        Returns:
        --------
        None
        """
        for result in all_results:
            if (result['model_name'] == model_name and 
                result['segmentation'] == segmentation and 
                result['distribution'] == distribution):
                
                # Model found
                model = result['model']
                classes = result['classes']
                
                # Create the save directory if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)                
                
                model_path_tf = os.path.join(save_dir, 'saved_model.keras')
                classes_path = os.path.join(save_dir, 'model_class_names.json')

                # Save the model in TensorFlow SavedModel format
                model.save(model_path_tf)

                # Save the classes to a JSON file
                with open(classes_path, 'w') as json_file:
                    json.dump(classes, json_file)
                
                print(f"Model saved at {model_path_tf}")
                print(f"Classes saved at {classes_path}")
                break
        else:
            print("Model not found.")
