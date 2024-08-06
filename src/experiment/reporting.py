import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class Reporting:
    """
    A class to handle the reporting and visualization of model performance metrics.

    Methods:
    --------
    plot_confusion_matrix_and_classification_report(all_results: List[Dict[str, Any]]) -> None:
        Plot confusion matrix and classification report for each result.
    """

    def __init__(self) -> None:
        pass

    def plot_confusion_matrix_and_classification_report(self, all_results: List[Dict[str, Any]]) -> None:
        """
        Plot confusion matrix and classification report for each result.

        Parameters:
        -----------
        all_results : List[Dict[str, Any]]
            A list of dictionaries containing model performance metrics and histories.

        Returns:
        --------
        None
        """
        for result in all_results:
            model_name = result['model_name']
            segmentation = result['segmentation']
            distribution = result['distribution']

            print(f"Model: {model_name}, Segmentation: {segmentation}, Distribution: {distribution}")
            
            # Plot training and validation metrics
            self._plot_metrics(result['history'], model_name, segmentation, distribution)
            
            # Plot confusion matrix
            cm = confusion_matrix(result['test_labels'], result['test_predictions'])
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=result['classes'], yticklabels=result['classes'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f"Confusion Matrix\nModel: {model_name}, Segmentation: {segmentation}, Distribution: {distribution}")
            plt.show()
            
            # Print classification report
            cr = classification_report(result['test_labels'], result['test_predictions'], target_names=result['classes'])
            print(f"Classification Report\nModel: {model_name}, Segmentation: {segmentation}, Distribution: {distribution}")
            print(cr)
        
    def _plot_metrics(self, history: Dict[str, List[float]], model_name: str, segmentation: str, distribution: str) -> None:
        """
        Plot training and validation metrics.

        Parameters:
        -----------
        history : Dict[str, List[float]]
            A dictionary containing the training history.
        model_name : str
            The name of the model.
        segmentation : str
            The segmentation method used.
        distribution : str
            The distribution of the data.

        Returns:
        --------
        None
        """
        best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))

        # Plot training and validation accuracy
        axs[0].plot(history['accuracy'], label='Training Accuracy', color='blue')
        axs[0].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        axs[0].scatter(best_epoch - 1, history['val_accuracy'][best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_title(f'Training and Validation Accuracy\nModel: {model_name}, Segmentation: {segmentation}, Distribution: {distribution}')
        axs[0].legend()

        # Plot training and validation loss
        axs[1].plot(history['loss'], label='Training Loss', color='blue')
        axs[1].plot(history['val_loss'], label='Validation Loss', color='red')
        axs[1].scatter(best_epoch - 1, history['val_loss'][best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_title(f'Training and Validation Loss\nModel: {model_name}, Segmentation: {segmentation}, Distribution: {distribution}')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
