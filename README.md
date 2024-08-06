
# Computer vision Experimentation Framework

## Overview

This project provides a framework for running computer vision experiments involving image classification. It includes components for data loading, preprocessing, model training, and result reporting.

## Project Structure

```
project/
├──src/
├──── config/
│       ├── __init__.py
│       └── config.py
├──── custom_model/
│       ├── __init__.py
│       └── custom_model.py
├──── data_loader/
│       ├── __init__.py
│       ├── data_generator.py
│       ├── data_loader.py
│       └── image_preprocessing.py
├──── experiment/
│       ├── __init__.py
│       ├── experiment.py
│       └── reporting.py
├──── trainer/
│       ├── __init__.py
│       └── trainer.py
├──── utils/
│       ├── __init__.py
│       └── utils.py
└── config.json
```

## Configuration

The configuration file `config.json` allows you to specify various parameters for your experiments, including:

- `data_path`: The path to the directory containing the image data.
- `seed`: A random seed for reproducibility.
- `batch_size`: The size of the batches used during training.
- `patience`: The patience for early stopping during training.
- `epochs`: The number of epochs to train the model.
- `target_size`: The target size for input images.
- `learning_rate`: The learning rate for model training.
- `distributions`: A list of training/validation/test splits to use in experiments.
- `segmentations`: A list of segmentation methods to apply to the images.
- `models`: A list of model architectures to use in experiments.

## Components

### Data Loading and Preprocessing

- **DataLoader**: Handles loading image data and splitting it into training, validation, and test sets.
- **ImagePreprocessing**: Contains methods for segmenting images.
- **DataGenerator**: Creates data generators with augmentation for training, validation, and test sets.

### Model Training

- **CustomModel**: Defines methods to create and compile different CNN architectures.
- **Trainer**: Handles the training process with early stopping.

### Reporting

- **Reporting**: Generates confusion matrices and classification reports for model performance.

### Experiment Management

- **Experiment**: Manages the setup and execution of experiments, including training, evaluating, and saving models.

## Running an Experiment

1. Create a configuration file (`config.json`) with the desired parameters.
2. Initialize and run the experiment:

```python
from experiment import Experiment

# Initialize Experiment with the path to your config.json
experiment = Experiment(config_path='path/to/config.json')

# Run the experiment
results = experiment.run_experiment()
```

## Saving a Model

To save a specific model based on its `model_name`, `segmentation`, and `distribution`, use the `save_model` method:

```python
# Save the model
experiment.save_model('ResNet152V2', 'segment_image_by_color', '0.7/0.15', '../model')
```

## Reporting Results

To generate reports from the experiment results:

```python
from reporting import Reporting

# Initialize Reporting
reporting = Reporting()

# Generate reports
reporting.plot_confusion_matrix_and_classification_report(results)
```