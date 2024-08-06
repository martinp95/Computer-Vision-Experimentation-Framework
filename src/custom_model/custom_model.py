from tensorflow.keras.applications import ResNet152V2, InceptionResNetV2, DenseNet201
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Tuple

class CustomModel:
    """
    A class to create and manage custom CNN models based on different architectures.

    Methods:
    --------
    create_model(model_name: str, learning_rate: float, input_shape: Tuple[int, int, int], num_classes: int) -> Model:
        Create and compile a CNN model based on the specified architecture.
    """

    def __init__(self) -> None:
        pass
    
    def _create_ResNet152V2_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> Model:
        """
        Creates a ResNet152V2-based model for image classification.

        Parameters:
        -----------
        input_shape : Tuple[int, int, int]
            The shape of the input images (height, width, channels).
        num_classes : int
            The number of classes for classification.

        Returns:
        --------
        model : Model
            The constructed ResNet152V2 model.
        """
        return self._build_model(input_shape, num_classes, ResNet152V2, 1.0 / 127.5, -1)
    
    def _create_InceptionResNetV2_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> Model:
        """
        Creates an InceptionResNetV2-based model for image classification.

        Parameters:
        -----------
        input_shape : Tuple[int, int, int]
            The shape of the input images (height, width, channels).
        num_classes : int
            The number of classes for classification.

        Returns:
        --------
        model : Model
            The constructed InceptionResNetV2 model.
        """
        return self._build_model(input_shape, num_classes, InceptionResNetV2, 1.0 / 127.5, -1)
    
    def _create_DenseNet201_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> Model:
        """
        Creates a DenseNet201-based model for image classification.

        Parameters:
        -----------
        input_shape : Tuple[int, int, int]
            The shape of the input images (height, width, channels).
        num_classes : int
            The number of classes for classification.

        Returns:
        --------
        model : Model
            The constructed DenseNet201 model.
        """
        return self._build_model(input_shape, num_classes, DenseNet201, 1.0 / 255.0, 0)

    def _build_model(self, input_shape: Tuple[int, int, int], num_classes: int, base_model_class, scale: float, offset: float) -> Model:
        """
        Helper function to build and return a model.

        Parameters:
        -----------
        input_shape : Tuple[int, int, int]
            The shape of the input images (height, width, channels).
        num_classes : int
            The number of classes for classification.
        base_model_class : class
            The base model class to use (e.g., ResNet152V2, InceptionResNetV2, DenseNet201).
        scale : float
            The scale factor for rescaling input images.
        offset : float
            The offset value for rescaling input images.

        Returns:
        --------
        model : Model
            The constructed model.
        """
        # Define input layer
        inputs = Input(input_shape)

        # Define rescaling layer to normalize pixel values
        x = Rescaling(scale=scale, offset=offset)(inputs)
        base_model = base_model_class(weights='imagenet', include_top=False)
        
        # Freeze the pre-trained layers to prevent them from being trained
        base_model.trainable = False

        # Pass the input through the base model
        x = base_model(x)
        
        # Apply global average pooling to reduce the dimensionality of the output
        x = GlobalAveragePooling2D()(x)

        # Add a fully connected layer with 1024 units and ReLU activation
        x = Dense(1024, activation='relu')(x)

        # Add a dropout layer to reduce overfitting
        x = Dropout(0.2)(x)

        # Add the output layer with softmax activation for multiclass classification
        outputs = Dense(num_classes, activation='softmax')(x)

        # Define the model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model

    def create_model(self, model_name: str, learning_rate: float, input_shape: Tuple[int, int, int], num_classes: int) -> Model:
        """
        Create a CNN model based on the specified architecture.

        Parameters:
        -----------
        model_name : str
            Name of the model architecture ('ResNet152V2', 'InceptionResNetV2', 'DenseNet201').
        learning_rate : float
            Learning rate for the Adam optimizer.
        input_shape : Tuple[int, int, int]
            Shape of the input images (height, width, channels).
        num_classes : int
            The number of classes for classification.

        Returns:
        --------
        Model
            The created and compiled model.
        """
        if model_name == 'ResNet152V2':
            model = self._create_ResNet152V2_model(input_shape, num_classes)
        elif model_name == 'InceptionResNetV2':
            model = self._create_InceptionResNetV2_model(input_shape, num_classes)
        elif model_name == 'DenseNet201':
            model = self._create_DenseNet201_model(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    