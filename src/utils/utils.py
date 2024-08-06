import os
from collections import defaultdict
from typing import Any
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Utils:
    """
    A class containing utility functions for handling and visualizing image datasets.

    Methods:
    --------
    get_image_sizes_per_directory(df: pd.DataFrame, data_path: str) -> None:
        Get the sizes of images per directory.
    display_dataset_images(df: pd.DataFrame, data_path: str, num_images_per_class: int = 5) -> None:
        Display the specified number of images per class in rows.
    plot_image_distribution(df: pd.DataFrame) -> None:
        Plot the distribution of images per class.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_image_sizes_per_directory(df: pd.DataFrame, data_path: str) -> None:
        """
        Get the sizes of images per directory.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing image paths and corresponding class labels.
        data_path : str
            The base directory containing the image files.

        Returns:
        --------
        None
        """
        sizes_per_directory = defaultdict(lambda: defaultdict(int))

        # Iterate through each row of the DataFrame
        for _, row in df.iterrows():
            class_name = row['class']
            image_path = os.path.join(data_path, row['path'])
            with Image.open(image_path) as img:
                width, height = img.size
                sizes_per_directory[class_name][(width, height)] += 1

        # Print the sizes of images per directory
        for class_name, sizes in sizes_per_directory.items():
            print(f"{class_name}:")
            for size, count in sizes.items():
                print(f"  Size: {size} -> {count}")

    @staticmethod
    def display_dataset_images(df: pd.DataFrame, data_path: str, num_images_per_class: int = 5) -> None:
        """
        Display the specified number of images per class in rows.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing image paths and corresponding class labels.
        data_path : str
            The base directory containing the image files.
        num_images_per_class : int, optional
            Number of images to display per class. Default is 5.

        Returns:
        --------
        None
        """
        classes = df['class'].unique()

        # Create a new figure
        plt.figure(figsize=(15, 3 * len(classes)))

        for i, class_name in enumerate(classes, 1):
            class_df = df[df['class'] == class_name]
            class_images = class_df.head(num_images_per_class)

            for j, (_, row) in enumerate(class_images.iterrows(), 1):
                plt.subplot(len(classes), num_images_per_class, (i - 1) * num_images_per_class + j)
                image_path = os.path.join(data_path, row['path'])
                with Image.open(image_path) as img:
                    plt.imshow(img)
                plt.axis('off')
                plt.title(class_name)

        # Adjust layout to prevent overlapping titles
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_image_distribution(df: pd.DataFrame) -> None:
        """
        Plot the distribution of images per class.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing image paths and corresponding class labels.

        Returns:
        --------
        None
        """
        class_counts = df['class'].value_counts()

        # Create a colormap with a unique color for each class
        cmap = cm.get_cmap('tab10', len(class_counts))

        plt.figure(figsize=(10, 5))
        bars = plt.bar(class_counts.index, class_counts.values, color=cmap(range(len(class_counts))))
        plt.xlabel('Class')
        plt.ylabel('Number of images')
        plt.title('Distribution of images per class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval), ha='center', va='bottom')

        plt.show()
