import cv2
import numpy as np
from typing import Any

class ImagePreprocessing:
    """
    A class to handle various image preprocessing techniques such as segmentation by color and contours.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def segment_image_by_color(image: np.ndarray) -> np.ndarray:
        """
        Segments an image to identify and highlight green pixels.

        This function takes an image in the form of a numpy array and performs the following steps:
        1. Converts the image from float32 to uint8.
        2. Converts the image from RGB color space to HSV.
        3. Defines the lower and upper limits for the green color in HSV space.
        4. Creates a mask for the pixels within the green color range.
        5. Applies the mask to the original image to extract green pixels.

        Parameters:
        -----------
        image : np.ndarray
            The input image in the form of a numpy array of type float32.

        Returns:
        --------
        np.ndarray
            The segmented image containing only the green pixels.
        """
        # Convert the image to uint8
        image_rgb = image.astype(np.uint8)
        
        # Convert from RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define the limits for the green color in HSV
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create a mask for the pixels within the green range
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result

    @staticmethod
    def segment_image_by_contour(image: np.ndarray) -> np.ndarray:
        """
        Segments the image by detecting contours.

        This function takes an input image, converts it to grayscale, applies Gaussian blur,
        detects edges using the Canny edge detector, finds contours, and creates a mask based 
        on these contours. Finally, it applies the mask to the original image to obtain the 
        segmented image.

        Parameters:
        -----------
        image : np.ndarray
            The input image.

        Returns:
        --------
        np.ndarray
            The segmented image.
        """
        # Convert the image to uint8
        image_rgb = image.astype(np.uint8)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to smooth the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to obtain a binary image
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty mask to draw the contours
        mask = np.zeros_like(gray)
        
        # Filter and draw contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours by area
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)

        return result
