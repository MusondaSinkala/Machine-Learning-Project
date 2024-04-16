"""
-------------------------------------------------------
contains class and functions for extracting features
-------------------------------------------------------
"""

# Imports
from typing import Any
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image
from scipy.signal import argrelextrema
from sklearn.preprocessing import FunctionTransformer


# Constants


def from_pil_to_cv(img_in):
    """
    -------------------------------------------------------
    convert a pil image into opencv/numpy image
    -------------------------------------------------------
    Parameters:
       img_in: PIL image object
    Returns:
       open_cv_image: (numpy array)
    -------------------------------------------------------
    """
    pil_image = img_in.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def from_skimage_to_cv(img_in):
    """
    -------------------------------------------------------
    convert a scikit-image in RGB to image into opencv
    image in BGR
    -------------------------------------------------------
    Parameters:
       img_in: Scikit image object
    Returns:
       open_cv_image: (numpy array)
    -------------------------------------------------------
    """
    return img_in[:, :, ::-1]


def extract_details_from_path(img_path: str):
    """
    -------------------------------------------------------
    extract class and coordinate of images from path
    -------------------------------------------------------
    Parameters:
       img_path:  (pathlike)
        format `{patient id}_idx5_x{y coordinate}_y{y coordinate}_class{0|1}.png`
    Returns:

    -------------------------------------------------------
    """
    filename = Path(img_path).stem
    details = filename.split('_', 4)
    patient_id = details[0]
    x_coordinates = int(details[2].lstrip('x'))
    y_coordinates = int(details[3].lstrip('y'))
    target = int(details[4].lstrip('class'))
    return patient_id, x_coordinates, y_coordinates, target


class ImageFeatureExtractor:
    """
    -------------------------------------------------------
    class for extracting features and statistics from images
    -------------------------------------------------------
    """

    def __init__(self, features: dict | None = None):
        """
        -------------------------------------------------------
        initializes a feature extractor object with what features to extract
        -------------------------------------------------------
        Parameters:
           features: contains optional features and defualts to true (dict)
             MPI: mean pixel intensity (bool)
             SDPI: Standard deviation of pixel intensities (bool)
             OTV: Otsu's threshold value (bool)
             LM: Number of local maxima (bool)
             UPP:  Percentage in upper quarter pixel intensities (bool)
             LPP:  Percentage in upper quarter pixel intensities (bool)
        Returns:
           self
        -------------------------------------------------------
        """
        if features is None:
            features = {}
        self.features = defaultdict(lambda: True, features)

    @staticmethod
    def otsu_thr_val(img: np.ndarray) -> float:
        """
        -------------------------------------------------------
        Otsu's threshold value of a Histopathology image
        -------------------------------------------------------
        Parameters:
           img: an image (opencv image)
        Returns:
           th2: otsu's value
        -------------------------------------------------------
        """
        th2, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th2

    @staticmethod
    def num_local_maxima(img: np.ndarray) -> float:
        """
        -------------------------------------------------------
        Number of local maxima in pixel intensity histogram of an image
        -------------------------------------------------------
        Parameters:
           img: an image (opencv image)
        Returns:
           Count of local maxima (int)
        -------------------------------------------------------
        """
        histg = cv2.calcHist([img], [0], None, [256], [0, 256])
        histr = histg.reshape(-1)
        return len(argrelextrema(histr, np.greater))

    @staticmethod
    def up_quart_pixel_int(img: np.ndarray) -> float:
        """
        -------------------------------------------------------
        Percentage of pixels belonging to upper quarter of the pixel intensities
        -------------------------------------------------------
        Parameters:
           img: an image (opencv image)
        Returns:
            upper quartile pixels (float)
        -------------------------------------------------------
        """
        g_u = img.max()
        g_i = img.min()
        R = (g_u - g_i) / 4
        N = img.size
        N_h = ((g_u - R <= img) & (img < g_u)).sum()
        return (N_h / N) * 100

    @staticmethod
    def low_quart_pixel_int(img: np.ndarray) -> float:
        """
        -------------------------------------------------------
        Percentage of pixels belonging to lower quarter of the pixel intensities
        -------------------------------------------------------
        Parameters:
           img: an image (opencv image)
        Returns:
            lower quartile pixels (float)
        -------------------------------------------------------
        """
        g_u = img.max()
        g_i = img.min()
        R = (g_u - g_i) / 4
        N = img.size
        N_l = ((g_i <= img) & (img < g_i + R)).sum()
        return (N_l / N) * 100

    def __call__(self, img: Image.Image | np.ndarray) -> list[Any]:
        """
        -------------------------------------------------------
        Extracts features of the image
        -------------------------------------------------------
        Parameters:
           img: an image (PIL image | opencv image[numpy array])
        Returns:
           features : tuple with features
        -------------------------------------------------------
        """
        if isinstance(img, Image.Image):
            img = from_pil_to_cv(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = []
        # Mean pixel intensity value of an image
        features += [img.mean()] if self.features['MPI'] else []
        # Standard deviation of pixel intensities of an image
        features += [img.std()] if self.features['SDPI'] else []
        # Otsu's threshold value of an image
        features += [self.otsu_thr_val(img)] if self.features['OTV'] else []
        # Number of local maxima in pixel intensity histogram of image
        features += [self.num_local_maxima(img)] if self.features['LM'] else []
        # Percentage of pixels belonging to upper quarter of the pixel intensities
        features += [self.up_quart_pixel_int(img)] if self.features['UPP'] else []
        # Percentage of pixels belonging to lower quarter of the pixel intensities
        features += [self.low_quart_pixel_int(img)] if self.features['LPP'] else []
        return features

    def sklearn_transformer(self):
        """
        -------------------------------------------------------
        Returns an sklearn preprocessing object that can be used
        in pipeline
        -------------------------------------------------------
        Returns:
           transformer: feature transformer (sklearn.preprocessing.FunctionTransformer)
        -------------------------------------------------------
        """
        return FunctionTransformer(func=self.__call__, validate=True)

