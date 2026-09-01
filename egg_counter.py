from shiny import App, ui, render
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os
import time
import pandas as pd
import tensorflow as tf
from scipy import signal
from PIL import Image, ImageOps
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def getBorders(cv2Image):
  """
  Detects the borders of the counting slide in an image.
  
  Uses adaptive thresholding and peak detection to identify the edges of
  the counting slide. Applies corrections if detected dimensions are outside
  expected ranges.
  
  Args:
      cv2Image: Input image as a numpy array (BGR format)
  
  Returns:
      list: A list containing [row1, row2, column1, column2] coordinates
            defining the borders of the counting slide
  """
  src = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
  src = cv2.blur(src,(src.shape[1], 1))
  src = cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,101,2)
  points = np.sum(src, axis = 1)
  points = signal.savgol_filter(points, 300, 2)
  plt.cla()
  plt.plot(range(len(points)), points)
  peaks, _ = signal.find_peaks(points, distance = 1000)
  row1 = peaks[0]
  row2 = peaks[-1]
  src = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
  src = cv2.blur(src,(1, src.shape[0]))
  src = cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,101,2)
  points = np.sum(src, axis = 0)
  points = signal.savgol_filter(points, 300, 2)
  plt.plot(range(len(points)), points)
  peaks, _ = signal.find_peaks(points, distance = 700)
  column1 = peaks[0]
  column2 = peaks[-1]
  if(abs((column2 - column1) - 4365) > 200):
    print("col corrected")
    column1 = 227
    column2 = 4643
  if(abs((row2 - row1) - 2100) > 150):
    print("row corrected")
    row1 = 725
    row2 = 2979
  return([row1, row2, column1, column2])

def tupleToList(t):
    """
    Recursively converts tuples to lists in a nested structure.
    
    This is useful for converting OpenCV tuple structures (which may contain
    nested tuples) to lists for easier manipulation in Python.
    
    Args:
        t: A tuple, list, or other value to convert
    
    Returns:
        list or original value: A list if input was a tuple/list, otherwise the original value
    """
    return list(map(tupleToList, t)) if isinstance(t, (list, tuple)) else t

def cropSquareFromContour(c, img):
    """
    Crops and warps a square region from an image based on a contour.
    
    Uses perspective transformation to extract a square region defined by
    the minimum area rectangle of the contour. The result is a square image
    regardless of the contour's orientation.
    
    Args:
        c: Contour points as a numpy array
        img: Source image as a numpy array
    
    Returns:
        numpy.ndarray: Warped square image extracted from the contour region
    """

    rect = cv2.minAreaRect(c)

    rect = tupleToList(rect)
    rect[1][0] = max(rect[1])
    rect[1][1] = max(rect[1])

    box = cv2.boxPoints(rect)
    box = np.intp(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped

def cropRectangleFromContour(c, img):
    """
    Crops and warps a rectangular region from an image based on a contour.
    
    Uses perspective transformation to extract a rectangular region defined by
    the minimum area rectangle of the contour. Preserves the aspect ratio
    of the detected rectangle.
    
    Args:
        c: Contour points as a numpy array
        img: Source image as a numpy array
    
    Returns:
        numpy.ndarray: Warped rectangular image extracted from the contour region
    """

    rect = cv2.minAreaRect(c)

    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped

def classifyObject(cv2Images):
    """
    Classifies objects in images as eggs or not eggs using a TensorFlow model.
    
    Preprocesses images (resize to 150x150, normalize) and runs them through
    a pre-trained TensorFlow SavedModel. Can handle both single images and
    batches of images for efficient processing.
    
    Args:
        cv2Images: A single image (numpy array) or list of images to classify
    
    Returns:
        numpy.ndarray: Array of prediction probabilities for each image,
                       shape (n_images, n_classes) where classes are ["egg", "not egg"]
    """
    # Handle both single image and batch of images
    if not isinstance(cv2Images, list):
        cv2Images = [cv2Images]
    
    # Preprocess all images
    processed_images = []
    for img in cv2Images:
        img = cv2.resize(img, (150, 150))
        img = Image.fromarray(img)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        processed_images.append(img_array)
    
    # Stack images into a batch
    batch = np.stack(processed_images)
    batch = batch / 255.0  # Normalize
    
    # Make predictions in batch
    predictions = model.signatures['serving_default'](tf.constant(batch))
    return predictions['output_0'].numpy()

modelDirectory = "saved_model"
model = tf.saved_model.load(modelDirectory)

def countImage(img):
    """
    Counts eggs in an image using computer vision and machine learning.
    
    The main function that orchestrates the egg counting process:
    1. Detects the borders of the counting slide
    2. Applies adaptive thresholding to find potential egg objects
    3. Filters contours by size and aspect ratio
    4. Classifies each candidate using a TensorFlow model
    5. Draws bounding boxes and count on the image
    
    Args:
        img: Input image as a numpy array (RGB format)
    
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Processed image with detected eggs highlighted
            - int: Total count of detected eggs
    """
    thisBorders = getBorders(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("thresholding image")
    thres = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,101,2)
    kernel = np.ones((7, 7), np.uint8)
    thres = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Collect potential egg images first
    potential_eggs = []
    potential_contours = []
    print("collecting potential eggs")
    for i, c in enumerate(cnts):
        objectIsInFrame = (
            thisBorders[0] < c.flatten()[1] < thisBorders[1] and
            thisBorders[2] < c.flatten()[0] < thisBorders[3]
        )
        area = cv2.contourArea(c)
        if(area < 4000 and area > 1500 and objectIsInFrame):
            width = min(cv2.minAreaRect(c)[1])
            length = max(cv2.minAreaRect(c)[1])
            if(length > 40 and length < 150 and length/width > 1.5 and length/width < 5):
                thisObject = cropSquareFromContour(c, img)
                potential_eggs.append(thisObject)
                potential_contours.append(c)

    # Process in batches
    batch_size = np.min([2 ** 13, len(potential_eggs)])
    cnts_filtered = []
    modelClasses = ("egg", "not egg")
    
    print("classifying eggs in batches")
    for i in range(0, len(potential_eggs), batch_size):
        batch = potential_eggs[i:i + batch_size]
        predictions = classifyObject(batch)
        for j, pred in enumerate(predictions):
            if modelClasses[np.argmax(pred)] == "egg":
                cnts_filtered.append(potential_contours[i + j])

    cv2.rectangle(img, (thisBorders[2], thisBorders[0]), (thisBorders[3], thisBorders[1]), 1, 10)
    cv2.drawContours(img, cnts_filtered, -1, (0,255, 20), 6)
    img = cv2.putText(img, 'Egg count: ' + str(len(cnts_filtered)), (100, 100), 1, 8, 1, 10)
    print("finished counting image")
    return img, len(cnts_filtered)
