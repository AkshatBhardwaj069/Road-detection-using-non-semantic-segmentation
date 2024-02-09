# Road-detection-using-non-semantic-segmentation
This repository contains Python scripts for road segmentation in satellite imagery. Two scripts are included, each utilizing different techniques for road segmentation.

# Road Segmentation Scripts
## Script 1: Road Segmentation Based on Color and Edge Detection

### Description
The `road_segmentation_color` script performs road segmentation based on color information and edge detection. It converts an RGB image to grayscale and LAB color space, sets color thresholds for road pixels, applies morphological operations, and detects edges using the Canny edge detector. The script then combines color and edge information to create a binary mask, which is further refined through connected component analysis. The final mask is used to calculate Intersection over Union (IOU) and F1 Score metrics.

### Requirements
- numpy
- cv2 (OpenCV)

### Usage
1. Load your RGB image and ground truth label.
2. Import the `road_segmentation_color` function from the script.
3. Call the function with your image and label arrays.
4. The function returns IOU and F1 Score metrics.
5. Print or use the metrics as needed.

### Function Signature
```python
def road_segmentation_color(image, label):
    """
    Perform road segmentation based on color information and edge detection.

    Parameters:
    - image (numpy.ndarray): RGB image.
    - label (numpy.ndarray): Ground truth binary label.

    Returns:
    - iou (float): Intersection over Union.
    - f1_score (float): F1 Score.
    """
    # ... (function code)
```

## Script 2: Multispectral Road Segmentation

### Description
The `multi_spectral_road_segmentation` script performs road segmentation on a multispectral image. It preprocesses the RGB image, extracts a custom channel, applies color and edge thresholding, performs clustering, and combines the information to create a preliminary mask. The script then clusters the image based on spectral information and adjusts the segmentation based on the intersection of edge and image map. Connected component analysis is used to remove small clusters of noise, and metrics for evaluation (IOU and F1 Score) are calculated.

### Requirements
- numpy
- cv2 (OpenCV)
- rasterio
- sklearn.mixture


### Usage
1. Load your multispectral image, RGB representation, and ground truth label.
2. Import the `multi_spectral_road_segmentation` function from the script.
3. Call the function with your image, RGB representation, and label arrays.
4. The function returns IOU and F1 Score metrics.
5. Print or use the metrics as needed.

### Function Signature
```python
def multi_spectral_road_segmentation(image_array, rgb_image, label):
    """
    Perform road segmentation on a multispectral image.

    Parameters:
    - image_array (numpy.ndarray): Multispectral image data.
    - rgb_image (numpy.ndarray): RGB representation of the image.
    - label (numpy.ndarray): Ground truth binary label.

    Returns:
    - iou (float): Intersection over Union.
    - f1_score (float): F1 Score.
    """
    # ... (function code)
```
