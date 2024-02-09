# Necessary imports
import numpy as np
import cv2
from scipy.ndimage import label
import matplotlib.pyplot as plt


def road_segmentation_color(image,label):
    
    """
    Perform road segmentation based on color information and edge detection.

    Parameters:
    - image (numpy.ndarray): RGB image.
    - label (numpy.ndarray): Ground truth binary label.

    Returns:
    - iou (float): Intersection over Union.
    - f1_score (float): F1 Score.
    """
        
    # Converting to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Converting image to lab space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Setting thresholds for road pixel values based on color
    lower_lab = np.array([110, 120, 120])
    upper_lab = np.array([180, 130, 130])  

    # Defining a kernel
    kernel = np.ones((3, 3), np.uint8)
    
    # Creating a binary mask
    mask = cv2.inRange(lab, lower_lab, upper_lab)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Applying erosion to remove unwanted white clusters
    binary_image = cv2.erode(mask, kernel, iterations=0)
    binary_image = binary_image.astype(np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Apply the Component analysis function 
    analysis = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8), 
                                                4, 
                                                cv2.CV_32S) 
    (totalLabels, label_ids, values, centroid) = analysis 
    
    # Initialize a new image to store  
    # all the output components 
    output = np.zeros(binary_image.shape, dtype="uint8") 
    
    # Loop through each component 
    for i in range(1, totalLabels): 
        
        # Area of the component 
        area = values[i, cv2.CC_STAT_AREA]  
        
        if (area > 0) and (area < 400): 
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask) 
    
    input = cv2.bitwise_not(cv2.dilate(output,(3,3),iterations=2))
    binary_image = cv2.bitwise_and(binary_image,input)
    
    # Getting the edge information
    Blurred_value = cv2.GaussianBlur(gray_image, (3,3), 0)
    edges = cv2.Canny(Blurred_value, 100, 200)
    edges = edges.astype(np.uint8)
    
    # Inverse of edge map
    edge_not = ~edges
    
    # Intersection of edge and image map
    masked_inter = cv2.bitwise_and(cv2.dilate(edge_not,kernel,iterations=1), binary_image)

    # Blank canvas
    blank = np.zeros_like(binary_image)
    
    # Probablistic hough transform for filling the blank canvas
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, threshold=100, minLineLength=20, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(blank, (x1, y1), (x2, y2), (255), 2)
            
    # Crafting the final mask
    Prediction = cv2.bitwise_and(masked_inter.astype(np.uint8),cv2.erode(blank.astype(np.uint8),kernel,iterations=2))
    
    # Applying the Component analysis function again to remove noise from the final mask
    analysis = cv2.connectedComponentsWithStats(Prediction.astype(np.uint8), 
                                                4, 
                                                cv2.CV_32S) 
    (totalLabels, label_ids, values, centroid) = analysis 
    
    output = np.zeros(Prediction.shape, dtype="uint8") 
    for i in range(1, totalLabels): 
        
        # Area of the component 
        area = values[i, cv2.CC_STAT_AREA]  
        
        if (area > 0) and (area < 1000): 
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask) 
    
    input = cv2.bitwise_not(cv2.dilate(output,(3,3),iterations=2))
    Prediction = cv2.bitwise_and(Prediction,input)
    
    # Standardizing the prediction and truth for comparison
    Prediction[Prediction == 255] = 1
    label[label == 255] = 1
    
    # Dilating the prediction
    Prediction = cv2.dilate(Prediction,kernel,iterations=1)
    
    # IOU calculation
    overlap = label.astype(np.uint8) * Prediction  # Intersection using Logical AND
    union = (label.astype(np.uint8) + Prediction) > 0  # Union using Logical OR
    iou = overlap.sum() / float(union.sum())
    
    # Calculate TP, FP, and FN and f1_score
    true_positives = np.sum((Prediction == 1) & (label == 1))
    false_positives = np.sum((Prediction == 1) & (label == 0))
    false_negatives = np.sum((Prediction == 0) & (label == 1))
    Precision = true_positives / (true_positives + false_positives)
    Recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (Precision * Recall) / (Precision + Recall)
    
    print(f"F1 Score: {f1_score}")
    print(f"intersection over union: {iou}")
    return iou, f1_score    
    
# Example usage
    
image = cv2.imread(r"D:\Python\CV_project\label\labels\image3.tiff")
mask = cv2.imread(r"D:\Python\CV_project\label\labels\image3_label.png",cv2.IMREAD_GRAYSCALE)
road_segmentation_color(image,mask)

    