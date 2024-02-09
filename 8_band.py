import numpy as np
import cv2
import rasterio
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def multi_spectral_road_segmentation(image_array, rgb_image, label):
    """
    Perform road segmentation on a multi-spectral image.

    Parameters:
    - image_array (numpy.ndarray): Multi-spectral image data.
    - rgb_image (numpy.ndarray): RGB representation of the image.
    - label (numpy.ndarray): Ground truth binary label.

    Returns:
    - iou (float): Intersection over Union.
    - f1_score (float): F1 Score.
    """
    
    # Preprocessing the image
    rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    equalized_rgb = cv2.merge([cv2.equalizeHist(channel) for channel in cv2.split(rgb)])
    Lab = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
    equalized_Lab = cv2.merge([cv2.equalizeHist(channel) for channel in cv2.split(Lab)])
    grayscale_image = cv2.cvtColor(equalized_rgb, cv2.COLOR_BGR2GRAY)
    
    nir2_band = image_array[7, :, :]
    
    # Deriving the chosen channel
    custom_channel = equalized_Lab[:, :, 0] - cv2.equalizeHist(nir2_band.astype(np.uint8))

    # Kernel 
    kernel = (3, 3)
    
    # Thresholding to obtain a preliminary mask
    custom_channel = np.where(custom_channel > 128, 255, 0)
    block_size = 21
    c = 8
    mask = cv2.adaptiveThreshold(custom_channel.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel, iterations=10)
    
    # Getting the edge information
    Blurred_value = cv2.GaussianBlur(grayscale_image, kernel, 0)
    edges = cv2.Canny(Blurred_value, 100, 200)
    edges = edges.astype(np.uint8)
    
    # Inverse of edge map
    edge_not = ~edges
    
    # Intersection of edge and image map
    new_mask = cv2.bitwise_and(cv2.dilate(edge_not,kernel,iterations=3), mask)
    
    # Clustering
    # Extracting features in the form of pixel data
    pixel_data = nir2_band.reshape((-1, 1))
    num_clusters = 2
    gmm = GaussianMixture(num_clusters, random_state=42)
    gmm.fit(pixel_data)
    labels = gmm.predict(pixel_data)
    segmented_image = labels.reshape(nir2_band.shape)
    segmented_image = segmented_image.astype(np.uint8)
    segmented_image = segmented_image * 255 # changing the labels in order to standardise
    
    # Getting the background labels 
    intersection = np.logical_and(new_mask, segmented_image)
    union = np.logical_or(new_mask, segmented_image)
    overlap_ratio_segmented = np.sum(intersection) / np.sum(union)
    overlap_ratio_inverse_segmented = np.sum(np.logical_and(new_mask, ~segmented_image)) / np.sum(np.logical_or(new_mask, ~segmented_image))
    
    if overlap_ratio_segmented > overlap_ratio_inverse_segmented:
        pass  
    else:
        segmented_image = cv2.bitwise_not(segmented_image)

    # Mask with combined color, edge and cluster information
    fin = cv2.bitwise_and(cv2.erode(new_mask.astype(np.uint8), kernel, iterations=3), segmented_image.astype(np.uint8))
    
    blank = np.zeros(fin.shape)    # Initializing a blank canvas
    
    # Applying Hough transform to detect lines with Probabilistic Hough Transform
    lines = cv2.HoughLinesP(fin, 1, np.pi / 180, threshold=100, minLineLength=180, maxLineGap=30)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(blank, (x1, y1), (x2, y2), (255), 2)
    Prediction = cv2.bitwise_and(segmented_image.astype(np.uint8), blank.astype(np.uint8), fin.astype(np.uint8))
    
    
    # Applying the Component analysis function to remove small clusters of noise
    analysis = cv2.connectedComponentsWithStats(Prediction.astype(np.uint8), 
                                                4, 
                                                cv2.CV_32S) 
    (totalLabels, label_ids, values, centroid) = analysis 

    output = np.zeros(Prediction.shape, dtype="uint8") 
    for i in range(1, totalLabels): 
        
        area = values[i, cv2.CC_STAT_AREA]  
        
        if (area > 2) and (area < 10000): 
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask) 

    # Updating the prediction map
    input = cv2.bitwise_not(cv2.dilate(output,kernel,iterations=2))
    Prediction = cv2.bitwise_and(Prediction,input)

    plt.imshow(Prediction,cmap= 'gray')
    plt.title('Resultant Mask')
    plt.show()
    # Standardizing the prediction and truth for comparison
    Prediction[Prediction == 255] = 1
    label[label == 255] = 1
    
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
    print(f"Intersection over Union: {iou}")
    return iou, f1_score    


# Example usage

#image loading
image_path = r"D:\Python\CV_project\label\labels\SN5_roads_train_AOI_8_Mumbai_PS-MS_chip35.tif"
# Open the image using rasterio
with rasterio.open(image_path) as src:
    image_array = src.read() # Multi spectral image
rgb_image_path = r"D:\Python\CV_project\label\labels\SN5_roads_train_AOI_8_Mumbai_PS-RGB_chip35.tif"
rgb_image = cv2.imread(rgb_image_path) # Rgb image
label = cv2.imread(r"D:\Python\CV_project\label\labels\mumbai35label.png", cv2.IMREAD_GRAYSCALE) # Label

        
multi_spectral_road_segmentation(image_array,rgb_image,label)




