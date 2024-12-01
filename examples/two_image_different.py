import cv2
import numpy as np
from ultralytics import YOLO

def compare_images_yolo(image1_path, image2_path):
    # Load YOLO model
    model = YOLO('yolov9c.pt')
    
    # Load and process images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Get YOLO predictions
    results1 = model(img1)[0]
    results2 = model(img2)[0]
    
    # Extract classes and boxes
    classes1 = results1.boxes.cls.cpu().numpy()
    classes2 = results2.boxes.cls.cpu().numpy()
    
    boxes1 = results1.boxes.xyxy.cpu().numpy()
    boxes2 = results2.boxes.xyxy.cpu().numpy()
    
    # Calculate class overlap
    unique_classes = np.unique(np.concatenate([classes1, classes2]))
    total_similarity = 0
    
    for cls in unique_classes:
        # Count objects of this class in both images
        count1 = np.sum(classes1 == cls)
        count2 = np.sum(classes2 == cls)
        
        # Calculate similarity for this class
        similarity = min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 0
        total_similarity += similarity
    
    # Calculate overall similarity percentage
    if len(unique_classes) > 0:
        match_percentage = (total_similarity / len(unique_classes)) * 100
    else:
        match_percentage = 0
    
    return match_percentage

# Example usage
if __name__ == "__main__":
    image1_path = "image1"
    image2_path = "image2"
    
    match = compare_images_yolo(image1_path, image2_path)
    print(f"Images match {match:.2f}%")