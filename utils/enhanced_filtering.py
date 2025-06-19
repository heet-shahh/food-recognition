import json
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any, Tuple

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def is_box_inside(inner_box: List[int], outer_box: List[int]) -> bool:
    """Checks if the inner_box is completely contained within the outer_box."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def filter_nested_boxes(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters out bounding boxes that are nested inside others."""
    if not detections:
        return []

    boxes = [d["box"] for d in detections]
    is_inner = [False] * len(boxes)

    for i, box1 in enumerate(boxes):
        for j, box2 in enumerate(boxes):
            if i == j:
                continue
            if is_box_inside(box1, box2):
                is_inner[i] = True
                break

    return [detections[i] for i, is_in in enumerate(is_inner) if not is_in]

def should_prioritize_detection(detection1: Dict[str, Any], detection2: Dict[str, Any], generic_classes: set) -> Dict[str, Any]:
    """
    Determine which detection should be prioritized when two boxes overlap significantly.
    Returns the detection that should be kept.
    """
    class1 = detection1['class_name']
    class2 = detection2['class_name']
    prob1 = detection1.get('probability', detection1.get('confidence', 0.5))  # Handle different probability key names
    prob2 = detection2.get('probability', detection2.get('confidence', 0.5))
    
    class1_lower = class1.lower()
    class2_lower = class2.lower()
    
    # Rule 1: Prioritize generic classes over OD specific ones because classification model is more accurate
    class1_is_generic = class1_lower in generic_classes
    class2_is_generic = class2_lower in generic_classes
    
    if class1_is_generic and not class2_is_generic:
        return detection1  # Keep generic class
    elif class2_is_generic and not class1_is_generic:
        return detection2  # Keep generic class
    
    return detection1 if prob1 >= prob2 else detection2

def filter_overlapping_boxes(detections: List[Dict[str, Any]], 
                           generic_classes: set,
                           iou_threshold: float = 0.3,
                           overlap_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Advanced filtering that handles both nested boxes and overlapping detections.
    
    Args:
        detections: List of detection dictionaries
        generic_classes: Set of generic class names
        iou_threshold: IoU threshold for considering boxes as overlapping
        overlap_threshold: Overlap percentage threshold for merging boxes
    
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Step 1: Remove completely nested boxes (existing logic)
    boxes = [d['box'] for d in detections]
    is_nested = [False] * len(boxes)
    
    for i, box1 in enumerate(boxes):
        for j, box2 in enumerate(boxes):
            if i == j:
                continue
            if is_box_inside(box1, box2):
                is_nested[i] = True
                break
    
    # Keep only non-nested boxes
    non_nested_detections = [detections[i] for i, nested in enumerate(is_nested) if not nested]
    
    print(f"After removing nested boxes: {len(non_nested_detections)} detections")
    
    # Step 2: Handle overlapping boxes
    filtered_detections = []
    processed = set()
    
    for i, detection1 in enumerate(non_nested_detections):
        if i in processed:
            continue
            
        box1 = detection1['box']
        class1 = detection1['class_name']
        
        # Find all overlapping boxes with this one
        overlapping_indices = [i]
        
        for j, detection2 in enumerate(non_nested_detections):
            if i == j or j in processed:
                continue
                
            box2 = detection2['box']
            class2 = detection2['class_name']
            
            # Calculate IoU and overlap percentages
            iou = calculate_iou(box1, box2)
            
            # Check if boxes should be considered as detecting the same object
            if (iou >= iou_threshold):
                
                print(f"Found overlapping boxes: '{class1}' and '{class2}' "
                      f"(IoU: {iou:.3f})")
                
                overlapping_indices.append(j)
        
        # If we found overlapping boxes, choose the best one
        if len(overlapping_indices) > 1:
            # Get all overlapping detections
            overlapping_detections = [non_nested_detections[idx] for idx in overlapping_indices]
            
            # Choose the best detection based on class priority and probability
            best_detection = overlapping_detections[0]
            for detection in overlapping_detections[1:]:
                better_detection = should_prioritize_detection(
                    best_detection, 
                    detection, 
                    generic_classes
                )
                if better_detection == detection:
                    best_detection = detection
            
            filtered_detections.append(best_detection)
            print(f"Chose '{best_detection['class_name']}' from overlapping group")
            
            # Mark all overlapping boxes as processed
            processed.update(overlapping_indices)
        else:
            # No overlaps, keep the detection
            filtered_detections.append(detection1)
            processed.add(i)
    
    print(f"After filtering overlapping boxes: {len(filtered_detections)} detections")
    return filtered_detections