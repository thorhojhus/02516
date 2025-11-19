import cv2
import numpy as np
import torch

def run_edge_boxes(image_path, model_path='project4/models/model.yml.gz', max_bounding_boxes=100):
    # 1. Load Image
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Edge Boxes requires a specific edge map, not just standard Canny
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    
    # Convert to RGB (model expects RGB) and normalize to [0, 1]
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    rgb_im_float = np.float32(rgb_im) / 255.0
    
    # Detect edges and compute orientation map
    edges = edge_detector.detectEdges(rgb_im_float)
    orimap = edge_detector.computeOrientation(edges)
    
    # 3. Setup Edge Boxes
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(max_bounding_boxes)       # Return top 'max_bounding_boxes' boxes
    edge_boxes.setAlpha(0.65)         # Step size of sliding window
    edge_boxes.setBeta(0.75)          # NMS threshold (overlap)
    edge_boxes.setMinScore(0.02)      # Minimum score to accept a box

    # 4. Run the Algorithm
    # Returns boxes in (x, y, w, h) format and their scores
    boxes, _ = edge_boxes.getBoundingBoxes(edges, orimap)
    return boxes

# Usage
# Ensure 'model.yml.gz' is in the same directory
if __name__ == "__main__":
    # Replace with your image path
    run_edge_boxes('input.jpg')