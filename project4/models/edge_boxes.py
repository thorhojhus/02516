import cv2
import numpy as np
import torch

class EdgeBoxesExtractor():

    def __init__(self, model_path='project4/models/model.yml.gz'):
        self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)

    def get_regions(self, im):
        # 1. Convert image to float32
        img_np = np.array(im) 

        # 3. Convert RGB to BGR (OpenCV Standard)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 4. Normalize to float32 [0, 1] (Highly recommended for this specific algorithm)
        img_float = img_bgr.astype(np.float32) / 255.0

        # Detect edges and compute orientation map
        edges = self.edge_detector.detectEdges(img_float)
        orimap = self.edge_detector.computeOrientation(edges)
        
        # 3. Setup Edge Boxes
        edge_boxes = cv2.ximgproc.createEdgeBoxes()
        # edge_boxes.setMaxBoxes(self.max_bounding_boxes)       # Return top 'max_bounding_boxes' boxes
        edge_boxes.setAlpha(0.65)         # Step size of sliding window
        edge_boxes.setBeta(0.75)          # NMS threshold (overlap)
        edge_boxes.setMinScore(0.02)      # Minimum score to accept a box

        # 4. Run the Algorithm
        # Returns boxes in (x, y, w, h) format and their scores
        boxes, _ = edge_boxes.getBoundingBoxes(edges, orimap)
        return boxes
