from ultralytics import YOLO
import cv2
#import numpy as np 

class cls_facemask:
    def __init__(self, model_path="facemaskdetection/yolomodel/best.pt"):
        self.model = YOLO(model_path)

    def detect_mask(self, frame):
        predictions = self.model(frame) 
        self.box(predictions, frame)
        return frame

    def box(self, results, frame):
        for result in results:
            classes_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:  # Only process detections with confidence > 40%
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if box.cls == 1:  
                        color = (0, 255, 0)  
                    elif box.cls == 0:  
                        color = (0, 0, 255)
                    elif box.cls == 2:
                        color = (0, 255, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{classes_names[int(box.cls)]} {box.conf[0]:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)