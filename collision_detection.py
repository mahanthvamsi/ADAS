import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

# Load YOLO model for collision detection
yolo_weights = "weights/yolov3.weights"
yolo_cfg = "weights/yolov3.cfg"
yolo_names = "weights/coco.names"

net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getLayerNames()
output_layers_names = net.getUnconnectedOutLayersNames()
output_layers = [layer_names.index(layer) for layer in output_layers_names]

with open(yolo_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load Hugging Face model for road sign detection
processor = AutoImageProcessor.from_pretrained("huyhuyvu01/detr-traffic")
model = AutoModelForObjectDetection.from_pretrained("huyhuyvu01/detr-traffic")

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    objects = []

    for i in range(len(boxes)):
        if i in indices:
            box = boxes[i]
            x, y, w, h = box
            objects.append({
                "class": classes[class_ids[i]],
                "confidence": confidences[i],
                "box": [x, y, w, h]
            })

    return objects

def detect_road_signs(frame):
    # Convert frame to the format expected by the model
    inputs = processor(images=frame, return_tensors="pt")
    outputs = model(**inputs)
    
    target_sizes = torch.tensor([frame.shape[:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    road_signs = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        road_signs.append({
            "class": model.config.id2label[label.item()],
            "confidence": score.item(),
            "box": box.tolist()
        })
    
    # Debugging: Print detected road signs
    print(f"Detected road signs: {road_signs}")
    
    return road_signs


def check_for_collisions(objects, roi):
    for obj in objects:
        x, y, w, h = obj['box']
        bottom_y = y + h
        if bottom_y > roi:
            return True
    return False

def process_video(video_path, roi):
    cap = cv2.VideoCapture(video_path)
    collision_detected = False
    road_signs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects (YOLO)
        objects = detect_objects(frame)
        # Detect road signs (Hugging Face)
        detected_signs = detect_road_signs(frame)
        road_signs.extend(detected_signs)
        
        # Check for collisions
        if check_for_collisions(objects, roi):
            collision_detected = True
            break

    cap.release()
    return "Collision detected!" if collision_detected else "No collision detected.", road_signs


def process_camera_feed(camera_index, roi):
    cap = cv2.VideoCapture(camera_index)
    collision_detected = False
    road_signs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        objects = detect_objects(frame)
        road_signs.extend(detect_road_signs(frame))
        if check_for_collisions(objects, roi):
            collision_detected = True
            break

    cap.release()
    return "Collision detected!" if collision_detected else "No collision detected.", road_signs
