import cv2
import numpy as np

# Load YOLO model
yolo_weights = "weights/yolov3.weights"
yolo_cfg = "weights/yolov3.cfg"
yolo_names = "weights/coco.names"

net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getLayerNames()
print("Layer Names:", layer_names)  # Debugging: Print layer names

# Get the names of the output layers
output_layers_names = net.getUnconnectedOutLayersNames()
print("Output Layers:", output_layers_names)  # Debugging: Print output layer names

output_layers = [layer_names.index(layer) for layer in output_layers_names]
print("Output Layer Indices:", output_layers)  # Debugging: Print output layer indices

output_layers = [layer_names[i] for i in output_layers]  # Adjusted line
print("Output Layer Names:", output_layers)  # Debugging: Print output layer names

with open(yolo_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
    print("Boxes:", boxes)  # Debugging: Print boxes
    print("Indices:", indices)  # Debugging: Print indices
    objects = []

    if len(indices) > 0:
        indices = indices.flatten()  # Flatten the indices array if needed
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            objects.append({
                "class": classes[class_ids[i]],
                "confidence": confidences[i],
                "box": [x, y, w, h]
            })

    return objects
