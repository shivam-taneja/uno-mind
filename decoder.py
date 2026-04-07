from ultralytics import YOLO
import pprint

model = YOLO("models/uno_model.pt")

pprint.pprint(model.names)
