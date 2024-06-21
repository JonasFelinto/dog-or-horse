
import torch
from ultralytics import YOLO
'''
python -m src.train
'''

def train():
    # Load a YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(data="configs\dataset_dog_horse.yaml", epochs=200,device='0',cache=True,workers=2,mixup=0.75,close_mosaic=25)

    # Validate on training data
    model.val()

if __name__=='__main__':
    train()