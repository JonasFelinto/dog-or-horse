import torch
from ultralytics import YOLO
import argparse
'''
python -m src.validation --model-path path/to/your/model.pt --split test
'''
def main():
    """
    Main function to load a YOLOv8 model and run validation on the specified split.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate a YOLOv8 model on a specified split.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the YOLOv8 model file.")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'], help="Data split to use for validation (train, validation, or test).")
    args = parser.parse_args()

    # Load the YOLOv8 model
    model = YOLO(args.model_path)

    # Validate on the specified data split
    model.val(split=args.split)

if __name__ == '__main__':
    main()