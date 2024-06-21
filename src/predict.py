import torch
from ultralytics import YOLO
import cv2
import argparse
import os

class YOLOInference:
    def __init__(self, model_path="runs/detect/train/weights/best.pt"):
        """
        Initialize the YOLO model with the given model path.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)

    def predict_video(self, video_path,imshow=True):
        """
        Perform object detection on a video and display the results.

        Args:
            video_path (str): Path to the video file.
        """
        cap = cv2.VideoCapture(video_path)

        dog = 0
        horse = 0
        list_frames = []
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = self.model(frame,verbose=False)
                for box in results[0].boxes:
                    if int(box.cls) == 0:
                        dog += 1
                    else:
                        horse += 1
                annotated_frame = results[0].plot()
                list_frames.append(annotated_frame)
                if imshow:
                    cv2.imshow("YOLOv8 Inference", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        print(f"dogs count: {dog}")
        print(f"horses count: {horse}")
        cap.release()
        cv2.destroyAllWindows()
        return list_frames

    def predict_one_image(self, img_path, imshow=True):
        """
        Perform object detection on a single image and print the results.

        Args:
            img_path (str): Path to the image file.
        """
        results = self.model(img_path)
        dog = 0
        horse = 0
        for box in results[0].boxes:
            if int(box.cls) == 0:
                dog += 1
            else:
                horse += 1
        print(f"dogs: {dog}")
        print(f"horses: {horse}")
        if imshow:
            cv2.imshow("YOLOv8 Inference", results[0].plot())
            print("press esc to continue")
            k = cv2.waitKey(5000)
            if k == 27:
                cv2.destroyAllWindows()
        return results[0].plot()

def main():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument('--path', type=str, required=True, help="Path to the video or image file")
    parser.add_argument('--model', type=str, default="runs/detect/train/weights/best.pt", help="Path to the model weights")

    args = parser.parse_args()

    yolo_inference = YOLOInference(model_path=args.model)

    # Determine if the path is to a video or an image based on the file extension
    file_extension = os.path.splitext(args.path)[1].lower()
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    if file_extension in video_extensions:
        yolo_inference.predict_video(args.path)
    elif file_extension in image_extensions:
        yolo_inference.predict_one_image(args.path)
    else:
        print("Unsupported file type. Please provide a valid video or image file.")

if __name__ == "__main__":
    main()
