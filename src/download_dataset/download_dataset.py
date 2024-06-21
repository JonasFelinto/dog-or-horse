import fiftyone as fo
import fiftyone.zoo as foz
import json
import os
'''
python -m src.download_dataset.download_dataset
'''
fo.config.dataset_zoo_dir = "./data"

class DatasetHandler:
    def __init__(self, dataset_source="open-images-v7",
                 max_samples=5000 ,
                 export_dir="./data/yolo_dataset"):
        """
        Initializes the DatasetHandler with empty dictionaries for dog and horse datasets
        and an empty list for classes.
        """
        self.dataset_dog = {}
        self.dataset_horse = {}
        self.classes = []
        self.dataset_source = dataset_source
        self.max_samples = max_samples
        self.export_dir = export_dir

    def download_datasets(self):
        """
        Downloads the datasets for dogs and horses for the train, validation, and test splits.
        """
        for split in ["train", "validation", "test"]:
            print("#" * 50)
            print(split)
            self.dataset_dog[split] = foz.load_zoo_dataset(
                self.dataset_source,
                split=split,
                label_types=["detections"],
                classes=["Dog"],
                max_samples=self.max_samples,
                dataset_name="dog_" + split
            )
            self.dataset_horse[split] = foz.load_zoo_dataset(
                self.dataset_source,
                split=split,
                label_types=["detections"],
                classes=["Horse"],
                max_samples=self.max_samples,
                dataset_name="horse_" + split
            )

    def load_classes(self):
        """
        Loads the class information from the JSON file into the classes attribute.
        """
        with open(os.path.join(fo.config.dataset_zoo_dir, self.dataset_source, "info.json"), 'r') as file:
            data = json.load(file)
            self.classes = data['classes']
        print(len(self.classes))

    def export_datasets(self):
        """
        Exports the downloaded datasets to YOLO format.
        """
        for split in ["train", "validation", "test"]:
            print("#" * 50)
            print(split)
            self.dataset_dog[split].export(
                export_dir=self.export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                classes=self.classes,
                split=split
            )
            self.dataset_horse[split].export(
                export_dir=self.export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                classes=self.classes,
                split=split
            )


def pipeline_download_dataset():
    handler = DatasetHandler()
    handler.download_datasets()
    handler.load_classes()
    handler.export_datasets()

if __name__ == "__main__":
    pipeline_download_dataset()