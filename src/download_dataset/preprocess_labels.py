import yaml
import os
import argparse
from tqdm import tqdm

'''
python -m src.download_dataset.preprocess_labels --yaml_yolo=data\yolo_dataset\dataset.yaml --yaml_dog_horse=configs\dataset_dog_horse.yaml --labels_dir=data\yolo_dataset/labels
'''

class YamlLabelUpdater:
    def __init__(self, yaml1_path, yaml2_path, root_directory):
        self.yaml1_path = yaml1_path
        self.yaml2_path = yaml2_path
        self.root_directory = root_directory

    def find_index_mappings(self):
        """
        Reads two YAML files and returns a dictionary mapping the indices of common names
        from the first YAML to their corresponding indices in the second YAML.

        Returns:
            dict: A dictionary where keys are indices from the first YAML and values are
                  corresponding indices from the second YAML of the common names.
        """
        with open(self.yaml1_path, 'r') as file:
            yaml1_data = yaml.safe_load(file)

        with open(self.yaml2_path, 'r') as file:
            yaml2_data = yaml.safe_load(file)

        # Creating inverse dictionaries to map names to indices
        names_to_indices_yaml1 = {value: key for key, value in yaml1_data['names'].items()}
        names_to_indices_yaml2 = {value: key for key, value in yaml2_data['names'].items()}

        # Finding common names between the two YAML files
        common_names = set(names_to_indices_yaml1.keys()).intersection(names_to_indices_yaml2.keys())

        # Creating a mapping of indices from YAML1 to YAML2 for the common names
        index_mappings = {names_to_indices_yaml1[name]: names_to_indices_yaml2[name] for name in common_names}

        return index_mappings

    def update_labels(self, index_mapping):
        """
        Recursively traverses through directories starting from 'root_directory' to find and update label files (.txt).
        Each label file is updated based on the provided 'index_mapping' dictionary, which maps old indices to new indices.

        Parameters:
            index_mapping (dict): A dictionary mapping old class indices to new indices.
        """
        for root, dirs, files in os.walk(self.root_directory):
            for filename in tqdm(files):
                if filename.endswith('.txt'):
                    path = os.path.join(root, filename)
                    with open(path, 'r') as file:
                        lines = file.readlines()

                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        class_id = int(parts[0])

                        if class_id in index_mapping:
                            parts[0] = str(index_mapping[class_id])  # Update the index
                            new_lines.append(' '.join(parts))

                    with open(path, 'w') as file:
                        file.write('\n'.join(new_lines) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update label indices based on YAML files.')
    parser.add_argument('--yaml_yolo', required=True, help='Path to the first YAML file')
    parser.add_argument('--yaml_dog_horse', required=True, help='Path to the second YAML file')
    parser.add_argument('--labels_dir', required=True, help='Root directory for labels')

    args = parser.parse_args()

    updater = YamlLabelUpdater(args.yaml_yolo, args.yaml_dog_horse, args.labels_dir)
    index_mapping = updater.find_index_mappings()
    updater.update_labels(index_mapping)
