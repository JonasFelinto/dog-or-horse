# Dog or Horse

<img src="midia_test\imgs_results\i3.png" alt="" />

Please review the results in the `media_test` folder and explore the metric demonstration notebooks in the `notebooks` folder.

# Overview

This project demonstrates how to use YOLO for training on a custom dataset. For testing purposes, we utilized the dataset from Google, which can be accessed [here](https://storage.googleapis.com/openimages/web/visualizer/index.html). To gain a deeper understanding of YOLO, visit the [official YOLO documentation page](https://docs.ultralytics.com/).


# Environment Setup and Dataset Preparation

This guide provides instructions on setting up the Python environment, installing necessary dependencies, and downloading the dataset required for model training.

## 1. Create Python Environment

First, create a new Python environment using Conda:

```bash
conda create --name verifymy python=3.8
```

Activate the environment:

```bash
conda activate verifymy
```

## 2. Install CUDA

For faster training, install CUDA. Training without CUDA will be significantly slower. Follow the installation instructions from the [CUDA Python Install Guide](https://nvidia.github.io/cuda-python/install.html).

## 3. Install PyTorch with CUDA Support

Install PyTorch along with its dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## 4. Verify CUDA Installation

To verify if CUDA is properly installed and working, run the following command:

```bash
python -m src.check_cuda
```

If everything is set up correctly, you should see output similar to:

```
Cuda available:  True
Device name: NVIDIA GeForce RTX 4070 Ti
```

## 5. Install Python Package Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 6. Download Dataset

To download the Open Images V7 dataset containing images of dogs and horses, use the following command:

```bash
python -m src.download_dataset.download_dataset
```

By default, this command attempts to download 5,000 images each of dogs and horses across all sets. If you lack sufficient disk space (~14GB), you can adjust the `max_samples` size in `src/download_dataset/download_dataset.py`.

The next script will delete other labels, keeping only those for dogs and horses according to the YAML file located in the configs folder. 

**WARNING:** Running this script multiple times in sequence can delete all labels, as the mapping does not work sequentially.

```bash
python -m src.download_dataset.preprocess_labels --yaml_yolo=data/yolo_dataset/dataset.yaml --yaml_dog_horse=configs/dataset_dog_horse.yaml --labels_dir=data/yolo_dataset/labels

```
The dataset will be downloaded to the `data` folder by default.

## Training, Validation, and Prediction

Run the training script:

```bash
python -m src.train
```

This command runs the training for the YOLO Nano model using mostly default parameters. If CUDA is not installed or you prefer to run on CPU (not recommended), modify the `src/train.py` script.

To validate and assess the results, execute:

```bash
python -m src.validation --model-path="model.pt" --split="val"
```

Specify the `--model-path` parameter as the path of the model to be evaluated, and the `--split` parameter as the split used in validation. You can use the val or test split to assess the results on each dataset.

For predictions, run:

```bash
python -m src.predict --path="video_path.mp4"
```
or 

```bash
python -m src.predict --path="image_path.png"
```

## Additional Resources

In the `notebooks` folder, there are a results notebook and a demonstration notebook to showcase the model's use with three images and three videos from the internet. The results have been saved in the `media_test` folder.
