from mnist import MNIST

import os
import gzip
import requests
import struct
import numpy as np
from io import BytesIO

FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}
FOLDER = "../data"
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"


def download_data(file_names, folder):
    os.makedirs(folder, exist_ok=True)
    for file in file_names:
        url = f"{BASE_URL}/{file}.gz"
        print(f"Downloading {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with gzip.open(BytesIO(response.content), "rb") as gz:
                data = gz.read()
            file_path = os.path.join(folder, file)
            with open(file_path, "wb") as out_file:
                out_file.write(data)
            print(f"Saved {file_path}")
        else:
            raise Exception(f"Failed to download {url} (status: {response.status_code})")
        
def check_data():
    os.makedirs(FOLDER, exist_ok=True)
    files_to_download = []
    for file in FILES.values():
        file_path = os.path.join(FOLDER, file)
        if not os.path.exists(file_path):
            files_to_download.append(file)
    if files_to_download:
        download_data(files_to_download, FOLDER)
        
        
def load_dataset():
    check_data()
    mndata = MNIST(FOLDER)
    training_images, training_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    
    training_images = np.array(training_images, dtype=np.uint8)
    training_labels = np.array(training_labels, dtype=np.uint8)
    test_images = np.array(test_images, dtype=np.uint8)
    test_labels = np.array(test_labels, dtype=np.uint8)
    
    return training_images, training_labels, test_images, test_labels 

if __name__ == "__main__":
    training_images, training_labels, test_images, test_labels = load_dataset()
    print(f"Loaded {len(training_images)} training images and {len(test_images)} test images.")
    print(training_labels[0])
