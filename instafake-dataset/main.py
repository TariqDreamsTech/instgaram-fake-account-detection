import os
from utils import import_data

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "data")
dataset_version = "fake-v1.0"

fake_dataset = import_data(dataset_path, dataset_version)
