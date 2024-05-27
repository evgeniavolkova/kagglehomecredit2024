"""Kaggle datasets utility functions."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from .config import KAGGLE_USERNAME, base_path


def update_dataset(dataset_id: str, source_path: str) -> None:
    """
    Update a Kaggle dataset.
    
    Arguments:
        dataset_id: The ID of the Kaggle dataset (e.g., "username/dataset-name").
        source_path: The local path where the dataset files are stored or the path to a single file to be uploaded.
    """
    original_source_path = Path(source_path)
    source_path = original_source_path
    
    metadata = {
        "id": f"{KAGGLE_USERNAME}/{dataset_id}",
    }
    
    metadata_path = source_path / 'dataset-metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    # Update the Kaggle dataset
    os.system(f'kaggle datasets version -p "{source_path}" -m "Updated dataset" -r zip')

    # Remove metadata
    os.remove(metadata_path)

def upload_code(dataset_id: str, source_path: str) -> None:
    """
    Upload a Python package to a Kaggle dataset.

    Arguments:
        dataset_id: The ID of the Kaggle dataset (e.g., "username/dataset-name").
        source_path: The local path where the Python package is stored.
    """
    os.chdir(base_path)
    os.system("python setup.py sdist bdist_wheel")

    original_source_path = Path(source_path)
    temp_dir = tempfile.mkdtemp()
    shutil.copy(original_source_path, temp_dir)
    source_path = Path(temp_dir)

    metadata = {
        "id": f"{KAGGLE_USERNAME}/{dataset_id}",
    }
    
    metadata_path = source_path / 'dataset-metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    os.system(f'kaggle datasets version -p "{source_path}" -m "Updated dataset"')

    os.remove(metadata_path)
    shutil.rmtree(temp_dir)