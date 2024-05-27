"""Set up the environment for the project."""

import os

from .config import COLAB, KAGGLE, base_path

def setup_environment():
    if COLAB:
        # Install required packages
        os.system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
        os.system('pip install catboost polars==0.20.5')
        os.system('pip install category_encoders')
        os.system('pip install wandb -qU')
        
        # Set up Kaggle API credentials
        os.system("mkdir -p ~/.kaggle")
        os.system(f"cp '{base_path}/kaggle.json' ~/.kaggle/")
        os.system("chmod 600 ~/.kaggle/kaggle.json")

        # Set up Weights & Biases
        import wandb
        wandb.login(key=os.environ.get("WANDB_KEY"))
    elif KAGGLE:
        # Install required packages
        os.system('pip install lightgbm==4.1.0 --no-index --find-links=file:///kaggle/input/lightgbm-4-1-0/')
        os.system('pip install polars==0.20.5 --no-index --find-links=file:///kaggle/input/polars/')
