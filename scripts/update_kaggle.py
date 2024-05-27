"""Update Kaggle datasets: models, features, and code."""

import sys
from pathlib import Path

script_directory = Path(__file__).parent
homecredit_directory = script_directory.parent
sys.path.append(str(homecredit_directory))

from homecredit.kaggle import update_dataset, upload_code
from homecredit.config import PATH_MODELS, PATH_FEATURES, PATH_CODE

update_dataset("homecredit-models", PATH_MODELS)
update_dataset("homecredit-features", PATH_FEATURES)
upload_code("homecredit-code", PATH_CODE)