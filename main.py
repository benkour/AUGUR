import json
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
# check if there are pt files in the processed_data directory
from pathlib import Path
processed_data_path = Path.cwd() / 'data_processed'
pt_files = list(processed_data_path.glob('*.pt'))
if len(pt_files) == 0 or config["reprocess_data_for_training"]:
    from create_data import *
    create_data_function()
if config["train"]:
    from create_models import *
    create_model_function()
if config["optimize"]:
    from optimize import *
    bo_optimization()
