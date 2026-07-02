import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
"src/__init__.py",
"src/helper.py",
"src/prompt.py",
".env",
"setup.py",
"experiments.ipynb",
"app.py",
"store_index.py",
"static/.gitkeep",
"templates/chat.html",

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True) # create directory if not exists if not written exist ok true it will replace the existing directory with new one
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filename}")
    else:
        logging.info(f"{filename} already exists and is not empty. Skipping file creation.")