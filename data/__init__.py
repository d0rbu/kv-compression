import os

RAW_FILE_DIR = os.path.join(os.path.dirname(__file__), "raw")
RAW_FILES = [filename for filename in os.listdir(RAW_FILE_DIR) if filename.endswith(".txt")]

# exports each file as a variable with the same name as the key in RAW_FILES
for filename in RAW_FILES:
    with open(os.path.join(RAW_FILE_DIR, filename)) as file:
        globals()[filename[:-4]] = file.read()
