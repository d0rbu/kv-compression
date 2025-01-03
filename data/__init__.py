import os

RAW_FILE_DIR = os.path.join(os.path.dirname(__file__), "raw")
RAW_FILES = {
    "alice": "alice29.txt"
}

# exports each file as a variable with the same name as the key in RAW_FILES
for file in RAW_FILES:
    with open(os.path.join(RAW_FILE_DIR, RAW_FILES[file])) as f:
        globals()[file] = f.read()
