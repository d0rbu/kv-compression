import os

RAW_FILE_DIR = os.path.join(os.path.dirname(__file__), "raw")
RAW_FILES = [filename for filename in os.listdir(RAW_FILE_DIR) if filename.endswith(".txt")]

TEXT_DATA = {
    filename[:-4]: open(os.path.join(RAW_FILE_DIR, filename)).read()
    for filename in RAW_FILES
}
