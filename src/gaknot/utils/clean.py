#!/usr/bin/env python3

import os
import glob
import shutil

def clean_project():
    # Find all .sage files in the current directory
    sage_files = glob.glob("*.sage")
    
    # Generate the list of .py files that would have been created
    # We use os.path.splitext to safely handle filenames
    py_files_to_remove = [os.path.splitext(f)[0] + ".py" for f in sage_files]
    
    # Also include the intermediate .sage.py files if any were left behind
    py_files_to_remove += [f + ".py" for f in sage_files]

    for file in set(py_files_to_remove):
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except OSError as e:
                print(f"Error deleting {file}: {e}")

    # Clean up Python bytecode cache
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")
        print("Removed: __pycache__/")

if __name__ == "__main__":
    clean_project()
