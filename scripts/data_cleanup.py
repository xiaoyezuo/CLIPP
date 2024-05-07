import os

def remove_empty_dirs(path):
    #Walking through the directory tree
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Removed empty directory: {dirpath}")
            except OSError as error:
                print(f"Error removing {dirpath}: {error}")

directory_path = "/home/zuoxy/VLA-Nav/data/img/"
remove_empty_dirs(directory_path)

