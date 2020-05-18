from pathlib import Path
import os
import re
import hashlib

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent

def get_files(root_folder, pattern):
    folders_and_files = os.listdir(root_folder + '/')
    files = []
    for i, el in enumerate(folders_and_files):
        if os.path.isfile(root_folder  + '/' + el):
            m = re.search(pattern, str(el))
            if m is not None:
                files.append(el)
    return files

def create_dir_if_not_exist(path):
    try:
        os.makedirs(path, exist_ok=False)
    except:
        pass
    return None


# if __name__ == '__main__':
#     hash_object = hashlib.md5(b'Hello World')
#     print(hash_object.hexdigest())




