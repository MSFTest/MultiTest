import os

from config import common_config as cf


def symlink(input_path, output_path):
    if not os.path.exists(input_path):
        raise ValueError("input: ", input_path)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.symlink(input_path, output_path)


def init_dirs():
    ...
    os.makedirs(os.path.join(cf.save_dir_guided), exist_ok=True)
    os.makedirs(os.path.join(cf.project_dir, "_queue_guided"), exist_ok=True)
    os.makedirs(os.path.join(cf.project_dir, "_workplace_dir"), exist_ok=True)


if __name__ == '__main__':
    init_dirs()
