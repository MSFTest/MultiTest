import os
import shutil

import numpy as np
from natsort import natsort
from tqdm import tqdm

import config as cf
import config.common_config
from init import symlink


def copy_data4epnet(dir_seq):
    system_name = "EPNet"
    return copy_data(dir_seq, system_name)


def copy_data4fconv(dir_seq):
    system_name = "FConv"
    return copy_data(dir_seq, system_name)


def copy_data4clocs(dir_seq):
    system_name = "CLOCs"
    return copy_data(dir_seq, system_name)


def copy_data4second(dir_seq):
    system_name = "Second"
    return copy_data(dir_seq, system_name)


def copy_data4rcnn(dir_seq):
    system_name = "Rcnn"
    return copy_data(dir_seq, system_name)


def copy_data(dir_seq, system_name):
    kitti_trainval_txt_file = "trainval.txt"
    kitti_val_txt_file = "val.txt"
    kitti_root_dir = cf.common_config.kitti_dataset_root
    kitti_aug_dir = cf.common_config.workplace_dir
    kitti_aug_dir = os.path.join(kitti_aug_dir, system_name, "kitti")
    os.makedirs(kitti_aug_dir, exist_ok=True)
    if os.path.exists(kitti_aug_dir):
        shutil.rmtree(kitti_aug_dir)
    kitti_imagesets_dir = os.path.join(kitti_aug_dir, "ImageSets")
    kitti_testing = os.path.join(kitti_aug_dir, "testing")
    os.makedirs(kitti_imagesets_dir, exist_ok=True)
    os.makedirs(kitti_testing, exist_ok=True)
    kitti_aug_train_val = os.path.join(kitti_imagesets_dir, kitti_trainval_txt_file)
    kitti_aug_val = os.path.join(kitti_imagesets_dir, kitti_val_txt_file)
    for _n in ["train", "test", "trainval"]:
        _p = os.path.join(kitti_imagesets_dir, _n + ".txt")
        with open(_p, "w") as f:
            ...

    if os.path.exists(kitti_aug_train_val):
        os.remove(kitti_aug_train_val)
    if os.path.exists(kitti_aug_val):
        os.remove(kitti_aug_val)
    kitti_aug_dir = os.path.join(kitti_aug_dir, "training")
    kitti_root_dir = os.path.join(kitti_root_dir, "training")
    os.makedirs(kitti_aug_dir, exist_ok=True)

    kitti_aug_dir_calib = os.path.join(kitti_aug_dir, "calib")
    kitti_root_dir_calib = os.path.join(kitti_root_dir, "calib")

    sub_dirs = ["image_2", "label_2", "calib", "velodyne", "velodyne_reduced"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(kitti_aug_dir, sub_dir), exist_ok=True)
    queue_dir = cf.common_config.save_dir_guided
    queue_dir = os.path.join(queue_dir, system_name)
    dir_seq_arr = []
    if not os.path.exists(os.path.join(queue_dir, dir_seq)):
        raise ValueError(os.path.join(queue_dir, dir_seq))
    image_dir = os.path.join(queue_dir, dir_seq, "image_2")
    to_image_dir = os.path.join(kitti_aug_dir, "image_2")

    pc_dir = os.path.join(queue_dir, dir_seq, "velodyne")
    to_pc_dir = os.path.join(kitti_aug_dir, "velodyne")

    insert_label_dir = os.path.join(queue_dir, dir_seq, "label_2")
    to_label_dir = os.path.join(kitti_aug_dir, "label_2")

    fns = natsort.natsorted(os.listdir(image_dir))
    if len(fns) == 0:
        raise ValueError(f"{image_dir}")
    else:

        dir_seq_arr.append(dir_seq)
        fn = fns[-1]

    shutil.copyfile(os.path.join(image_dir, fn), os.path.join(to_image_dir, "{}.png".format(dir_seq)))

    shutil.copyfile(os.path.join(kitti_root_dir_calib, "{}.txt".format(dir_seq)),
                    os.path.join(kitti_aug_dir_calib, "{}.txt".format(dir_seq)))

    shutil.copyfile(os.path.join(pc_dir, "{}.bin".format(dir_seq)),
                    os.path.join(to_pc_dir, "{}.bin".format(dir_seq)))

    shutil.copyfile(os.path.join(insert_label_dir, "{}.txt".format(dir_seq)),
                    os.path.join(to_label_dir, "{}.txt".format(dir_seq)))

    with open(kitti_aug_train_val, "a") as f:
        for seq in dir_seq_arr:
            f.writelines(str(seq) + "\n")

    with open(kitti_aug_val, "a") as f:
        for seq in dir_seq_arr:
            f.writelines(str(seq) + "\n")


def copy_ori_data4retrain(target_dir, train_size=100):
    kitti_dataset_root = config.common_config.kitti_dataset_root
    kitti_dataset_data_root = os.path.join(kitti_dataset_root, "training")
    kitti_imagesets_dir = os.path.join(kitti_dataset_root, "ImageSets")
    kitti_train_split = os.path.join(kitti_imagesets_dir, "train.txt")
    with open(kitti_train_split, "r") as f:
        idx_arr = f.readlines()
    idx_arr = [x.strip() for x in idx_arr]
    if idx_arr[-1] == "":
        del idx_arr[-1]
    np.random.seed(0)
    if train_size == "ALL":
        idx_train = np.array(idx_arr)
    else:
        idx_selected = np.random.permutation(len(idx_arr))[:train_size]
        idx_train = np.array(idx_arr)[idx_selected]
    sub_dirs = ["image_2", "label_2", "calib", "velodyne"]
    suffixs = ["png", "txt", "txt", "bin"]
    for idx in idx_train:
        for sub_dir, suffix in zip(sub_dirs, suffixs):
            pi_sub = os.path.join(kitti_dataset_data_root, sub_dir, f"{idx}.{suffix}")
            po_sub = os.path.join(target_dir, sub_dir, f"{idx}.{suffix}")
            shutil.copyfile(pi_sub, po_sub)
    return idx_train


def copy_data4retrain(kitti_aug_dir, kitti_workplace_dir, dir_seq_arr_train, dir_seq_arr_val, ori_train_size):
    kitti_aug_dir_data = os.path.join(kitti_aug_dir, "training")
    kitti_workplace_dir_data = os.path.join(kitti_workplace_dir, "training")
    kitti_workplace_dir_test = os.path.join(kitti_workplace_dir, "testing")

    sub_dirs = ["image_2", "label_2", "calib", "velodyne", "label_2_new"]
    for sub_dir in sub_dirs:
        po_sub = os.path.join(kitti_workplace_dir_data, sub_dir)
        po_test = os.path.join(kitti_workplace_dir_test, sub_dir)
        os.makedirs(po_test, exist_ok=True)
        if os.path.exists(po_sub):
            shutil.rmtree(po_sub)
        pi_sub = os.path.join(kitti_aug_dir_data, sub_dir)
        shutil.copytree(pi_sub, po_sub)

    ori_train_idx = copy_ori_data4retrain(kitti_workplace_dir_data, train_size=ori_train_size)
    dir_seq_arr_train = np.sort(np.concatenate([dir_seq_arr_train, ori_train_idx], axis=0))
    assert len(set(dir_seq_arr_train).intersection(set(dir_seq_arr_val))) == 0

    kitti_imagesets_dir = os.path.join(kitti_workplace_dir, "ImageSets")
    os.makedirs(kitti_imagesets_dir, exist_ok=True)

    kitti_aug_train = os.path.join(kitti_imagesets_dir, "train.txt")
    kitti_aug_val = os.path.join(kitti_imagesets_dir, "val.txt")
    kitti_aug_train_val = os.path.join(kitti_imagesets_dir, "trainval.txt")
    kitti_aug_test = os.path.join(kitti_imagesets_dir, "test.txt")

    with open(kitti_aug_train, "w") as f:
        pre = ""
        for seq in dir_seq_arr_train:
            f.writelines(pre + str(seq))
            pre = "\n"

    with open(kitti_aug_val, "w") as f:
        pre = ""
        for seq in dir_seq_arr_val:
            f.writelines(pre + str(seq))
            pre = "\n"

    with open(kitti_aug_test, "w") as f:
        ...

    with open(kitti_aug_train_val, "w") as f:
        pre = ""
        for seq in np.concatenate([dir_seq_arr_train, dir_seq_arr_val], axis=0):
            f.writelines(pre + str(seq))
            pre = "\n"


if __name__ == '__main__':
    ...
