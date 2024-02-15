import os
import sys
import config
os.environ['PROJECT_DIR'] = config.common_config.project_dir
sys.path.append(config.common_config.project_dir)

import argparse
import copy
import json
import math
import os.path
import shutil
import time
import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm

import importlib


# mtest2 = importlib.import_module("mtest")
# sys.path.append("/home/niangao/PycharmProjects/MultiTest_com/MultiTest/mtest/utils")
from mtest.core.sensor_simulation.lidar_simulator import complet_pc, lidar_simulation
from mtest.utils import calibration_kitti
from mtest.utils.Utils_label import read_labels_2, write_labels_2, sort_labels
from mtest.utils.Utils_o3d import pc_numpy_2_o3d, load_normalized_mesh_obj
from mtest.utils.Utils_common import read_Bin_PC, extact_initial_objs_from_bg, get_initial_box3d_in_bg, get_geometric_info, \
    update_occ_only_image, change_3dbox, get_truncation_ratio, get_labels
from mtest.core.occusion_handing.combine_img import combine_bg_with_obj
from mtest.core.occusion_handing.combine_pc import combine_pcd, update_init_label
from mtest.core.pose_estimulation.collision_detection import collision_detection, is_on_road
from mtest.core.pose_estimulation.pose_generator import generate_pose, tranform_mesh_by_pose, get_valid_pints
from mtest.core.pose_estimulation.road_split import road_split
from mtest.core.sensor_simulation.camera_simulator import camera_simulation
from logger import CLogger
from mtest.utils.box_utils import get_2d_box_from_image, get_2d_box_from_points, get_2d_box_center, trunc_2d_box, \
    iou_2d



from visual import show_img_with_labels

# print(os.getenv('PROJECT_DIR'))
# assert 1==2

def init_aug_dir(kitti_base_aug_dir, system_name):
    kitti_aug_dir = os.path.join(kitti_base_aug_dir, system_name)

    kitti_testing = os.path.join(kitti_aug_dir, "testing")
    os.makedirs(kitti_testing, exist_ok=True)
    kitti_aug_dir = os.path.join(kitti_aug_dir, "training")
    os.makedirs(kitti_aug_dir, exist_ok=True)
    sub_dirs = ["image_2", "label_2", "calib", "velodyne", "calib", "result", "label_2_insert"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(kitti_aug_dir, sub_dir), exist_ok=True)
    return kitti_aug_dir


def geninfo_ImageSets(kitti_base_aug_dir, system_name):
    kitti_aug_dir = os.path.join(kitti_base_aug_dir, system_name, "training")
    kitti_imagesets_dir = os.path.join(kitti_base_aug_dir, system_name, "ImageSets")
    os.makedirs(kitti_imagesets_dir, exist_ok=True)
    kitti_trainval_txt_file = "trainval.txt"
    kitti_val_txt_file = "val.txt"
    kitti_aug_train_val = os.path.join(kitti_imagesets_dir, kitti_trainval_txt_file)
    kitti_aug_val = os.path.join(kitti_imagesets_dir, kitti_val_txt_file)
    kitti_aug_train = os.path.join(kitti_imagesets_dir, "train.txt")
    kitti_aug_test = os.path.join(kitti_imagesets_dir, "test.txt")

    if os.path.exists(kitti_aug_train_val):
        os.remove(kitti_aug_train_val)
    if os.path.exists(kitti_aug_val):
        os.remove(kitti_aug_val)

    fns = natsorted(os.listdir(os.path.join(kitti_aug_dir, "label_2")))
    dir_seq_arr = []
    for fn_name in fns:
        dir_seq_arr.append(fn_name.split(".")[0])

    with open(kitti_aug_train_val, "a") as f:
        for seq in dir_seq_arr:
            f.writelines(str(seq) + "\n")

    with open(kitti_aug_val, "a") as f:
        for seq in dir_seq_arr:
            f.writelines(str(seq) + "\n")

    with open(kitti_aug_train, "w") as f:
        ...
    with open(kitti_aug_test, "w") as f:
        ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--system_name', type=str, default='demo')
    parser.add_argument('--select_size', type=int, default=100)
    parser.add_argument('--modality', type=str, default="multi")
    args = parser.parse_args()
    system_name = args.system_name
    select_size = args.select_size

    kitti_base_augdir = config.common_config.kitti_aug_dataset_root
    kitti_aug_dir = init_aug_dir(kitti_base_augdir, system_name)

    project_dir = config.common_config.project_dir
    workplace_dir = config.common_config.workplace_dir
    debug_log = os.path.join(config.common_config.project_dir, "debug_log.txt")
    bg_dir_path = config.common_config.bg_dir_path
    bg_road_dir = os.path.join(bg_dir_path, config.common_config.road_split_name)
    os.makedirs(bg_road_dir, exist_ok=True)
    bg_pc_dir_name = config.common_config.bg_pc_dir_name
    bg_img_dir_name = config.common_config.bg_img_dir_name
    bg_label_dir_name = config.common_config.bg_label_dir_name
    bg_calib_dir_name = config.common_config.bg_calib_dir_name
    bg_split_dir_name = config.common_config.bg_split_name

    assets_dir = config.common_config.obj_dir_path
    car_type_json_path = os.path.join(config.common_config.project_dir, "_assets", "car_type.json")
    obj_filename = config.common_config.obj_filename
    occlusion_th = config.common_config.occlusion_th

    modality = config.modality
    if system_name == "Second":
        modality = "pc"
    elif system_name == "Rcnn":
        modality = "image"
    objs_inserted_max_num = config.algorithm_config.objs_max_num

    score_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/RQ/RQ1/consistent"
    score_path = os.path.join(score_dir, f"{system_name}.txt")

    bg_num = len(os.listdir(os.path.join(bg_dir_path, bg_img_dir_name)))
    with open(car_type_json_path, "r") as f:
        car_type_dict = dict(json.load(f))
    obj_car_dirs = os.listdir(config.common_config.obj_dir_path)
    obj_num = len(obj_car_dirs)
    CLogger.info(
        f"system_name {system_name} select_size {select_size} object size: {obj_num}, "
        f"background size: {bg_num}, modality mode {modality}")

    CLogger.info(f"system_name: {system_name}")

    kitti_val_txt_path = os.path.join(config.common_config.kitti_dataset_root, "ImageSets", "val.txt")

    with open(kitti_val_txt_path, "r") as f:
        idx_arr = f.readlines()
    idx_arr = [int(x.strip()) for x in idx_arr]
    # np.random.seed(0)
    shuffle_idx = np.random.permutation(list(range(len(idx_arr))))
    idx_arr = np.array(idx_arr)[shuffle_idx]
    bg_index_list = idx_arr[:select_size]

    for bg_index in tqdm(bg_index_list):
        CLogger.info(f"select background {bg_index} ")

        csv_data = {}
        start_time = time.time()
        bg_pc_path = os.path.join(bg_dir_path, bg_pc_dir_name, f"{bg_index:06d}.bin")
        bg_img_path = os.path.join(bg_dir_path, bg_img_dir_name, f"{bg_index:06d}.png")
        bg_label_path = os.path.join(bg_dir_path, bg_label_dir_name, f"{bg_index:06d}.txt")
        bg_calib_path = os.path.join(bg_dir_path, bg_calib_dir_name, f"{bg_index:06d}.txt")

        save_dir = os.path.join(config.common_config.save_dir_guided, system_name, f"{bg_index:06d}")
        save_image_dir = os.path.join(save_dir, "image_2")
        save_image_dir_label = os.path.join(save_dir, "image_2_label")
        save_image_dir_fitness = os.path.join(save_dir, "image_2_score")
        save_image_dir_noref = os.path.join(save_dir, "image_2_noref")
        save_objs_image_dir = os.path.join(save_dir, "image_objs")
        save_pc_dir = os.path.join(save_dir, "velodyne")
        save_label_dir = os.path.join(save_dir, "label_2")
        save_log_dir = os.path.join(save_dir, "log")

        final_image_path = os.path.join(kitti_aug_dir, bg_img_dir_name, f"{bg_index:06d}.png")
        final_pc_path = os.path.join(kitti_aug_dir, bg_pc_dir_name, f"{bg_index:06d}.bin")
        final_label_path = os.path.join(kitti_aug_dir, bg_label_dir_name, f"{bg_index:06d}.txt")
        final_label_insert_path = os.path.join(kitti_aug_dir, bg_label_dir_name + "_insert", f"{bg_index:06d}.txt")
        final_calib_path = os.path.join(kitti_aug_dir, bg_calib_dir_name, f"{bg_index:06d}.txt")
        final_result_path = os.path.join(kitti_aug_dir, "result", f"{bg_index:06d}.txt")

        os.makedirs(save_pc_dir, exist_ok=True)
        os.makedirs(save_image_dir, exist_ok=True)
        os.makedirs(save_image_dir_label, exist_ok=True)
        os.makedirs(save_label_dir, exist_ok=True)
        os.makedirs(save_objs_image_dir, exist_ok=True)
        os.makedirs(save_image_dir_noref, exist_ok=True)
        os.makedirs(save_log_dir, exist_ok=True)

        mixed_pc_save_path = os.path.join(save_pc_dir, f"{bg_index:06d}.bin")
        save_label_path = os.path.join(save_label_dir, f'{bg_index:06d}.txt')

        img_bg = cv2.imread(bg_img_path)
        CLogger.debug("bg size:{}".format(img_bg.shape))

        bg_xyz = read_Bin_PC(bg_pc_path)
        pcd_bg = pc_numpy_2_o3d(bg_xyz)

        bg_labels = read_labels_2(bg_label_path)
        bg_labels_dont_care = []
        bg_labels_care = []
        for x in bg_labels:
            if "DontCare" in x:
                bg_labels_dont_care.append(x)
            else:
                bg_labels_care.append(x)

        calib_info = calibration_kitti.Calibration(bg_calib_path)

        road_pc, road_labels, non_road_pc, idx_road = road_split(bg_index, bg_pc_path, bg_road_dir, save_log_dir)
        if system_name == "only_road":
            continue
        if road_pc is None or len(road_pc) <= 10:
            CLogger.warning(f"{bg_index} don't have valid road")
            with open(debug_log, 'a') as f:
                f.writelines(f"{bg_index} don't have valid road")
            continue
        road_pc_valid = get_valid_pints(calib_info, road_pc)
        if len(road_pc_valid) == 0:
            CLogger.warning(f"*****{bg_index} don't have valid road")
            continue

        score_pre = 0
        _objs_index_arr = []
        _pcds = []
        _meshes = []
        _objs_img = []
        _coordinates = []
        _pcd_xyz_show_dict = {}

        _objs_half_diagonal = []
        _objs_center = []
        _labels_2 = []
        _labels_2_insert = []

        initial_boxes = []

        if modality is "pc" or modality is "multi":

            for i in range(len(bg_xyz)):
                _pcd_xyz_show_dict[i] = list(bg_xyz[i]) + [math.inf]

        info = extact_initial_objs_from_bg(calib_info, bg_label_path, bg_pc_path)

        initial_boxes, _objs_half_diagonal, _objs_center = get_initial_box3d_in_bg(info["corners_lidar"])

        assert len(bg_labels_care) == len(info["corners_lidar"]) == info['num_objs']
        i = 0
        j = 0
        k = 3
        cur_objs_inserted_max_num = np.random.randint(1, objs_inserted_max_num + 1)
        while i < cur_objs_inserted_max_num:
            if k == 0:
                i = np.inf
                CLogger.info("***obs not in image ,skip")
                break
            start_time = time.time()
            objs_index_arr = _objs_index_arr.copy()
            meshes = _meshes.copy()
            objs_half_diagonal = _objs_half_diagonal.copy()
            objs_center = _objs_center.copy()
            pcds = _pcds.copy()
            objs_img = _objs_img.copy()
            coordinates = _coordinates.copy()
            labels_ins = _labels_2.copy()
            pcd_xyz_show_dict = copy.deepcopy(_pcd_xyz_show_dict)
            try_num = config.algorithm_config.try_num
            np.random.seed(None)
            objs_index = np.random.randint(1, obj_num)
            obj_name = f"Car_{objs_index}"
            obj_mesh_path = os.path.join(assets_dir, obj_car_dirs[objs_index], obj_filename)
            mesh_obj_initial = load_normalized_mesh_obj(obj_mesh_path)
            road_num = 0
            while try_num > 0:
                CLogger.info(f"try {try_num}: inserting object {i + 1}")
                half_diagonal, _, _ = get_geometric_info(mesh_obj_initial)
                position, rz_degree = generate_pose(mesh_obj_initial, road_pc_valid, road_labels)
                CLogger.debug(f"position:{position},rz_degree:{rz_degree}")
                mesh_obj = tranform_mesh_by_pose(mesh_obj_initial, position, rz_degree)
                onroad_flag = is_on_road(mesh_obj, road_pc, non_road_pc)
                if not onroad_flag:
                    try_num -= 1
                    continue

                barycenter_xy = mesh_obj.get_center()[:2]
                success_flag = collision_detection(barycenter_xy, half_diagonal, objs_half_diagonal,
                                                   objs_center, len(initial_boxes))

                if not success_flag:
                    try_num -= 1
                    continue

                box_inserted_o3d = mesh_obj.get_minimal_oriented_bounding_box()

                break

            if try_num <= 0:
                if i == 0:
                    i = np.inf

                with open(debug_log, 'a') as f:
                    f.writelines(f"data {bg_index} insert {i} objs\n, finess {score_pre}, try_num=0")
                break

            box, angle = change_3dbox(box_inserted_o3d)

            coordinate = None
            box_image_trunc = None
            truncation_ratio = None
            if modality is "pc" or modality is "multi":
                pcd_obj = lidar_simulation(mesh_obj)
                pcds.append(pcd_obj)
            if modality is "image" or modality is "multi":

                img_obj_instance = camera_simulation(save_objs_image_dir, position, rz_degree,
                                                     obj_mesh_path,
                                                     obj_name,
                                                     mesh_obj, calib_info, save_log_dir, bg_calib_path)

                if len(img_obj_instance) == 1 or len(img_obj_instance) == 0:
                    k -= 1
                    CLogger.info("obj not in camera field")
                    continue

                objs_img.append(img_obj_instance)

                if modality is "image":

                    box_2d_8con, _ = calib_info.lidar_to_img(np.array(box.get_box_points()))
                    box_2d = get_2d_box_from_points(box_2d_8con)
                    box_2d_trunc = trunc_2d_box(box_2d, img_bg.shape[1], img_bg.shape[0])
                    coordinate_trunc = get_2d_box_center(box_2d_trunc)
                    box_image = get_2d_box_from_image(img_obj_instance, coordinate_trunc)
                    box_image_trunc = trunc_2d_box(box_image, img_bg.shape[1], img_bg.shape[0])

                    assert len(coordinate_trunc) == 2
                else:
                    img_pts_fonv, _ = calib_info.lidar_to_img(np.asarray(pcd_obj.points))
                    box_projection = get_2d_box_from_points(img_pts_fonv)
                    coordinate_ori = get_2d_box_center(box_projection)
                    box_projection_trunc = trunc_2d_box(box_projection, img_bg.shape[1], img_bg.shape[0])
                    coordinate_trunc = get_2d_box_center(box_projection_trunc)
                    box_image = get_2d_box_from_image(img_obj_instance, coordinate_trunc)
                    box_image_trunc = trunc_2d_box(box_image, img_bg.shape[1], img_bg.shape[0])

                truncation_ratio = get_truncation_ratio(
                    box_image, [0, 0, config.camera_config.img_width,
                                config.camera_config.img_height])
                coordinates.append(coordinate_trunc)
            if modality is "multi":

                consitance_score = iou_2d(box_projection_trunc, box_image_trunc)

                if consitance_score is not None and i == 0:
                    with open(score_path, "a") as f:
                        f.write(f"{str(np.round(consitance_score, 3))}\n")

            objs_index_arr.append(objs_index)
            meshes.append(mesh_obj)
            objs_half_diagonal.append(half_diagonal)
            objs_center.append(barycenter_xy)

            label_ins = get_labels(rz_degree, box, calib_info, box_image_trunc, truncation_ratio)

            labels_ins.append(label_ins)

            if modality is "image":
                labels_insert = labels_ins.copy()
                labels = np.concatenate([bg_labels, labels_ins], axis=0)
                labels = update_occ_only_image(labels.copy())
                for _l in labels:
                    if "DontCare" not in _l and "DontCa" in _l:
                        print("==", labels)
                        assert 1 == 2

            if modality is "pc" or modality is "multi":  # 点云与多模态显示
                combine_pc, labels_insert = combine_pcd(bg_xyz, pcds, meshes, labels_ins)
                corners_lidar = info['corners_lidar']
                bg_labels_update = update_init_label(bg_labels_care.copy(), bg_xyz, combine_pc, corners_lidar)
                bg_labels_update = bg_labels_update + bg_labels_dont_care
                labels = np.concatenate([labels_insert, bg_labels_update], axis=0)
                mixed_pc = complet_pc(combine_pc)
                mixed_pc = mixed_pc.astype(np.float32)  # KITTI使用float32格式存储为bin文件
            elif modality == "image":
                mixed_pc = complet_pc(bg_xyz.copy()).astype(np.float32)
            else:
                raise ValueError()
            mixed_pc.tofile(mixed_pc_save_path)

            mixed_img_save_filename = str(i) + "_" + "Car_" + "_".join([str(x) for x in objs_index_arr])
            mixed_img_save_path = f"{save_image_dir}/{mixed_img_save_filename}.png"
            if modality is "image" or modality is "multi":
                img_mix_gbr_no_refine, img_mix_gbr = combine_bg_with_obj(img_bg, objs_img, coordinates,
                                                                         objs_center[len(initial_boxes):], refine=False)
                img_mix_gbr_with_label = show_img_with_labels(img_mix_gbr.copy(), labels, is_shown=False)
                mixed_img_save_path_label = f"{save_image_dir_label}/{mixed_img_save_filename}.png"
                mixed_img_save_noref = f"{save_image_dir_noref}/{mixed_img_save_filename}.png"
                cv2.imwrite(mixed_img_save_noref, img_mix_gbr_no_refine)
                cv2.imwrite(mixed_img_save_path_label,
                            img_mix_gbr_with_label)
            elif modality == "pc":
                img_mix_gbr = img_bg.copy()
            else:
                raise ValueError()
            cv2.imwrite(mixed_img_save_path, img_mix_gbr)
            write_labels_2(save_label_path, sort_labels(labels))
            score = 0
            score_flag = True
            end_time = time.time()
            per_time = (end_time - start_time) / 60
            if score_flag:
                i += 1
                j = 0
                _objs_index_arr = objs_index_arr
                _meshes = meshes
                _objs_half_diagonal = objs_half_diagonal
                _objs_center = objs_center
                _pcds = pcds
                _objs_img = objs_img
                _coordinates = coordinates
                _labels_2 = labels_ins
                _pcd_xyz_show_dict = pcd_xyz_show_dict

        cv2.imwrite(final_image_path, img_mix_gbr)
        mixed_pc.tofile(final_pc_path)
        write_labels_2(final_label_path, sort_labels(labels))
        write_labels_2(final_label_insert_path, sort_labels(labels_insert))
        shutil.copyfile(bg_calib_path, final_calib_path)

        geninfo_ImageSets(kitti_base_augdir, system_name)


def bbox_overlaps_3d(anchors, gt_boxes):
    import torch
    assert anchors.dim() == 2 and gt_boxes.dim() == 2
    anchors = anchors[:, [0, 2, 3, 5, 1, 4]]
    gt_boxes = gt_boxes[:, [0, 2, 3, 5, 1, 4]]
    N = anchors.shape[0]
    K = gt_boxes.shape[0]
    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0])
                     * (gt_boxes[:, 3] - gt_boxes[:, 1])
                     * (gt_boxes[:, 5] - gt_boxes[:, 4])).view(1, K)
    anchors_area = ((anchors[:, 2] - anchors[:, 0])
                    * (anchors[:, 3] - anchors[:, 1])
                    * (anchors[:, 5] - anchors[:, 4])).view(N, 1)

    gt_area_zero = (gt_boxes_area == 0)
    anchors_area_zero = (anchors_area == 0)

    boxes = anchors.view(N, 1, 6).expand(N, K, 6)
    query_boxes = gt_boxes.view(1, K, 6).expand(N, K, 6)
    il = (torch.min(boxes[:, :, 2], query_boxes[:, :, 2]) - torch.max(boxes[:, :, 0], query_boxes[:, :, 0]))
    il[il < 0] = 0
    iw = (torch.min(boxes[:, :, 3], query_boxes[:, :, 3]) - torch.max(boxes[:, :, 1], query_boxes[:, :, 1]))
    iw[iw < 0] = 0
    ih = (torch.min(boxes[:, :, 5], query_boxes[:, :, 5]) - torch.max(boxes[:, :, 4], query_boxes[:, :, 4]))
    ih[ih < 0] = 0
    inter = il * iw * ih
    ua = anchors_area + gt_boxes_area - inter
    overlaps = inter / ua
    overlaps.masked_fill_(gt_area_zero.expand(N, K), 0)
    overlaps.masked_fill_(anchors_area_zero.expand(N, K), -1)

    return overlaps
