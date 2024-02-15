import argparse
import os.path

from natsort import natsorted
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning, NumbaWarning
from tqdm import tqdm

import config
from mtest.utils.Utils_label import read_labels_2, get_care_labels
from evaluate_script import run_epnet, run_fconv, run_clocs, run_second, run_rcnn
from mtest.fitness.fitness_score import get_objs_attr, get_iou3d, get_iou2d, min_max_prob, get_prob_max
from mtest.fitness.ioutest import get_3d_boxes
from mtest.utils.object3d_kitti import Object3d
import numpy as np
import warnings
import pandas as pd

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


def cal_error_type(dt_dir, gt_dir, gt_split_file, iou_th=0.7, is_d2_box=False):
    location_error_cnt, dt_recognition_error_cnt, gt_recognition_error_cnt, serious_error_cnt, gt_cnt = 0, 0, 0, 0, 0
    with open(gt_split_file, "r") as f:
        idx_arr = f.readlines()
        idx_arr = [x.strip() for x in idx_arr]
        if idx_arr[-1] == "":
            del idx_arr[-1]
    for id in tqdm(idx_arr):
        dt_path = os.path.join(dt_dir, f"{id}.txt")
        gt_path = os.path.join(gt_dir, f"{id}.txt")
        try:
            a, b, c, d, e = _cal_error_type(dt_path, gt_path, iou_th=iou_th, is_d2_box=is_d2_box)
        except Exception as exce:
            a, b, c, d, e = 0, 0, 0, 0, 0
            print(exce)
        location_error_cnt += a
        dt_recognition_error_cnt += b
        gt_recognition_error_cnt += c
        serious_error_cnt += d
        gt_cnt += e
    print(
        f"location_error_cnt {location_error_cnt}, dt_recognition_error_cnt {dt_recognition_error_cnt}, "
        f"gt_recognition_error_cnt {gt_recognition_error_cnt},serious_error_cnt {serious_error_cnt}, gt_cnt{gt_cnt}")
    return location_error_cnt, dt_recognition_error_cnt, gt_recognition_error_cnt, serious_error_cnt


def _cal_error_type(dt_path, gt_path, iou_th=0.7, prob_th=0.5, dis_th=5, is_d2_box=False):
    ignore_type = ("Pedestrian", "Cyclist")
    dt_labels = read_labels_2(dt_path, ignore_type=ignore_type)  # , ignore_type=("DontCare",)
    gt_labels_ori = read_labels_2(gt_path, ignore_type=ignore_type)  # , ignore_type=("DontCare",)
    # print(gt_labels)
    gt_labels_care, gt_labels_dont_care = get_care_labels(gt_labels_ori, key="Car")
    gt_labels = gt_labels_care + gt_labels_dont_care
    assert len(gt_labels) == len(gt_labels_ori)
    gt_objects = [Object3d(label=line) for line in gt_labels]
    dt_objects = [Object3d(label=line) for line in dt_labels]

    gt_level_arr = [o.level for o in gt_objects]

    gt_objs_attr, _, gt_dis_arr, gt_image_box_arr = get_objs_attr(gt_objects, True)  # mark add 2d eval
    dt_objs_attr, dt_score_arr, dt_dis_arr, dt_image_box_arr = get_objs_attr(dt_objects, True)

    dt_corners_lidar = get_3d_boxes(np.array(dt_objs_attr))
    gt_corners_lidar = get_3d_boxes(np.array(gt_objs_attr))

    if is_d2_box:
        gtbox = gt_image_box_arr
        dtbox = dt_image_box_arr
    else:
        gtbox = gt_corners_lidar
        dtbox = dt_corners_lidar
    if len(dtbox) == 0 and len(gtbox) == 0:
        return 0, 0, 0, 0, len(gt_labels)
    elif len(gtbox) == 0:
        return 0, len(dtbox), 0, 0, len(gt_labels)  # mark
    elif len(dtbox) == 0:
        return 0, 0, len(gtbox), 0, len(gt_labels)  # mark
    if is_d2_box:
        ious = get_iou2d(dtbox, gtbox)
    else:
        ious = get_iou3d(dtbox, gtbox)
    iou_max_num = np.max(ious, axis=1)
    iou_max_index = np.argmax(ious, axis=1)
    # gt_index_set = set()
    location_error_cnt = 0
    gt_recognition_error_cnt = 0
    dt_recognition_error_cnt = 0
    serious_error = 0
    for dt_index, (iou_num, gt_index) in enumerate(zip(iou_max_num, iou_max_index)):
        dt_dis = dt_dis_arr[dt_index]
        prob = dt_score_arr[dt_index]
        prob_max = get_prob_max(system_name)
        prob = min_max_prob(prob, prob_max)
        flag = False
        if dt_dis < dis_th:
            flag = True
        if iou_num > 0:
            if iou_num < iou_th:
                if gt_index < len(gt_labels_care) and prob > prob_th:
                    # Localization failures
                    location_error_cnt += 1
                    # print(dt_path, prob, dt_index,iou_num)
                    if flag:
                        serious_error += 1
                else:
                    ...
                    # if prob > prob_th:  # iou with no care_lable very small
                    #     # det failures
                    #     dt_recognition_error_cnt += 1
                    #     if flag:
                    #         serious_error += 1
        else:
            # det failures
            if prob > prob_th:
                # print(dt_path, prob, dt_index)
                dt_recognition_error_cnt += 1
                if flag:
                    serious_error += 1

    gt_iou_max_num = np.max(ious, axis=0)
    assert len(gt_iou_max_num) == len(gt_labels)
    gt_index_set = np.array(list(range(0, len(gt_iou_max_num))))[gt_iou_max_num > 0]
    for i in range(len(gt_labels_care)):
        flag = False
        gt_dis = gt_dis_arr[i]
        gt_level = gt_level_arr[i]
        if gt_dis < dis_th:
            flag = True
        if i in gt_index_set:
            continue
        else:
            if 0 < gt_level < 2:
                ...
                # print(gt_path, i, gt_index_set, gt_labels_care[i],gt_level)
                # print(gtbox[i])
                # print(dtbox)
                gt_recognition_error_cnt += 1
            if flag:
                serious_error += 1

    return location_error_cnt, dt_recognition_error_cnt, gt_recognition_error_cnt, serious_error, len(gt_labels)


def cal_AP(det_path, gt_path, gt_split_file, prob_max):
    from eval_tools.eval import get_official_eval_result, get_coco_eval_result
    from eval_tools import kitti_common
    def _read_imageset_file(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return [int(line) for line in lines]

    # det_path = "/path/to/your_result_folder"
    val_image_ids = _read_imageset_file(gt_split_file)
    print(f"eval {len(val_image_ids)} data")
    dt_annos = kitti_common.get_label_annos(det_path, val_image_ids)
    # gt_path = "/path/to/your_gt_label_folder"
    # gt_split_file = "/path/to/val.txt"  # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
    gt_annos = kitti_common.get_label_annos(gt_path, val_image_ids)
    return get_official_eval_result(gt_annos, dt_annos, 0, prob_max=prob_max)  # 6s in my computer
    # print(get_coco_eval_result(gt_annos, dt_annos, 0))  # 18s in my computer


def get_common_id(p1, p2, p3, p4):
    l1 = natsorted(os.listdir(p1))
    l2 = natsorted(os.listdir(p2))
    l3 = natsorted(os.listdir(p3))
    l4 = natsorted(os.listdir(p4))

    common_ids = []
    for id in l1:
        if id in l2 and id in l3 and id in l4:
            common_ids.append(id)
    common_ids = [x.split(".")[0] for x in common_ids]
    return common_ids


# len(common_ids) 173
# ori dataset
# 100%|██████████| 173/173 [00:01<00:00, 140.39it/s]
# location_error_cnt 220, dt_recognition_error_cnt 542, gt_recognition_error_cnt 81
# random dataset
# 100%|██████████| 173/173 [00:01<00:00, 116.71it/s]
#   0%|          | 0/173 [00:00<?, ?it/s]location_error_cnt 393, dt_recognition_error_cnt 467, gt_recognition_error_cnt 222
# guided dataset
# 100%|██████████| 173/173 [00:01<00:00, 109.16it/s]
# location_error_cnt 356, dt_recognition_error_cnt 467, gt_recognition_error_cnt 250

def get_random_seq(max_num):
    np.random.seed(0)
    x = np.random.permutation(3000)
    x = x[x < max_num]
    np.random.seed(None)
    return x


def add_df(df, csv_data):
    if df is None:  # 如果是空的
        df = pd.DataFrame(csv_data, index=[0])
    else:
        df.loc[df.shape[0]] = csv_data
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arg parser")
    # parser.add_argument('--system_name', type=str, default='Rcnn')
    parser.add_argument('--system_name', type=str, default='CLOCs')
    parser.add_argument('--seed_num', type=int, default=200)
    args = parser.parse_args()
    system_name = args.system_name
    seed_num = args.seed_num

    perpare_data = False
    random_name = "random"
    iou_th = 0.5
    max_iter = 5

    is_d2_box = False
    if perpare_data:
        if system_name == "EPNet":
            run_epnet(is_random=True)
        elif system_name == "FConv":
            run_fconv(is_random=True)
        elif system_name == "CLOCs":
            run_clocs(is_random=True)
        elif system_name == "Second":
            run_second(is_random=True)
        elif system_name == "Rcnn":
            run_rcnn(is_random=True)
    if system_name == "Rcnn":
        is_d2_box = True
    prob_max = get_prob_max(system_name)
    print("config: ", "system_name", system_name, "iter", max_iter, "perpare_data", perpare_data, "seed_num", seed_num,
          "prob_max", prob_max)

    # is_d2_box = True
    data_root = config.common_config.kitti_aug_dataset_root
    data_ori = config.common_config.kitti_dataset_root
    RQ_dir = os.path.join(config.common_config.project_dir, "RQ", "RQ2")
    df_dir = os.path.join(RQ_dir, "csv", system_name)
    df_path = os.path.join(df_dir, "res.csv")
    df_mean_path = os.path.join(df_dir, "res_mean.csv")
    common_split_dir = os.path.join(RQ_dir, "common_split_file", str(max_iter))
    ap_dir = os.path.join(RQ_dir, "map", system_name, str(max_iter))
    fault_dir = os.path.join(RQ_dir, "fault", system_name, str(max_iter))
    os.makedirs(common_split_dir, exist_ok=True)
    os.makedirs(df_dir, exist_ok=True)
    os.makedirs(ap_dir, exist_ok=True)
    os.makedirs(fault_dir, exist_ok=True)
    gt_split_file = os.path.join(common_split_dir, "common_val.txt")
    if os.path.exists(gt_split_file):
        os.remove(gt_split_file)


    dt_path_guided = os.path.join(data_root, system_name, "training", "result")
    # dt_path_guided = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/_workplace_re/Second/kitti/training/result_before"
    gt_path_guided = os.path.join(data_root, system_name, "training", "label_2")

    dt_path_random = os.path.join(data_root, random_name, "training", "result_{}".format(system_name))
    gt_path_random = os.path.join(data_root, random_name, "training", "label_2")

    dt_path_ori = os.path.join(config.common_config.workplace_dir, system_name, "result_ori")
    gt_path_ori = os.path.join(data_ori, "training", "label_2")

    guided_split = natsorted(os.listdir(gt_path_guided))  # rcnn only 842
    random_split = natsorted(os.listdir(gt_path_random))
    # dt_split = natsorted(os.listdir(dt_path_guided))
    print(len(guided_split))
    print(len(random_split))
    df = None

    for iter in range(max_iter):
        # if iter == 1:
        #     break
        common_ids = []
        for id in guided_split:
            if id in random_split:
                # if id in dt_split:
                common_ids.append(id)
        shuffle_ix = get_random_seq(len(common_ids))
        common_ids = np.array(common_ids)[shuffle_ix]
        if len(common_ids) <= seed_num * max_iter:
            common_ids2 = common_ids[get_random_seq(len(common_ids))][:seed_num * max_iter - len(common_ids)]
            common_ids = np.concatenate([common_ids, common_ids2], axis=0)
        assert len(common_ids) >= seed_num * max_iter
        common_ids = common_ids[iter * seed_num: (iter + 1) * seed_num]
        # common_ids = common_ids
        # print(len(common_ids))
        # if len(common_ids) != seed_num:
        #     print("warning len(common_ids) != seed_num")
        #     common_ids2 = np.array(common_ids[:seed_num - len(common_ids)])

        # common_ids = ["000002"] *50
        # print(common_ids[10:20])
        # assert 1==2
        # common_ids = common_ids[-50:]
        # common_ids = [common_ids[-1]]
        with open(gt_split_file, "w") as f:
            for id in common_ids:
                id = id.split(".")[0]
                f.writelines(str(id) + "\n")
        # print("len(common_ids)", len(common_ids))

        # # assert 1 == 2
        # print(f"eval {len(common_ids)} data")
        # print("=" * 10, "Ori dataset", "=" * 10)
        # with open(os.path.join(ap_dir, "ori.txt"), "w") as f:
        #     res, res_dict = cal_AP(dt_path_ori, gt_path_ori, gt_split_file, prob_max)
        #     res_dict["name"] = "ori"
        #     res_dict["iter"] = iter
        #     df = add_df(df, res_dict)
        #     f.writelines(res)
        # # print(res)
        # print("=" * 10, "Random dataset", "=" * 10)
        # with open(os.path.join(ap_dir, "random.txt"), "w") as f:
        #     res, res_dict = cal_AP(dt_path_random, gt_path_random, gt_split_file, prob_max)
        #     res_dict["name"] = "random"
        #     res_dict["iter"] = iter
        #     df = add_df(df, res_dict)
        #     f.writelines(res)
        # print(res)
        print("=" * 10, "Guided dataset", "=" * 10)
        with open(os.path.join(ap_dir, "guided.txt"), "w") as f:
            res, res_dict = cal_AP(dt_path_guided, gt_path_guided, gt_split_file, prob_max)
            print(res)
            res_dict["name"] = "guided"
            res_dict["iter"] = iter
            df = add_df(df, res_dict)
            f.writelines(res)
        df = df.sort_values(by=["name", "iter"], ascending=False)
        df.to_csv(df_path)

        # mean value
        df_mean_arr = []
        name_arr = ["guided", "random", "ori"]
        for name in name_arr:
            df_nam = df[df["name"] == name]
            del df_nam["name"]
            del df_nam["iter"]
            df_mean = df_nam.mean(axis=0)
            df_mean_arr.append(df_mean)
        df_mean_total = pd.concat(df_mean_arr, axis=1)
        df_mean_total.columns = name_arr
        df_mean_total.to_csv(df_mean_path)

        #
        print("ori dataset")
        with open(os.path.join(fault_dir, "ori.txt"), "w") as f:
            a, b, c, d = cal_error_type(dt_path_ori, gt_path_ori, gt_split_file, iou_th, is_d2_box)
            lines = (
                f"location_error_cnt, {a}\n"
                f"dt_recognition_error_cnt, {b}\n"
                f"gt_recognition_error_cnt, {c}\n"
                f"serious_error_cnt, {d}")
            f.writelines(lines)
        print("random dataset")
        with open(os.path.join(fault_dir, "random.txt"), "w") as f:
            a, b, c, d = cal_error_type(dt_path_random, gt_path_random, gt_split_file, iou_th, is_d2_box)

            lines = (
                f"location_error_cnt, {a}\n"
                f"dt_recognition_error_cnt, {b}\n"
                f"gt_recognition_error_cnt, {c}\n"
                f"serious_error_cnt, {d}")
            f.writelines(lines)
        print("guided dataset")
        with open(os.path.join(fault_dir, "guided.txt"), "w") as f:
            a, b, c, d = cal_error_type(dt_path_guided, gt_path_guided, gt_split_file, iou_th, is_d2_box)
            lines = (
                f"location_error_cnt, {a}\n"
                f"dt_recognition_error_cnt, {b}\n"
                f"gt_recognition_error_cnt, {c}\n"
                f"serious_error_cnt, {d}")
            print(lines)
            f.writelines(lines)




