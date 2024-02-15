import os
import pickle
import shutil
import subprocess
from datetime import datetime

import fire
from natsort import natsorted

import config
from init import symlink


def add_python_path(p):
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = p
    else:
        os.environ["PYTHONPATH"] += ':{}'.format(p)


def remove_python_path(p):
    p = ':{}'.format(p)
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"].replace(p, "")


def construct_kitti_dataset_4_epnet(input_result_dir):
    output_result_dir = os.path.join(config.common_config.project_dir, "system/EPNet/data/KITTI/")

    symlink(os.path.join(input_result_dir, "training"),
            os.path.join(output_result_dir, "object/training"))
    symlink(os.path.join(input_result_dir, "ImageSets"),
            os.path.join(output_result_dir, "ImageSets"))


def copy_d2_data(config):
    ...


def inference_d2_results(input_dir, my_work_dir=None, rm_result_dir=True):
    from mmdetection.train_2D import my_inference as d2_inference
    from mmdetection.train_2D import get_d2_model
    from tqdm import tqdm
    model = get_d2_model(data_root=input_dir, my_work_dir=my_work_dir)
    python_path = '/home/niangao/PycharmProjects/Multimodality/mmdetection'
    add_python_path(python_path)

    image_2_dir = os.path.join(input_dir, "training", "image_2")
    txt_dir = os.path.join(input_dir, "training", "d2_detection_data")
    if rm_result_dir and os.path.exists(txt_dir):
        shutil.rmtree(txt_dir)
    os.makedirs(txt_dir, exist_ok=True)
    fns = os.listdir(image_2_dir)
    fns = natsorted(fns)
    for fn in tqdm(fns):
        img_path = os.path.join(image_2_dir, fn)
        _, d2_str_results = d2_inference(img_path, model=model)
        fn2 = fn.split(".")[0]
        txt_path = "{}/{}.txt".format(txt_dir, fn2)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        with open(txt_path, "w") as f:
            f.write(d2_str_results)
    remove_python_path(python_path)
    del model


def construct_kitti_dataset_4_fconv(input_result_dir, output_log_dir, my_work_dir=None, infer=False):
    output_result_dir = os.path.join(config.common_config.project_dir, "system/FConv/data/kitti")
    symlink(input_result_dir, output_result_dir)

    shutil.copyfile(os.path.join(output_result_dir, "ImageSets", "val.txt"),
                    os.path.join(config.common_config.project_dir, "system/FConv/kitti/image_sets/val.txt"))
    shutil.copyfile(os.path.join(output_result_dir, "ImageSets", "trainval.txt"),
                    os.path.join(config.common_config.project_dir, "system/FConv/kitti/image_sets/trainval.txt"))
    shutil.copyfile(os.path.join(output_result_dir, "ImageSets", "train.txt"),
                    os.path.join(config.common_config.project_dir, "system/FConv/kitti/image_sets/train.txt"))
    shutil.copyfile(os.path.join(output_result_dir, "ImageSets", "test.txt"),
                    os.path.join(config.common_config.project_dir, "system/FConv/kitti/image_sets/test.txt"))

    if infer:
        inference_d2_results(output_result_dir, my_work_dir)

    d2_dir = os.path.join(output_result_dir, "training", "d2_detection_data")
    cmd1 = "cd ./system/FConv"
    cmd2 = "python convert_2D_dector_result.py --from_dir {}".format(d2_dir)

    output_log_path = os.path.join(output_log_dir, "log_data_convert.txt")
    os.system("{} && {} > {} 2>&1".format(cmd1, cmd2, output_log_path))


def construct_kitti_dataset_4_clocs(input_result_dir, output_log_dir, my_work_dir=None, no_infer2d=False):
    pp = "{}/system/CLOCs".format(config.common_config.project_dir)
    add_python_path(pp)

    image_set_path = os.path.join(config.common_config.project_dir, "system/CLOCs/second/data/ImageSets")
    if os.path.exists(image_set_path):
        shutil.rmtree(image_set_path)
    shutil.copytree(os.path.join(input_result_dir, "ImageSets"), image_set_path)
    output_log_path = os.path.join(output_log_dir, "log_data_convert.txt")
    output_log_path = os.path.join(output_log_dir, "log_data_databse.txt")
    output_log_path2 = os.path.join(output_log_dir, "log_data_reduce.txt")

    os.system(f"python {pp}/second/create_data.py create_kitti_info_file "
              f"--data_path={input_result_dir} --imageset_path {image_set_path} > {output_log_path} 2>&1")
    os.system(f"python {pp}/second/create_data.py create_groundtruth_database "
              f"--data_path={input_result_dir} > {output_log_path} 2>&1")

    if not no_infer2d:
        inference_d2_results(input_result_dir, my_work_dir)

    os.makedirs(f"{input_result_dir}/training/velodyne_reduced", exist_ok=True)
    os.system(f"python {pp}/second/create_data.py create_reduced_point_cloud "
              f"--data_path={input_result_dir} > {output_log_path2} 2>&1")
    remove_python_path(pp)


def evaluate_epnet(output_result_dir, output_log_dir):
    name = "EPNet"
    project_dir = config.common_config.project_dir
    system_dir = os.path.join(project_dir, "system", name)
    output_log_path = os.path.join(output_log_dir, "log.txt")
    result_dir = f"{system_dir}/tools/log/Car/models/full_epnet_without_iou_branch/eval_results"
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    cmd1 = f"cd {system_dir}/tools"
    cmd2 = "CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml " \
           "--eval_mode rcnn_online    --output_dir ./log/Car/models/full_epnet_without_iou_branch/eval_results/  " \
           "--ckpt ./log/Car/models/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_45.pth --set  " \
           "LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 " \
           "RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False "
    os.system("{} && {} > {} 2>&1".format(cmd1, cmd2, output_log_path))

    input_result_dir = os.path.join(result_dir, "eval/epoch_45/val/eval/final_result/data")
    if os.path.exists(output_result_dir):
        shutil.rmtree(output_result_dir)
    shutil.copytree(input_result_dir, output_result_dir)


def evaluate_epnet_feature(output_log_dir):
    name = "EPNet"
    project_dir = config.common_config.project_dir
    system_dir = os.path.join(project_dir, "system", name)
    output_log_path = os.path.join(output_log_dir, "log.txt")

    cmd1 = f"cd {system_dir}/tools"
    cmd2 = "CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml " \
           "--eval_mode rcnn_online  --fid True  --output_dir ./log/Car/models/full_epnet_without_iou_branch/eval_results/  " \
           "--ckpt ./log/Car/models/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_45.pth --set  " \
           "LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 " \
           "RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False "
    os.system("{} && {} > {} 2>&1".format(cmd1, cmd2, output_log_path))


def run_epnet(guided=False, is_random=False):
    name = "EPNet"
    input_result_dir, output_result_dir, output_log_dir = get_data_dir(name, guided, is_random)
    construct_kitti_dataset_4_epnet(input_result_dir)
    os.makedirs(output_log_dir, exist_ok=True)
    evaluate_epnet(output_result_dir, output_log_dir)


def get_data_dir(name, guided=False, is_random=False):
    if guided:

        input_result_dir = os.path.join(config.common_config.workplace_dir, name, "kitti")
        output_result_dir = os.path.join(config.common_config.workplace_dir, name, "result")
        output_log_dir = os.path.join(config.common_config.workplace_dir, name, "log")
    elif is_random:

        from main import geninfo_ImageSets
        input_result_dir = os.path.join(config.common_config.kitti_aug_dataset_root, "random")
        geninfo_ImageSets(config.common_config.kitti_aug_dataset_root, "random")
        output_result_dir = os.path.join(input_result_dir, "training", "result_{}".format(name))
        output_log_dir = os.path.join(input_result_dir, "training", "log", "log_{}".format(name))
    else:

        input_result_dir = config.common_config.kitti_dataset_root
        output_result_dir = os.path.join(config.common_config.workplace_dir, name, "result_ori")
        output_log_dir = os.path.join(config.common_config.workplace_dir, name, "log")
    return input_result_dir, output_result_dir, output_log_dir


def evaluate_fconv(output_result_dir, output_log_dir):
    output_log_path = os.path.join(output_log_dir, "log.txt")
    input_result_dir = os.path.join(config.common_config.project_dir,
                                    "system/FConv/pretrained_models/car_refine/val_nms/result/data")
    if os.path.exists(input_result_dir):
        shutil.rmtree(input_result_dir)

    cmd1 = "cd ./system/FConv"
    cmd2 = "bash scripts/eval_pretrained_models.sh"
    os.system("{} && {} > {} 2>&1".format(cmd1, cmd2, output_log_path))

    if os.path.exists(output_result_dir):
        shutil.rmtree(output_result_dir)
    shutil.copytree(input_result_dir, output_result_dir)


def evaluate_clocs(output_result_dir, output_log_dir, guided=False, is_random=False):
    output_log_path = os.path.join(output_log_dir, "log.txt")

    input_result_dir = f"{config.common_config.project_dir}/system/CLOCs/trained_model/CLOCs_SecCas_pretrained/eval_results/step_30950"
    if os.path.exists(input_result_dir):
        shutil.rmtree(input_result_dir)

    pp = "{}/system/CLOCs".format(config.common_config.project_dir)
    add_python_path(pp)

    CLOCs_path = f"{config.common_config.project_dir}/system/CLOCs"

    if guided:
        cfp = f"{CLOCs_path}/second/configs/car_ins_guided.fhd.config"
    elif is_random:
        cfp = f"{CLOCs_path}/second/configs/car_ins_random.fhd.config"
    else:
        cfp = f"{CLOCs_path}/second/configs/car_ins_ori.fhd.config"

    cmd = f"python {CLOCs_path}/second/pytorch/train.py evaluate " \
          f"--config_path={cfp} " \
          f"--model_dir={CLOCs_path}/trained_model/CLOCs_SecCas_pretrained " \
          f"--measure_time=True --batch_size=1 --pickle_result=False --eval_map=False"
    os.system("{} > {} 2>&1".format(cmd, output_log_path))
    remove_python_path(pp)

    if os.path.exists(output_result_dir):
        shutil.rmtree(output_result_dir)
    shutil.copytree(input_result_dir, output_result_dir)


def run_fconv(guided=False, is_random=False):
    name = "FConv"
    input_result_dir, output_result_dir, output_log_dir = get_data_dir(name, guided, is_random)
    construct_kitti_dataset_4_fconv(input_result_dir, output_log_dir)
    os.makedirs(output_log_dir, exist_ok=True)
    evaluate_fconv(output_result_dir, output_log_dir)


def run_clocs(guided=False, is_random=False):
    ...

    name = "CLOCs"
    input_result_dir, output_result_dir, output_log_dir = get_data_dir(name, guided, is_random)
    os.makedirs(output_log_dir, exist_ok=True)
    construct_kitti_dataset_4_clocs(input_result_dir, output_log_dir)
    evaluate_clocs(output_result_dir, output_log_dir, guided=guided, is_random=is_random)


def construct_kitti_dataset_4_second(input_result_dir):
    symlink(os.path.join(input_result_dir, "testing"),
            f"{config.common_config.project_dir}/openpcdet/data/kitti/testing")
    symlink(os.path.join(input_result_dir, "training", "calib"),
            f"{config.common_config.project_dir}/openpcdet/data/kitti/training/calib")
    symlink(os.path.join(input_result_dir, "training", "image_2"),
            f"{config.common_config.project_dir}/openpcdet/data/kitti/training/image_2")
    symlink(os.path.join(input_result_dir, "training", "label_2"),
            f"{config.common_config.project_dir}/openpcdet/data/kitti/training/label_2")
    symlink(os.path.join(input_result_dir, "training", "velodyne"),
            f"{config.common_config.project_dir}/openpcdet/data/kitti/training/velodyne")
    symlink(os.path.join(input_result_dir, "ImageSets"),
            f"{config.common_config.project_dir}/openpcdet/data/kitti/ImageSets")


def run_rcnn(guided=False, is_random=False):
    name = "Rcnn"
    input_result_dir, output_result_dir, output_log_dir = get_data_dir(name, guided, is_random)
    os.makedirs(output_log_dir, exist_ok=True)

    inference_d2_results(input_result_dir)

    d2_dir = os.path.join(input_result_dir, "training", "d2_detection_data")
    if os.path.exists(output_result_dir):
        shutil.rmtree(output_result_dir)
    shutil.copytree(d2_dir, output_result_dir)


def run_second(guided=False, is_random=False):
    ...

    name = "Second"
    input_result_dir, output_result_dir, output_log_dir = get_data_dir(name, guided, is_random)
    construct_kitti_dataset_4_second(input_result_dir)
    os.makedirs(output_log_dir, exist_ok=True)
    evaluate_second(output_result_dir, output_log_dir, is_random)


def evaluate_second(output_result_dir, output_log_dir, is_random):
    if is_random:
        inference_d3_results('./env_log_random.sh', output_log_dir, clear_anno=True)
    else:
        inference_d3_results('./env_log.sh', output_log_dir, clear_anno=True)

    input_result_dir = f"{config.common_config.project_dir}/openpcdet/output/cfgs/kitti_models/second/default/" \
                       f"eval/epoch_7862/val/default/final_result/data"
    if os.path.exists(output_result_dir):
        shutil.rmtree(output_result_dir)
    shutil.copytree(input_result_dir, output_result_dir)


def inference_d3_results(bash_path, output_log_dir, clear_anno=True):
    os.system('export PATH="~/anaconda3/bin:$PATH"')

    bp = config.common_config.project_dir
    output_idr = os.path.join(bp, "openpcdet/output/cfgs/kitti_models/second/default/eval/epoch_7862/val/default/")
    if os.path.exists(output_idr):
        shutil.rmtree(output_idr)

    if clear_anno:
        for suffix in ["train", "val", "trainval", "test"]:
            kitti_info = os.path.join(bp, "openpcdet/data/kitti/kitti_infos_{}.pkl".format(suffix))
            if os.path.exists(kitti_info):
                os.remove(kitti_info)
        kitti_info = os.path.join(bp, "openpcdet/data/kitti/kitti_dbinfos_train.pkl.pkl")
        if os.path.exists(kitti_info):
            os.remove(kitti_info)
        gt_database_path = os.path.join(bp, "openpcdet/data/kitti/gt_database")
        if os.path.exists(gt_database_path):
            shutil.rmtree(gt_database_path)
    print("run", bash_path)
    p = subprocess.run([bash_path], shell=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        print(p.returncode)
        assert 1 == 2


def run_epnet_feature():
    op = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/fusion_feature/temp/"
    os.makedirs(op, exist_ok=True)
    data_root = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/_datasets"
    data_map = {
        "ori": f"{data_root}/kitti/",
        "random": f"{data_root}/kitti_construct/random/",
        "only_sim": f"{data_root}/kitti_construct/only_sim/",
    }

    key = "ori"
    ip = os.path.join(data_map[key], "training")
    op = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/system/EPNet/data/KITTI/object/training"
    symlink(ip, op)
    ip = os.path.join(data_map[key], "ImageSets")
    op = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/system/EPNet/data/KITTI/ImageSets"
    symlink(ip, op)
    name = "EPNet"
    output_log_dir = os.path.join(config.common_config.workplace_dir, name, "log")
    evaluate_epnet_feature(output_log_dir)


def gen_info():
    pc_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/_datasets/kitti/training/velodyne"
    info_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/_datasets/kitti/ImageSets/val.txt"
    if os.path.exists(info_path):
        os.remove(info_path)

    fns = natsorted(os.listdir(pc_dir))
    dir_seq_arr = []
    for fn_name in fns:
        dir_seq_arr.append(fn_name.split(".")[0])

    with open(info_path, "a") as f:
        for seq in dir_seq_arr:
            f.writelines(str(seq) + "\n")


if __name__ == '__main__':
    import time

    run_clocs()
