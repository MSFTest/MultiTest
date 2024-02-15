occlusion_th = 0.95
occ_point_max = 20

project_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/MultiTest_com/MultiTest"
benchmark_dir = "{}/system".format(project_dir)
workplace_dir = "{}/_workplace".format(project_dir)
retrain_workplace_dir = "{}/_workplace_re".format(project_dir)

image_harmonization_model_path = "{}/third/S2CRNet/model/S2CRNet_pretrained.pth".format(project_dir)

road_split_dir = f"{project_dir}/third/CENet"
road_split_pc_dir = f"{project_dir}/third/CENet/data/sequences/00/velodyne"
road_split_label_dir = f"{project_dir}/third/CENet/result/sequences/00/predictions"

kitti_aug_dataset_root = "{}/_datasets/kitti_construct".format(project_dir)

kitti_retrain_dataset_root = "{}/_datasets/kitti_construct_retrain".format(project_dir)

kitti_dataset_root = "{}/_datasets/kitti".format(project_dir)

bg_dir_path = "{}/training".format(kitti_dataset_root)
bg_pc_dir_name = "velodyne"
bg_img_dir_name = "image_2"
bg_label_dir_name = "label_2"
bg_calib_dir_name = "calib"
bg_split_name = "ImageSets"
road_split_name = "road_label"

assets_dir = "{}/_assets".format(project_dir)
obj_dir_path = "{}/shapenet".format(assets_dir)
obj_cp_dir = "{}/copy_paste".format(assets_dir)

obj_filename = "models/model_normalized.gltf"

multi_scale = 5.5
multi_scale_blender = 5.5

save_dir = "{}/_queue".format(project_dir)
save_dir_guided = "{}/_queue_guided".format(project_dir)

distance_threshold = 0.01
ransac_n = 3
num_iterations = 1000

number_of_decimal = 6
road_range = [0, 80, -12, 12, 0.35]

image_vis_mode = "plt"
