# from mtest.core.sensor_simulation.lidar_simulator import *
# from _typeshed import Incomplete

def get_occlusion_level(occlusion_ratio): ...
def get_care_labels(bg_labels): ...
def sort_labels(labels_input): ...
def del_labels(bg_labels, keys): ...
def update_bg_labels_care(bg_labels_care, bg_box_corners, pcd_bg, pcd_bg_update): ...
def write_labels_2(path, labels) -> None: ...
def read_labels_2(path, ignore_type): ...