from _typeshed import Incomplete

def get_iou3d(corners3d, query_corners3d, need_bev: bool = ...): ...
def get_iou2d(corners2d, query_corners2d): ...
def get_objs_attr(obj_list, has_image_box: bool = ...): ...
def fitness_score_localization(iou): ...
def fitness_score_recognition_dt(prob, distance): ...
def fitness_score_recognition_gt(distance): ...
def min_max_prob(prob, prob_max): ...
def get_prob_max(system_name): ...
def cal_fitness_score(dt_path, gt_path, system_name, iou_th: float = ..., img: Incomplete | None = ..., is_show: bool = ..., save_path: Incomplete | None = ..., ix: int = ..., modality: Incomplete | None = ...): ...
def cal_fitness_score2d(dt_path, gt_path, system_name, iou_th: float = ..., ignore_type: Incomplete | None = ..., img: Incomplete | None = ..., is_show: bool = ..., save_path: Incomplete | None = ..., ix: int = ...): ...
def cal_fitness_score3d(dt_path, gt_path, system_name, iou_th: float = ..., ignore_type: Incomplete | None = ..., img: Incomplete | None = ..., is_show: bool = ..., save_path: Incomplete | None = ..., ix: int = ...): ...
