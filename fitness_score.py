from mtest.fitness.fitness_score import cal_fitness_score2d, cal_fitness_score3d


def cal_fitness_score(dt_path, gt_path, system_name, iou_th=0.1, img=None, is_show=False, save_path=None, ix=0,
                      modality=None):
    ignore_type = ("Pedestrian", "Cyclist")
    if modality == "image":
        return cal_fitness_score2d(dt_path, gt_path, system_name, ignore_type=ignore_type,iou_th=iou_th, img=img, is_show=is_show,
                                   save_path=save_path, ix=ix)
    else:
        return cal_fitness_score3d(dt_path, gt_path, system_name, ignore_type=ignore_type,iou_th=iou_th, img=img, is_show=is_show,
                                   save_path=save_path, ix=ix)