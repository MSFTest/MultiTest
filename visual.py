import numpy as np
import open3d as o3d
import cv2
import copy
import matplotlib.pyplot as plt

import config
from mtest.utils import box_utils, calibration_kitti, object3d_kitti
from mtest.utils.object3d_kitti import Object3d


def show_img_with_labels(image, labels, dt_labels=None, is_shown=False, dt_scores=None, title=None, save_path=None):
    image = image.copy()

    for label in labels:

        if label[0] == "DontCare":
            color = (100, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.rectangle(image, pt1=(int(float(label[4])), int(float(label[5]))),
                      pt2=(int(float(label[6])), int(float(label[7]))), color=color, thickness=2)
    if dt_labels is not None:
        for label in dt_labels:
            cv2.rectangle(image, pt1=(int(float(label[4])), int(float(label[5]))),
                          pt2=(int(float(label[6])), int(float(label[7]))), color=(0, 0, 255), thickness=2)
    if dt_scores is not None:
        for label, score in zip(dt_labels, dt_scores):
            score = np.round(score, 3)
            score_str = str(score)
            cv2.putText(
                image,
                score_str,
                (int(float(label[4])), int(float(label[5]))),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                5,
                cv2.LINE_AA)

    if is_shown:

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.imshow(image)
        if title is not None:
            plt.title(f"{title}")
        if save_path is not None:
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.xticks([])
            plt.yticks([])
            plt.show()
    return image


def show_mixed_pcd_with_inserted_labels(pcd_road, pcd_non_road, pc_objs, initial_boxes, meshes, temp_boxes):
    from mtest.utils.Utils_common import change_3dbox

    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)
    render.show_coordinate_frame = config.lidar_config.render_show_coordinate_frame

    vis.add_geometry(pcd_road)
    vis.add_geometry(pcd_non_road)

    for obj in pc_objs:
        vis.add_geometry(obj)

    for i in initial_boxes:
        vis.add_geometry(i)

    for i in temp_boxes:
        vis.add_geometry(i)

    for i in meshes:
        box3d = i.get_oriented_bounding_box()
        box3d.color = [0, 1, 0]

        box3d_initial_oriented = copy.deepcopy(box3d)
        box3d_initial_oriented.color = [1, 0, 0]

        box3d, angle = change_3dbox(box3d)

        vis.add_geometry(box3d_initial_oriented)
        vis.add_geometry(box3d)

    vis.run()
    vis.destroy_window()


def show_mixed_pcd(pcd_road, pcd_non_road, pc_objs, initial_boxes, meshes):
    from mtest.utils.Utils_common import change_3dbox

    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)
    render.show_coordinate_frame = config.lidar_config.render_show_coordinate_frame

    vis.add_geometry(pcd_road)
    vis.add_geometry(pcd_non_road)

    for obj in pc_objs:
        vis.add_geometry(obj)

    for i in initial_boxes:
        vis.add_geometry(i)

    for i in meshes:
        box3d = i.get_minimal_oriented_bounding_box()
        box3d2 = i.get_axis_aligned_bounding_box()
        box3d.color = [0, 1, 0]
        box3d2.color = [1, 1, 0]

        box3d_initial_oriented = copy.deepcopy(box3d)
        box3d_initial_oriented.color = [1, 0, 0]

        box3d, angle = change_3dbox(box3d)

        vis.add_geometry(box3d_initial_oriented)
        vis.add_geometry(box3d)
        vis.add_geometry(i)
        vis.add_geometry(box3d2)

    vis.run()
    vis.destroy_window()


def _get_corners(obj_list, calib_path):
    rots = np.array([obj.ry for obj in obj_list])
    dims = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    calib_info = calibration_kitti.Calibration(calib_path)
    loc_lidar = calib_info.rect_to_lidar(loc)
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
    return corners_lidar


def get_coners_from_label_path(label_path, calib_path, return_doncare=False):
    obj_list_ori = object3d_kitti.get_objects_from_label(label_path)
    obj_list = [obj for obj in obj_list_ori if obj.cls_type != 'DontCare']
    corners_lidar = _get_corners(obj_list, calib_path)
    if return_doncare:
        obj_list = [obj for obj in obj_list_ori if obj.cls_type == 'DontCare']
        if len(obj_list) == 0:
            return corners_lidar, []
        corners_lidar2 = _get_corners(obj_list, calib_path)
        return corners_lidar, corners_lidar2
    else:
        return corners_lidar


def get_coners_from_label_arr(label_arr, calib_path):
    obj_list = [Object3d(label=label) for label in label_arr]
    obj_list = [obj for obj in obj_list if obj.cls_type != 'DontCare']
    rots = np.array([obj.ry for obj in obj_list])
    dims = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    calib_info = calibration_kitti.Calibration(calib_path)
    loc_lidar = calib_info.rect_to_lidar(loc)
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
    return corners_lidar


def get_boxes_from_label(label, calib_path, color=[0, 1, 0]):
    from open3d.cuda.pybind.utility import Vector2iVector, Vector3dVector
    from open3d.cuda.pybind.geometry import LineSet
    if isinstance(label, str):
        gt_corners_lidar = get_coners_from_label_path(label, calib_path)
    else:
        gt_corners_lidar = get_coners_from_label_arr(label, calib_path)
    bbox_arr = []
    for corners_3d in gt_corners_lidar:
        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [color for _ in range(len(bbox_lines))]

        bbox = LineSet()
        bbox.lines = Vector2iVector(bbox_lines)
        bbox.colors = Vector3dVector(colors)
        bbox.points = Vector3dVector(corners_3d)
        bbox_arr.append(bbox)
    return bbox_arr


def change_view_detail(vis, p):
    import json
    with open(p, "r") as f:
        vis_map = dict(json.load(f))
    t = vis_map["trajectory"][0]
    front = t["front"]
    lookat = t["lookat"]
    up = t["up"]
    zoom = t["zoom"]
    ctr = vis.get_view_control()
    # print(t)

    ctr.set_lookat(lookat)
    ctr.set_up(np.array(up))  # set the positive direction of the x-axis as the up direction
    ctr.set_front(np.array(front))  # set the positive direction of the x-axis toward you
    ctr.set_zoom(zoom)
    # ctr.rotate(10.0, 0.0)
    vis.poll_events()
    vis.update_renderer()


def show_pc_with_labels(pc, label_path, calib_path, dt_path=None, view_path=None):
    from open3d.cuda.pybind.utility import Vector2iVector, Vector3dVector
    from open3d.cuda.pybind.geometry import LineSet
    gt_corners_lidar, gt_corners_lidar2 = get_coners_from_label_path(label_path, calib_path, return_doncare=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(pc)
    vis.add_geometry(pcd)
    print("gt_corners_lidar", len(gt_corners_lidar))
    print("gt_corners_lidar_dc", len(gt_corners_lidar2))

    # for corners_3d in gt_corners_lidar:
    #     bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    #     colors = [[0, 1, 0] for _ in range(len(bbox_lines))]
    #
    #     bbox = LineSet()
    #     bbox.lines = Vector2iVector(bbox_lines)
    #     bbox.colors = Vector3dVector(colors)
    #     bbox.points = Vector3dVector(corners_3d)
    #     vis.add_geometry(bbox)

    if dt_path is not None:
        dt_corners_lidar = get_coners_from_label_path(dt_path, calib_path)
        print("dt_corners_lidar", len(dt_corners_lidar))
        for corners_3d in dt_corners_lidar:
            bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6],
                          [3, 7]]
            colors = [[1, 0, 0] for _ in range(len(bbox_lines))]

            bbox = LineSet()
            bbox.lines = Vector2iVector(bbox_lines)
            bbox.colors = Vector3dVector(colors)
            bbox.points = Vector3dVector(corners_3d)
            vis.add_geometry(bbox)
    change_view_detail(vis,view_path)
    vis.run()
    vis.destroy_window()
