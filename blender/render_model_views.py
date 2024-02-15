#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
RENDER_MODEL_VIEWS.py
brief:
	render projections of a 3D model from viewpoints specified by an input parameter file
usage:
	blender blank.blend --background --python render_model_views.py -- <shape_obj_filename> <shape_category_synset> <shape_model_md5> <shape_view_param_file> <syn_img_output_folder>

inputs:
       <shape_obj_filename>: .obj file of the 3D shape model
       <shape_category_synset>: synset string like '03001627' (chairs)
       <shape_model_md5>: md5 (as an ID) of the 3D shape model
       <shape_view_params_file>: txt file - each line is '<azimith angle> <elevation angle> <in-plane rotation angle> <distance>'
       <syn_img_output_folder>: output folder path for rendered images of this model

author: hao su, charles r. qi, yangyan li
'''



import os
import bpy
import sys
import math
import random
import numpy as np

# import sys
# sys.stdout = sys.stderr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
# import calibration_kitti
from blender import calibration_kitti

sys.path.append(os.environ['PROJECT_DIR'])
from blender.render_model_views_main import get_calibration_matrix_K_from_blender




def render_model(argv):
    # Input parameters
    K_file = argv[-6]
    shape_file = argv[-5]
    shape_synset = argv[-4]
    shape_md5 = argv[-3]
    shape_view_params_file = argv[-2]
    syn_images_folder = argv[-1]
    if not os.path.exists(syn_images_folder):
        os.mkdir(syn_images_folder)
    view_params = [[float(x) for x in line.strip().split(' ')] for line in open(shape_view_params_file).readlines()]

    bpy.ops.import_scene.gltf(filepath=shape_file)

    x = view_params[0][0]
    y = view_params[0][1]
    z = view_params[0][2]
    rz_degree = view_params[0][3]

    for obj in bpy.data.objects:
        if obj.name == 'Camera':
            continue
        else:
            import config

            obj.rotation_mode = 'XYZ'
            rz_degree_rad = math.radians(rz_degree)
            obj.rotation_euler = (np.pi / 2, rz_degree_rad, 0)
            scale_multiple = config.common_config.multi_scale_blender
            obj.scale = (scale_multiple, scale_multiple, scale_multiple)
            obj.location = (obj.location[0] + x, obj.location[1] + y, obj.location[2] + z)

    scene = bpy.context.scene
    scene.display.shading.color_type = 'TEXTURE'
    scene.display.render_aa = 'OFF'

    light_data = bpy.data.lights.new(name='my_light_data', type='SUN')
    light_data.energy = 5
    light = bpy.data.objects.new(name='my_light', object_data=light_data)
    light.location = ((x + 8), (y - 10), (z + 2))
    light.color = (1, 1, 1, 1)
    light.rotation_mode = 'XYZ'
    light.rotation_euler = (np.pi / 3, np.pi / 3, np.pi / 18)
    bpy.context.scene.collection.objects.link(light)

    calib_info = calibration_kitti.Calibration(K_file)
    K = calib_info.P2[:, 0:3]

    lens = config.camera_config.lens
    shift_x = config.camera_config.shift_x
    shift_y = config.camera_config.shift_y
    sensor_height = config.camera_config.sensor_height
    sensor_width = config.camera_config.sensor_width
    sensor_fit = "AUTO"
    image_width = config.camera_config.img_width
    image_height = config.camera_config.img_height
    location = config.camera_config.location
    coordinate_type = config.camera_config.coordinate_type

    if lens == "AUTO":
        lens = (K[0, 0] + K[1, 1]) / 2 * sensor_width / image_width
    if shift_x == "AUTO":
        shift_x = (image_width / 2 - K[0, 2]) / image_width
    if shift_y == "AUTO":
        shift_y = (K[1, 2] - image_height / 2) / image_width

    bpy.data.scenes['Scene'].render.film_transparent = True
    bpy.data.scenes['Scene'].render.resolution_x = image_width
    bpy.data.scenes['Scene'].render.resolution_y = image_height

    camObj = bpy.data.objects['Camera']
    camObj.location[0] = location[0]
    camObj.location[1] = location[1]
    camObj.location[2] = location[2]
    camObj.rotation_mode = coordinate_type
    camObj.rotation_euler = (np.pi, 0, 0)

    camObj.data.lens = lens
    camObj.data.shift_x = shift_x
    camObj.data.shift_y = shift_y
    camObj.data.sensor_height = sensor_height
    camObj.data.sensor_width = sensor_width
    camObj.data.sensor_fit = sensor_fit


    K2 = get_calibration_matrix_K_from_blender(camObj.data,bpy.context.scene)
    syn_image_file = './%s_%s_a%03d_e%03d_t%03d_d.png' % (
        shape_synset, shape_md5, round(x), round(y), round(z))
    bpy.data.scenes['Scene'].render.filepath = os.path.join(syn_images_folder, syn_image_file)
    bpy.ops.render.render(write_still=True)


render_model(sys.argv)