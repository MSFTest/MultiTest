#!/usr/bin/python

import os.path as osp
import sys
import argparse
import os, tempfile, glob, shutil

# BASE_DIR = osp.dirname(__file__)
# sys.path.append(osp.join(BASE_DIR, '../../../../../PycharmProjects/MultiTest_com/MultiTest/'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))  # mark
from blender.global_variables import *

parser = argparse.ArgumentParser(description='Render Mo'
                                             'del Images of a certain class and view')
# parser.add_argument('-m', '--model_file', help='CAD Model obj filename', default=osp.join(BASE_DIR,'sample_model/model.obj'))
parser.add_argument('-m', '--model_file', help='CAD Model obj filename', )
parser.add_argument('-x', '--positionX', default='45')  # x
parser.add_argument('-y', '--positionY', default='20')  # y
parser.add_argument('-z', '--positionZ', default='0')  # z
parser.add_argument('-d', '--distance', default='2.0')  # 距离
parser.add_argument('-o', '--output_img', help='Output img filename.', default=osp.join(BASE_DIR, 'demo_img.png'))
parser.add_argument('-l', '--log')  # log存储目录
parser.add_argument('-c', '--calib')  # calib存储位置
parser.add_argument('-r', '--rz_degree')  # calib存储位置

args = parser.parse_args()

blank_file = osp.join(g_blank_blend_file_path)
render_code = osp.join(g_render4cnn_root_folder, 'render_model_views.py')

# MK TEMP DIR
temp_dirname = tempfile.mkdtemp()
view_file = osp.join(temp_dirname, 'view.txt')
view_fout = open(view_file, 'w')
view_fout.write(' '.join([args.positionX, args.positionY, args.positionZ, args.rz_degree]))
view_fout.close()

try:
    # print(args.calib)
    # render_cmd = '%s %s --background --python %s -- %s %s %s %s %s 1> %s/blender_log.txt' % (g_blender_executable_path, blank_file, render_code, args.model_file, 'xxx', 'xxx', view_file, temp_dirname, args.log)
    render_cmd = '%s %s --background --python %s -- %s %s %s %s %s %s 1> %s/blender_log.txt' % (
        g_blender_executable_path, blank_file, render_code, args.calib, args.model_file, 'xxx', 'xxx', view_file,
        temp_dirname, args.log)
    # print(render_cmd)
    os.system(render_cmd)
    imgs = glob.glob(temp_dirname + '/*.png')  # 获取当前目录下的所有png图片
    shutil.move(imgs[0], args.output_img)  # 移动到output_img位置
except:
    print('render failed. render_cmd: %s' % (render_cmd))

# CLEAN UP
shutil.rmtree(temp_dirname)
