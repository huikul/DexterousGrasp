#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hui Zhang 
# E-mail     : hui.zhang@kuleuven.be
# Description: 
# Date       : 23/02/2020 21:37
# File Name  : generate-dataset-vacuumG.py
import numpy as np
import sys
import pickle
# from vstsim.grasping.quality import PointGraspMetrics3D
from vstsim.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler, \
    VacuumGraspSampler, DexterousVacuumGrasp
from vstsim.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
from vstsim.visualization.visualizer3d import DexNetVisualizer3D as Vis
import vstsim
from autolab_core import YamlConfig
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
import os
import multiprocessing
import matplotlib.pyplot as plt
from mayavi import mlab

import logging
try:
    import pcl
    import pcl.pcl_visualization
except ImportError:
    logging.warning('Failed to import pcl!')


# home_dir = os.environ['HOME']

work_dir = os.environ['PWD'][:-13]
file_dir = work_dir + "/3D_meshes"



def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        if root.count('/') == file_dir_.count('/') + 1:
            file_list.append(root)
    file_list.sort()
    return file_list

''''''
def display_object(obj_):
    """display object only using mayavi"""
    Vis.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    Vis.mesh(obj_.mesh.trimesh, color=(0.5, 0.5, 0.5), style='surface')
    Vis.show()


def worker(obj_name, resolution=24, flg_vis=False):

    print("Log: do job")
    object_name = obj_name
    print('a worker of task {} start'.format(object_name))

    # yaml_config = YamlConfig(home_dir + "/Dexterous_grasp_01/vst_sim/test/config.yaml")
    yaml_config = YamlConfig(work_dir + "/vst_sim/test/config.yaml")

    gripper_name = "dexterous_vacuum"
    # gripper = RobotGripper.load_dex_vacuum(gripper_name, home_dir + "/Dexterous_grasp_01/vst_sim/data/grippers")
    gripper = RobotGripper.load_dex_vacuum(gripper_name, work_dir + "/vst_sim/data/grippers")

    grasp_sample_method = "dexterous_vacuum"
    if grasp_sample_method == "uniform":
        ags = UniformGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "vacuum_point":
        ags = VacuumGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "dexterous_vacuum":
        ags = DexterousVacuumGrasp(gripper, yaml_config)

    else:
        raise NameError("Can't support this sampler")

    # file_gripper = home_dir + "/Dexterous_grasp_01/vst_sim/data/grippers/dexterous_vacuum/gripper.obj"
    file_gripper = work_dir + "/vst_sim/data/grippers/dexterous_vacuum/gripper.obj"

    mesh_gripper = None
    if flg_vis:
        of = ObjFile(file_gripper)
        mesh_gripper = of.read()

    # file_dir = home_dir + "/dataset/ycb_meshes_google/objects"
    # file_dir = home_dir + "/dataset/ycb_meshes_google/backup/002_basic"
    # file_dir = home_dir + "/dataset/ycb_meshes_google/backup/003_typical"
    file_dir = work_dir + "/3D_meshes"


    if os.path.exists(file_dir + "/" + obj_name + "/google_512k/nontextured.obj"):
        of = ObjFile(file_dir + "/" + obj_name + "/google_512k/nontextured.obj")
        sf = SdfFile(file_dir + "/" + obj_name + "/google_512k/nontextured.sdf")
        nf = np.load(file_dir + "/" + obj_name + "/google_512k/surface_normals_pcl.npy")
    else:
        print("can't find any obj or sdf file!")
        raise NameError("can't find any obj or sdf file!")
    mesh = of.read()
    sdf = sf.read()
    mesh.set_normals(nf)
    obj = GraspableObject3D(sdf, mesh)

    flg_success = \
        ags.generate_pc_PCL_vis(obj, mesh_gripper=mesh_gripper, resolution=int(ags.resolution_pc),
                                flg_disturb_bd=True, noise_bd=0.008,
                                flg_random_rotZ=False)

def main():
    flg_vis = True
    resolution = 24

    # home_dir = os.environ['HOME']

    obj_name = "011_banana"
    worker(obj_name, resolution=resolution, flg_vis=flg_vis)



    return True


if __name__ == '__main__':
    main()
