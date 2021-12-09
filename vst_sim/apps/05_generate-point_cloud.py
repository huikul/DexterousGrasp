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
from vstsim.grasping.quality import PointGraspMetrics3D
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
# file_dir = home_dir + "/dataset/ycb_meshes_google/backup/003_typical"

# os_path = os.environ
# print(os_path)

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


def worker(curr_obj, obj_name, resolution=24, flg_vis=False, path_save_grasp='', name_file=''):

    print("Log: do job", curr_obj)
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
    print("Log: do job", curr_obj)

    # file_gripper = home_dir + "/Dexterous_grasp_01/vst_sim/data/grippers/dexterous_vacuum/gripper.obj"
    file_gripper = work_dir + "/vst_sim/data/grippers/dexterous_vacuum/gripper.obj"

    mesh_gripper = None
    if flg_vis:
        of = ObjFile(file_gripper)
        mesh_gripper = of.read()

    # file_dir = home_dir + "/dataset/ycb_meshes_google/objects"
    # file_dir = home_dir + "/dataset/ycb_meshes_google/backup/002_basic"
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

    path_obj_grasp = path_save_grasp + '/' + name_file

    lst_grasp = pickle.load(open(path_obj_grasp, 'rb'))

    path_save_pc = path_save_grasp + "/r{}_2mm_disturb/".format(str(resolution))
    flg_success = \
        ags.generate_pc_PCL(obj, lst_grasp=lst_grasp,
                            path_save=path_save_pc,
                            vis=flg_vis, mesh_gripper=mesh_gripper,
                            resolution=int(ags.resolution_pc),
                            flg_disturb_bd=True, noise_bd=0.002,
                            flg_random_rotZ=False)


    print("Log: job {} done.".format(curr_obj))


def main():
    flg_vis = False
    resolution = 24
    '''
    home_dir = os.environ['HOME']
    path_save_grasp = home_dir + \
                      "/Dexterous_grasp_01/vst_sim/apps/generated_grasps/20201115"
    '''
    path_save_grasp = work_dir + \
                      "/vst_sim/apps/generated_grasps/20201115"

    for root, dirs, files in os.walk(path_save_grasp):
        # only for the 1st layer
        if root.count('/') == path_save_grasp.count('/'):
            lst_file_grasp = files

    # get the name list of objects
    lst_name_object = []
    for i in range(0, len(lst_file_grasp)):
        tmp_str = lst_file_grasp[i]
        ind_t = tmp_str.find('ttt')
        lst_name_object.append(tmp_str[0:ind_t-1])

    # file_list_all = get_file_name(file_dir)
    # object_numbers = file_list_all.__len__()

    job_list = np.arange(len(lst_file_grasp))
    job_list = list(job_list)
    cores = multiprocessing.cpu_count()

    # Initialize pool
    pool = []
    curr_obj = 0
    '''     Test    
    object_name = lst_name_object[curr_obj]
    path_grasp = lst_file_grasp[curr_obj]
    worker(curr_obj, object_name, resolution, True, path_save_grasp, path_grasp)
    '''
    #########################
    # number of jobs done at same time
    # pool_size = np.max([1, int(cores // 3)])
    pool_size = 6
    assert (pool_size <= len(job_list))
    for _ in range(pool_size):
        job_i = job_list.pop(0)
        object_name = lst_name_object[curr_obj]
        path_grasp = lst_file_grasp[curr_obj]
        pool.append(multiprocessing.Process(target=worker,
                                            args=(curr_obj,
                                                  object_name,
                                                  resolution,
                                                  flg_vis,
                                                  path_save_grasp,
                                                  path_grasp)))
        curr_obj += 1
    [p.start() for p in pool]

    while len(job_list) > 0:
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                job_i = job_list.pop(0)
                object_name = lst_name_object[curr_obj]
                path_grasp = lst_file_grasp[curr_obj]
                p = multiprocessing.Process(target=worker,
                                            args=(curr_obj,
                                                  object_name,
                                                  resolution,
                                                  flg_vis,
                                                  path_save_grasp,
                                                  path_grasp))
                curr_obj += 1
                p.start()
                pool.append(p)
                break

    print('All job done.')

    return True


if __name__ == '__main__':
    main()
