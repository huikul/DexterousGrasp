#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hui Zhang 
# E-mail     : hui.zhang@kuleuven.be
# Description: 
# Date       : 21/10/2020 14:20
# File Name  : generate-dataset-Dexterous_vacuum.py
import numpy as np
import sys
import pickle
import datetime
from vstsim.grasping.quality import PointGraspMetrics3D
from vstsim.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler, \
    VacuumGraspSampler, DexterousVacuumGrasp
from vstsim.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
# from vstsim.visualization.visualizer3d import DexNetVisualizer3D as Vis
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

'''
def display_object(obj_):
    """display object only using mayavi"""
    Vis.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    Vis.mesh(obj_.mesh.trimesh, color=(0.5, 0.5, 0.5), style='surface')
    Vis.show()
'''

def worker(curr_obj, obj_name, sample_nums, grasp_generations, flg_vis=False, flg_test=False):
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
    obj = GraspableObject3D(sdf, mesh, model_name=str(obj_name))

    print("dim_grasp_matrix: ", int(ags.dim_grasp_matrix))
    grasps = \
        ags.generate_grasps_dex_vacuum(obj,
                                       target_num_grasps=sample_nums,
                                       grasp_gen_mult=grasp_generations,
                                       multi_approach_angle=True,
                                       vis_surface=flg_vis, mesh_gripper=mesh_gripper,
                                       flg_test=flg_test,
                                       flg_desample_g=True, dim_g_matrix=int(ags.dim_grasp_matrix),
                                       num_repeat_QP=3)
    str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    '''
    grasps_file_name = home_dir + "/Dexterous_grasp_01/vst_sim/apps/generated_grasps/{}_ttt{}_d{}_a{}_s{}_n{}".format(
        object_name,
        str(str_time[0:12]),
        str(int(ags.dim_grasp_matrix)),
        str(int(ags.angle_range_max)),
        str(int(ags.num_angle_steps)),
        str(np.shape(grasps)[0]))
    '''
    grasps_file_name = work_dir + "/vst_sim/apps/generated_grasps/{}_ttt{}_d{}_a{}_s{}_n{}".format(
        object_name,
        str(str_time[0:12]),
        str(int(ags.dim_grasp_matrix)),
        str(int(ags.angle_range_max)),
        str(int(ags.num_angle_steps)),
        str(np.shape(grasps)[0]))

    # np.save(grasps_file_name + '.npy', np.array(grasps))
    with open(grasps_file_name + '.pickle', 'wb') as f:
        pickle.dump(grasps, f)


def main():

    target_num_grasps = 8

    file_list_all = get_file_name(file_dir)
    object_numbers = file_list_all.__len__()

    job_list = np.arange(object_numbers)
    job_list = list(job_list)

    cores = multiprocessing.cpu_count()
    # number of jobs done at the same time
    pool_size = np.max([1, int(cores // 3)])
    pool_size = 4
    # Initialize pool
    pool = []
    curr_obj = 0

    '''Test    
    object_name = file_list_all[curr_obj][len(home_dir) + 35:]
    worker(curr_obj, object_name, 1, 1, flg_vis=True, flg_test=True)
    '''
    #########################
    assert (pool_size <= len(job_list))

    for _ in range(pool_size):
        job_i = job_list.pop(0)
        # object_name = file_list_all[curr_obj][len(home_dir) + 35:]
        object_name = file_list_all[curr_obj][len(file_dir):]

        pool.append(multiprocessing.Process(target=worker, args=(curr_obj,
                                                                 object_name,
                                                                 target_num_grasps, 5)))
        curr_obj += 1
    [p.start() for p in pool]

    while len(job_list) > 0:
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                job_i = job_list.pop(0)
                # object_name = file_list_all[curr_obj][len(home_dir) + 35:]
                object_name = file_list_all[curr_obj][len(file_dir):]

                p = multiprocessing.Process(target=worker, args=(curr_obj, object_name, target_num_grasps,
                                                                 5))
                curr_obj += 1
                p.start()
                pool.append(p)
                break
    print('All job done.')


if __name__ == '__main__':
    main()
