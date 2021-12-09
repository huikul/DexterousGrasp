#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hui Zhang
# E-mail     : hui.zhang@kuleuven.be
# Description:
# Date       : 14/10/2020 11:39 AM
# File Name  : pre_process_sdf_obj_normals.py
import os
import multiprocessing
import subprocess
from vstsim.grasping import GraspableObject3D
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
import numpy as np
import logging
try:
    import pcl
except ImportError:
    logging.warning('Failed to import pcl')
"""
This file convert obj file to sdf file automatically and multiprocessingly, 
then estimate the surface normal for each vertex of the object by SDF
All the cores of a computer can do the job parallel.
"""

def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # current path
        if root.count('/') == file_dir_.count('/')+1:
            file_list.append(root)
        # print(dirs)  # all the directories in current path
        # print(files)  # all the files in current path
    file_list.sort()
    return file_list


def generate_sdf(path_to_sdfgen, obj_filename, dim, padding):
    """ Converts mesh to an sdf object """

    # create the SDF using binary tools
    sdfgen_cmd = '%s \"%s\" %d %d' % (path_to_sdfgen, obj_filename, dim, padding)
    os.system(sdfgen_cmd)
    # print('SDF Command: %s' % sdfgen_cmd)
    return


def do_job_convert_obj_to_sdf(x):
    # file_list_all = get_file_name(file_dir)
    generate_sdf(path_sdfgen, str(file_list_all[x])+"/google_512k/nontextured.obj", 100, 5)  # for google scanner
    # generate_sdf(path_sdfgen, str(file_list_all[x])+"/google_512k/nontextured.obj", 100, 5)  # for google scanner
    print("Done job number", x)


def generate_obj_from_ply(file_name_):
    base = file_name_.split(".")[0]
    p = subprocess.Popen(["pcl_ply2obj", base + ".ply", base + ".obj"])
    p.wait()


def do_job_estimate_obj_normals_sdf(obj_num):
    if os.path.exists(str(file_list_all[obj_num]) + "/google_512k/nontextured.obj"):
        of = ObjFile(str(file_list_all[obj_num]) + "/google_512k/nontextured.obj")
        sf = SdfFile(str(file_list_all[obj_num]) + "/google_512k/nontextured.sdf")
    else:
        print("can't find any obj or sdf file!")
        raise NameError("can't find any obj or sdf file!")
    print("Estimating the surface normals for the object: ", file_list_all[obj_num])
    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)

    # pts_grid = obj.sdf.transform_pt_obj_to_grid(obj.mesh.vertices.T).T

    normals_obj = np.zeros([obj.mesh.vertices.shape[0], 3])
    for i in range(0, obj.mesh.vertices.shape[0]):
        pt_grid = obj.sdf.transform_pt_obj_to_grid(obj.mesh.vertices[i, 0:3])
        normals_obj[i, 0:3] = obj.sdf.estimate_normals(pt_grid)

    np.save(str(file_list_all[obj_num]) + "/google_512k/surface_normals_sdf.npy", normals_obj)
    print("Estimation is completed: ", file_list_all[obj_num])

def do_job_estimate_obj_normals_pcl(obj_num):
    """
    CAUTION: The surface normal is always outwards to the object surface
    :param obj_num:
    :return:
    """
    if os.path.exists(str(file_list_all[obj_num]) + "/google_512k/nontextured.obj"):
        of = ObjFile(str(file_list_all[obj_num]) + "/google_512k/nontextured.obj")
        sf = SdfFile(str(file_list_all[obj_num]) + "/google_512k/nontextured.sdf")
    else:
        print("can't find any obj or sdf file!")
        raise NameError("can't find any obj or sdf file!")
    print("Estimating the surface normals for the object: ", file_list_all[obj_num])
    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)

    pc_obj = pcl.PointCloud(obj.mesh.vertices.astype(np.float32))
    norm = pc_obj.make_NormalEstimation()
    norm.set_KSearch(30)
    # norm.set_RadiusSearch(0.003)
    normals = norm.compute()
    surface_normal = normals.to_array()
    # np.random.shuffle(surface_normal)

    #
    for cnt_pt in range(0, obj.mesh.vertices.shape[0]):

        tmp_pt = obj.mesh.vertices[cnt_pt, :] + surface_normal[cnt_pt, 0:3] * 0.005
        tmp_pt = obj.sdf.transform_pt_obj_to_grid(tmp_pt)
        dist_1 = obj.sdf[tmp_pt]
        # dist_1 = self.graspable.sdf._signed_distance(tmp_pt)

        tmp_pt = obj.mesh.vertices[cnt_pt, :] - surface_normal[cnt_pt, 0:3] * 0.005
        tmp_pt = obj.sdf.transform_pt_obj_to_grid(tmp_pt)
        dist_2 = obj.sdf[tmp_pt]
        # dist_2 = self.graspable.sdf._signed_distance(tmp_pt)

        if dist_1 < dist_2:
            surface_normal[cnt_pt, 0:3] = -1 * surface_normal[cnt_pt, 0:3]

    surface_normal = surface_normal[:, 0:3]
    normals_obj = surface_normal.astype(np.float64)

    np.save(str(file_list_all[obj_num]) + "/google_512k/surface_normals_pcl.npy", normals_obj)
    print("Estimation is completed: ", file_list_all[obj_num])



if __name__ == '__main__':
    # home_dir = os.environ['HOME']
    work_dir = os.environ['PWD'][:-13]
    file_dir = work_dir + "/3D_meshes"  # for google ycb

    # path_sdfgen = home_dir + "/Dexterous_grasp_01/SDFGen/bin/SDFGen"
    path_sdfgen = work_dir + "/SDFGen/bin/SDFGen"

    file_list_all = get_file_name(file_dir)
    object_numbers = file_list_all.__len__()

    ''''''
    # generate obj from ply file
    for i in file_list_all:
        generate_obj_from_ply(i+"/google_512k/nontextured.ply")
        # generate_obj_from_ply(i+"/google_512k/nontextured.ply")
        print("finish", i)
    # The operation for the multi core
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    pool.map(do_job_convert_obj_to_sdf, range(object_numbers))

    '''    
    cores = multiprocessing.cpu_count()
    cores = int(cores // 2)
    # do_job_estimate_obj_normals_pcl(0)
    pool = multiprocessing.Pool(processes=cores)
    pool.map(do_job_estimate_obj_normals_sdf, range(object_numbers))
    # pool.map(do_job_estimate_obj_normals_pcl, range(object_numbers))
    '''
