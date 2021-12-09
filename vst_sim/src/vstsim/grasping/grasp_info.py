#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hui Zhang
# E-mail     : hui.zhang@kuleuven.be
# Description:
# Date       : 07/10/2020 16:37
# File Name  : grasp_info.py
import numpy as np
"""
    Define a class to save the basic information of grasp
"""
class GraspInfo(object):
    def __init__(self, name_grasp='001',
                 scale_obj=1.0, g_rad=0.025,
                 pos=np.array([.0, .0, .0]),
                 dir=np.array([.0, .0, .0]), t1=np.array([.0, .0, .0]), t2=np.array([.0, .0, .0]),
                 pts_project=np.zeros([3, 1, 1]),
                 map_project=np.zeros([3, 1, 1]),
                 normals_project=np.zeros([3, 1, 1]),
                 grasp_matrix=np.zeros([4, 1]),
                 quality=0.0,
                 rot_x=0.0, rot_y=0.0):
        """
        :param name_grasp:            str:            name of grasp
        :param scale_obj:           float:          scale of object
        :param g_rad:               float:          radius of gripper
        :param pos:                 3*1 array:
        :param dir:                 3*1 array:      Z-axis of GCS (Gripper Coordinate system)
        :param t1:                  3*1 array:      X-axis
        :param t2:                  3*1 array:      Y-axis
        :param quality:             float:          grasp quality
        :param rot_x:               float:          rotation angle of grasp along with x-axis of GCS (metric: degree)
        :param rot_y:               float:                                             y-axis
        """
        self.__name_grasp = name_grasp
        self.scale_obj = 1.0 * scale_obj
        self.g_rad = 1.0 * g_rad
        self.pos_grasp = 1.0 * pos
        self.dir_grasp = 1.0 * dir
        self.t1_grasp = 1.0 * t1
        self.t2_grasp = 1.0 * t2
        self.quality_grasp = 1.0 * quality
        self.rot_x = 1.0 * rot_x
        self.rot_y = 1.0 * rot_y
        self.pts_project = 1.0 * pts_project
        self.map_project = 1.0 * map_project

        self.normals_project = 1.0 * normals_project
        self.grasp_matrix = 1.0 * grasp_matrix

    def get_name_grasp(self):
        return self.__name_grasp

    def write_quality_grasp(self, new_quality):
        self.quality_grasp = 1.0 * new_quality

    def write_grasp_matrix(self, new_grasp_matrix):
        self.grasp_matrix = 1.0 * new_grasp_matrix

    def wirte_radius_gripper(self, new_rad_gripper):
        self.g_rad = 1.0 * new_rad_gripper

    def write_normals_project(self, new_normals):
        ind_z = np.argwhere(new_normals[2, :, :] > 0)
        new_normals[:, ind_z[:, 0], ind_z[:, 1]] = -1.0 * new_normals[:, ind_z[:, 0], ind_z[:, 1]]
        self.normals_project = 1.0 * new_normals


class GraspInfo_TongueGrasp(object):
    def __init__(self, name_grasp='001',
                 scale_obj=1.0, g_rad_min=0.01, g_rad_max=0.0175,
                 geo_center_obj=np.array([.0, .0, .0]),
                 coor=np.array([.0, .0, .0]),
                 dir=np.array([.0, .0, .0]), t1=np.array([.0, .0, .0]), t2=np.array([.0, .0, .0]),
                 pts_project=np.zeros([3, 1, 1]),
                 map_project=np.zeros([3, 1, 1]),
                 normals_project=np.zeros([3, 1, 1]),
                 grasp_matrix=np.zeros([4, 1]),
                 q1=0.0, q2=0.0, q12=0.0, q123=0.0,
                 q_ft_wrench=0.0,
                 P_air=1.0, P_tongue=0.1):
        """
        :param name_grasp:            str:            name of grasp
        :param scale_obj:           float:          scale of object
        :param g_rad_min:           float:          min radius of gripper
        :param g_rad_max:           float:          max radius of gripper
        :param geo_center_obj       3*1 array:      geometric center of object
        :param coor:                3*1 array:
        :param dir:                 3*1 array:      Z-axis of GCS (Gripper Coordinate system)
        :param t1:                  3*1 array:      X-axis
        :param t2:                  3*1 array:      Y-axis
        :param q1:                  float:          grasp quality -> atmosphere
        :param q2:                  float:          grasp quality -> tongue pressure -> friction
        :param q12:                 float:          grasp quality -> atmosphere + friction
        :param q123:                float:          grasp quality -> atmosphere + friction + captured by tongue
        :param q_ft_wrench          float:          grasp quality evaluated based on force-torque wrench and QP
        :param P_air                float:          air pressure in the atmosphere
        :param P_tongue             float:          pressure in the tongue
        """
        self.__name_grasp = name_grasp
        self.__geo_center_obj = 1.0 * geo_center_obj
        self.scale_obj = 1.0 * scale_obj
        self.g_rad_min = 1.0 * g_rad_min
        self.g_rad_max = 1.0 * g_rad_max
        self.coor_grasp = 1.0 * coor
        self.dir_grasp = 1.0 * dir
        self.t1_grasp = 1.0 * t1
        self.t2_grasp = 1.0 * t2
        self.q_1 = 1.0 * q1
        self.q_2 = 1.0 * q2
        self.q_12 = 1.0 * q12
        self.q_123 = 1.0 * q123
        self.q_ft_wrench = 1.0 * q_ft_wrench
        self.P_air = 1.0 * P_air
        self.P_tongue = 1.0 * P_tongue
        self.pts_project = 1.0 * pts_project
        self.map_project = 1.0 * map_project

        self.normals_project = 1.0 * normals_project
        self.grasp_matrix = 1.0 * grasp_matrix

    def get_name_grasp(self):
        return self.__name_grasp

    def get_geo_center_obj(self):
        return self.__geo_center_obj

    def write_quality_grasp(self, q1=None, q2=None, q12=None, q123=None):
        if q1 is not None:
            self.q_1 = 1.0 * q1
        if q2 is not None:
            self.q_2 = 1.0 * q2
        if q12 is not None:
            self.q_12 = 1.0 * q12
        if q123 is not None:
            self.q_123 = 1.0 * q123

    def write_grasp_matrix(self, new_grasp_matrix):
        self.grasp_matrix = 1.0 * new_grasp_matrix

    def wirte_radius_gripper(self, new_rad_gripper):
        self.g_rad = 1.0 * new_rad_gripper

    def write_normals_project(self, new_normals):
        ind_z = np.argwhere(new_normals[2, :, :] > 0)
        new_normals[:, ind_z[:, 0], ind_z[:, 1]] = -1.0 * new_normals[:, ind_z[:, 0], ind_z[:, 1]]
        self.normals_project = 1.0 * new_normals


