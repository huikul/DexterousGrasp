# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import copy
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# import os, IPython, sys
import math
import random
import time
import scipy.stats as stats
from vstsim.visualization.GL_visualizer3d import GL_Visualizer

try:
    import pcl
except ImportError:
    logging.warning('Failed to import pcl!')
import vstsim
import itertools as it
import multiprocessing
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import datetime
import pickle
from vstsim.grasping import Grasp, Contact3D, ParallelJawPtGrasp3D, PointGraspMetrics3D, \
    VacuumPoint, GraspQuality_Vacuum, DexterousVacuumPoint, DexterousQuality_Vacuum, \
    ChameleonTongueContact, ChameleonTongue_Quality
from vstsim.grasping import GraspInfo, GraspInfo_TongueGrasp
from vstsim.grasping import math_robot

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from autolab_core import RigidTransform
import scipy

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

USE_OPENRAVE = True
try:
    import openravepy as rave
except ImportError:
    logger.warning('Failed to import OpenRAVE')
    USE_OPENRAVE = False

try:
    import rospy
    import moveit_commander
except ImportError:
    logger.warning("Failed to import rospy, you can't grasp now.")

try:
    from mayavi import mlab
except ImportError:
    mlab = []
    logger.warning('Do not have mayavi installed, please set the vis to False')

"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Classes for sampling grasps.
Author: Jeff Mahler
"""
"""
Modified by: Hui Zhang
Email      : hui.zhang@kuleuven.be
Date       : 23/02/2020   09:53
"""

color_ls = np.ones((14, 3), dtype=np.int32)
color_ls[0, :] = np.array([255, 0, 0])  # Red
color_ls[1, :] = np.array([60, 180, 75])  # Green
color_ls[2, :] = np.array([255, 225, 25])  # Yellow
color_ls[3, :] = np.array([0, 130, 200])  # Blue
color_ls[4, :] = np.array([245, 130, 48])  # Orange
color_ls[5, :] = np.array([145, 30, 180])  # Purple
color_ls[6, :] = np.array([70, 240, 240])  # Cyan
color_ls[7, :] = np.array([240, 50, 230])  # Magenta
color_ls[8, :] = np.array([210, 245, 60])  # Lime
color_ls[9, :] = np.array([250, 190, 190])  # Pink
color_ls[10, :] = np.array([0, 128, 128])  # Teal
color_ls[11, :] = np.array([128, 0, 0])  # Maroon
color_ls[12, :] = np.array([128, 128, 0])  # Olive
color_ls[13, :] = np.array([0, 0, 128])  # Navy


# class GraspSampler(metaclass=ABCMeta):
class GraspSampler:
    """ Base class for various methods to sample a number of grasps on an object.
    Should not be instantiated directly.

    Attributes
    ----------
    gripper : :obj:`RobotGripper`
        the gripper to compute grasps for
    config : :obj:`YamlConfig`
        configuration for the grasp sampler
    """
    __metaclass__ = ABCMeta

    def __init__(self, gripper, config):
        self.gripper = gripper
        self._configure(config)

    def _configure(self, config):
        """ Configures the grasp generator."""
        #########################################
        if 'sampling_friction_coef' in list(config.keys()):
            self.friction_coef = config['sampling_friction_coef']
        else:
            self.friction_coef = 2.0
        #########################################
        if 'num_cone_faces' in list(config.keys()):
            self.num_cone_faces = config['num_cone_faces']
        else:
            self.num_cone_faces = 8
        #########################################
        if 'grasp_samples_per_surface_point' in list(config.keys()):
            self.num_samples = config['grasp_samples_per_surface_point']
        else:
            self.num_samples = 1
        #########################################
        if 'target_num_grasps' in list(config.keys()):
            self.target_num_grasps = config['target_num_grasps']
        else:
            self.target_num_grasps = 1
        #########################################
        if self.target_num_grasps is None:
            self.target_num_grasps = 1
        #########################################
        if 'min_contact_dist' in list(config.keys()):
            self.min_contact_dist = config['min_contact_dist']
        else:
            self.min_contact_dist = 0.0
        #########################################
        if 'num_grasp_rots' in list(config.keys()):
            self.num_grasp_rots = config['num_grasp_rots']
        else:
            self.num_grasp_rots = 0.0
        ###########################################
        ###########################################
        # parameters for virtual camera
        #########################################
        if 'back_up_dis' in list(config.keys()):
            self.back_up_dis = config['back_up_dis']
        else:
            self.back_up_dis = 1
        #########################################
        if 'max_projection_dis' in list(config.keys()):
            self.max_projection_dis = config['max_projection_dis']
        else:
            self.max_projection_dis = 1
        #########################################
        if 'num_projection_steps' in list(config.keys()):
            self.num_projection_steps = config['num_projection_steps']
        else:
            self.num_projection_steps = 20
        #########################################
        if 'resolution_pc' in list(config.keys()):
            self.resolution_pc = config['resolution_pc']
        else:
            self.resolution_pc = 24
        #########################################
        if 'angle_range_max' in list(config.keys()):
            self.angle_range = config['angle_range_max']
        else:
            self.angle_range = 30.0
        #########################################
        if 'angle_range_max' in list(config.keys()):
            self.angle_range_max = config['angle_range_max']
        else:
            self.angle_range_max = 5.0
        #########################################
        if 'angle_range_min' in list(config.keys()):
            self.angle_range_min = config['angle_range_min']
        else:
            self.angle_range_min = 1.0
        #########################################
        if 'num_angle_steps' in list(config.keys()):
            self.num_angle_steps = config['num_angle_steps']
        else:
            self.num_angle_steps = 5
        #########################################
        if 'scale_obj' in list(config.keys()):
            self.scale_obj = config['scale_obj']
        else:
            self.scale_obj = 1.0
        #########################################
        if 'dim_grasp_matrix' in list(config.keys()):
            self.dim_grasp_matrix = config['dim_grasp_matrix']
        else:
            self.dim_grasp_matrix = 100
        #########################################
        if 'max_num_surface_points' in list(config.keys()):
            self.max_num_surface_points_ = config['max_num_surface_points']
        else:
            self.max_num_surface_points_ = 100
        #########################################
        if 'grasp_dist_thresh' in list(config.keys()):
            self.grasp_dist_thresh_ = config['grasp_dist_thresh']
        else:
            self.grasp_dist_thresh_ = 0

    @abstractmethod
    def sample_grasps(self, graspable, num_grasps_generate, vis, **kwargs):
        """
        Create a list of candidate grasps for a given object.
        Must be implemented for all grasp sampler classes.

        Parameters
        ---------
        graspable : :obj:`GraspableObject3D`
            object to sample grasps on
        num_grasps_generate : int
        vis : bool
        """
        grasp = []
        return grasp
        # pass

    def generate_grasps_stable_poses(self, graspable, stable_poses, target_num_grasps=None, grasp_gen_mult=5,
                                     max_iter=3, sample_approach_angles=False, vis=False, **kwargs):
        """Samples a set of grasps for an object, aligning the approach angles to the object stable poses.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        stable_poses : :obj:`list` of :obj:`meshpy.StablePose`
            list of stable poses for the object with ids read from the database
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles
        vis : bool
        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # sample dense grasps
        unaligned_grasps = self.generate_grasps(graspable, target_num_grasps=target_num_grasps,
                                                grasp_gen_mult=grasp_gen_mult,
                                                max_iter=max_iter, vis=vis)

        # align for each stable pose
        grasps = {}
        print(sample_approach_angles)  # add by Liang
        for stable_pose in stable_poses:
            grasps[stable_pose.id] = []
            for grasp in unaligned_grasps:
                aligned_grasp = grasp.perpendicular_table(grasp)
                grasps[stable_pose.id].append(copy.deepcopy(aligned_grasp))
        return grasps

    def generate_grasps(self, graspable, target_num_grasps=None, grasp_gen_mult=5, max_iter=3,
                        sample_approach_angles=False, vis=False, **kwargs):
        """Samples a set of grasps for an object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles
        vis : bool
            whether show the grasp on picture

        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # get num grasps
        if target_num_grasps is None:
            target_num_grasps = self.target_num_grasps
        num_grasps_remaining = target_num_grasps

        grasps = []
        k = 1
        while num_grasps_remaining > 0 and k <= max_iter:
            # SAMPLING: generate more than we need
            num_grasps_generate = grasp_gen_mult * num_grasps_remaining
            new_grasps = self.sample_grasps(graspable, num_grasps_generate, vis, **kwargs)

            # COVERAGE REJECTION: prune grasps by distance
            pruned_grasps = []
            for grasp in new_grasps:
                min_dist = np.inf
                for cur_grasp in grasps:
                    dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                    if dist < min_dist:
                        min_dist = dist
                for cur_grasp in pruned_grasps:
                    dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist >= self.grasp_dist_thresh_:
                    pruned_grasps.append(grasp)

            # ANGLE EXPANSION sample grasp rotations around the axis
            candidate_grasps = []
            if sample_approach_angles:
                for grasp in pruned_grasps:
                    # construct a set of rotated grasps
                    for i in range(self.num_grasp_rots):
                        rotated_grasp = copy.copy(grasp)
                        delta_theta = 0  # add by Hongzhuo Liang
                        print("This function can not use yes, as delta_theta is not set. --Hongzhuo Liang")
                        rotated_grasp.set_approach_angle(i * delta_theta)
                        candidate_grasps.append(rotated_grasp)
            else:
                candidate_grasps = pruned_grasps

            # add to the current grasp set
            grasps += candidate_grasps
            logger.info('%d/%d grasps found after iteration %d.',
                        len(grasps), target_num_grasps, k)

            grasp_gen_mult *= 2
            num_grasps_remaining = target_num_grasps - len(grasps)
            k += 1

        # shuffle computed grasps
        random.shuffle(grasps)
        if len(grasps) > target_num_grasps:
            logger.info('Truncating %d grasps to %d.',
                        len(grasps), target_num_grasps)
            grasps = grasps[:target_num_grasps]
        logger.info('Found %d grasps.', len(grasps))
        return grasps

    def show_points(self, point, color='lb', scale_factor=.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
        if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
            point = point.reshape(3, )
            mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
        else:  # vis for multiple points
            mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)

    def show_line(self, un1, un2, color='g', scale_factor=0.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        else:
            color_f = (1, 1, 1)
        mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)

    def show_grasp_norm_oneside(self, grasp_bottom_center,
                                grasp_normal, grasp_axis, minor_pc, scale_factor=0.001):

        # un1 = grasp_bottom_center + 0.5 * grasp_axis * self.gripper.max_width
        un2 = grasp_bottom_center
        # un3 = grasp_bottom_center + 0.5 * minor_pc * self.gripper.max_width
        # un4 = grasp_bottom_center
        # un5 = grasp_bottom_center + 0.5 * grasp_normal * self.gripper.max_width
        # un6 = grasp_bottom_center
        self.show_points(grasp_bottom_center, color='g', scale_factor=scale_factor * 4)
        # self.show_points(un1, scale_factor=scale_factor * 4)
        # self.show_points(un3, scale_factor=scale_factor * 4)
        # self.show_points(un5, scale_factor=scale_factor * 4)
        # self.show_line(un1, un2, color='g', scale_factor=scale_factor)  # binormal/ major pc
        # self.show_line(un3, un4, color='b', scale_factor=scale_factor)  # minor pc
        # self.show_line(un5, un6, color='r', scale_factor=scale_factor)  # approach normal
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_axis[0], grasp_axis[1], grasp_axis[2],
                      scale_factor=.03, line_width=0.25, color=(0, 1, 0), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], minor_pc[0], minor_pc[1], minor_pc[2],
                      scale_factor=.03, line_width=0.1, color=(0, 0, 1), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_normal[0], grasp_normal[1], grasp_normal[2],
                      scale_factor=.03, line_width=0.05, color=(1, 0, 0), mode='arrow')

    def get_hand_points(self, grasp_bottom_center, approach_normal, binormal):
        hh = self.gripper.hand_height
        fw = self.gripper.finger_width
        hod = self.gripper.hand_outer_diameter
        hd = self.gripper.hand_depth
        open_w = hod - fw * 2
        minor_pc = np.cross(approach_normal, binormal)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)
        p5_p6 = minor_pc * hh * 0.5 + grasp_bottom_center
        p7_p8 = -minor_pc * hh * 0.5 + grasp_bottom_center
        p5 = -binormal * open_w * 0.5 + p5_p6
        p6 = binormal * open_w * 0.5 + p5_p6
        p7 = binormal * open_w * 0.5 + p7_p8
        p8 = -binormal * open_w * 0.5 + p7_p8
        p1 = approach_normal * hd + p5
        p2 = approach_normal * hd + p6
        p3 = approach_normal * hd + p7
        p4 = approach_normal * hd + p8

        p9 = -binormal * fw + p1
        p10 = -binormal * fw + p4
        p11 = -binormal * fw + p5
        p12 = -binormal * fw + p8
        p13 = binormal * fw + p2
        p14 = binormal * fw + p3
        p15 = binormal * fw + p6
        p16 = binormal * fw + p7

        p17 = -approach_normal * hh + p11
        p18 = -approach_normal * hh + p15
        p19 = -approach_normal * hh + p16
        p20 = -approach_normal * hh + p12
        p = np.vstack([np.array([0, 0, 0]), p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                       p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
        return p

    def show_grasp_3d(self, hand_points, color=(0.003, 0.50196, 0.50196)):
        # for i in range(1, 21):
        #     self.show_points(p[i])
        if color == 'd':
            color = (0.003, 0.50196, 0.50196)
        triangles = [(9, 1, 4), (4, 9, 10), (4, 10, 8), (8, 10, 12), (1, 4, 8), (1, 5, 8),
                     (1, 5, 9), (5, 9, 11), (9, 10, 20), (9, 20, 17), (20, 17, 19), (17, 19, 18),
                     (14, 19, 18), (14, 18, 13), (3, 2, 13), (3, 13, 14), (3, 6, 7), (3, 6, 2),
                     (3, 14, 7), (14, 7, 16), (2, 13, 15), (2, 15, 6), (12, 20, 19), (12, 19, 16),
                     (15, 11, 17), (15, 17, 18), (6, 7, 8), (6, 8, 5)]
        mlab.triangular_mesh(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
                             triangles, color=color, opacity=0.5)

    def check_collision_square(self, grasp_bottom_center, approach_normal, binormal,
                               minor_pc, graspable, p, way, vis=False):
        approach_normal = approach_normal.reshape(1, 3)
        approach_normal = approach_normal / np.linalg.norm(approach_normal)
        binormal = binormal.reshape(1, 3)
        binormal = binormal / np.linalg.norm(binormal)
        minor_pc = minor_pc.reshape(1, 3)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)
        matrix = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
        grasp_matrix = matrix.T  # same as cal the inverse
        if isinstance(graspable, vstsim.grasping.graspable_object.GraspableObject3D):
            points = graspable.sdf.surface_points(grid_basis=False)[0]
        else:
            points = graspable
        points = points - grasp_bottom_center.reshape(1, 3)
        # points_g = points @ grasp_matrix
        tmp = np.dot(grasp_matrix, points.T)
        points_g = tmp.T
        if way == "p_open":
            s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
        elif way == "p_left":
            s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
        elif way == "p_right":
            s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
        elif way == "p_bottom":
            s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
        else:
            raise ValueError('No way!')
        a1 = s1[1] < points_g[:, 1]
        a2 = s2[1] > points_g[:, 1]
        a3 = s1[2] > points_g[:, 2]
        a4 = s4[2] < points_g[:, 2]
        a5 = s4[0] > points_g[:, 0]
        a6 = s8[0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

        if vis:
            print("points_in_area", way, len(points_in_area))
            mlab.clf()
            # self.show_one_point(np.array([0, 0, 0]))
            self.show_grasp_3d(p)
            self.show_points(points_g)
            if len(points_in_area) != 0:
                self.show_points(points_g[points_in_area], color='r')
            mlab.show()
        # print("points_in_area", way, len(points_in_area))
        return has_p, points_in_area

    def show_all_grasps(self, all_points, grasps_for_show):

        for grasp_ in grasps_for_show:
            grasp_bottom_center = grasp_[4]  # new feature: ues the modified grasp bottom center
            approach_normal = grasp_[1]
            binormal = grasp_[2]
            hand_points = self.get_hand_points(grasp_bottom_center, approach_normal, binormal)
            self.show_grasp_3d(hand_points)
        # self.show_points(all_points)
        # mlab.show()

    def check_collide(self, grasp_bottom_center, approach_normal, binormal, minor_pc, graspable, hand_points):
        bottom_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                    binormal, minor_pc, graspable, hand_points, "p_bottom")
        if bottom_points[0]:
            return True

        left_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                  binormal, minor_pc, graspable, hand_points, "p_left")
        if left_points[0]:
            return True

        right_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                   binormal, minor_pc, graspable, hand_points, "p_right")
        if right_points[0]:
            return True

        return False

    def cal_surface_property(self, graspable, selected_surface, r_ball,
                             point_amount, max_trial, vis=False):
        tmp_count = 0
        M = np.zeros((3, 3))
        trial = 0
        old_normal = graspable.sdf.surface_normal(graspable.sdf.transform_pt_obj_to_grid(selected_surface))
        if old_normal is None:
            logger.warning("The selected point has no norm according to meshpy!")
            return None
        while tmp_count < point_amount and trial < max_trial:
            trial += 1
            neighbor = selected_surface + 2 * (np.random.rand(3) - 0.5) * r_ball
            normal = graspable.sdf.surface_normal(graspable.sdf.transform_pt_obj_to_grid(neighbor))
            if normal is None:
                continue
            normal = normal.reshape(-1, 1)
            if np.linalg.norm(normal) != 0:
                normal /= np.linalg.norm(normal)
            if vis:
                # show the r-ball performance
                neighbor = neighbor.reshape(-1, 1)
                # self.show_line(neighbor, normal * 0.05 + neighbor)
            M += np.matmul(normal, normal.T)
            tmp_count = tmp_count + 1

        if trial == max_trial:
            logger.warning("rball computation failed over %d", max_trial)
            return None

        eigval, eigvec = np.linalg.eig(M)  # compared computed normal
        minor_pc = eigvec[np.argmin(eigval)]  # minor principal curvature
        minor_pc /= np.linalg.norm(minor_pc)
        new_normal = eigvec[np.argmax(eigval)]  # estimated surface normal
        new_normal /= np.linalg.norm(new_normal)
        major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
        if np.linalg.norm(major_pc) != 0:
            major_pc = major_pc / np.linalg.norm(major_pc)
        return old_normal, new_normal, major_pc, minor_pc


class UniformGraspSampler(GraspSampler):
    """ Sample grasps by sampling pairs of points on the object surface uniformly at random.
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=1000, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        vis :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        num_surface = surface_points.shape[0]
        i = 0
        grasps = []

        # get all grasps
        while len(grasps) < num_grasps and i < max_num_samples:
            # get candidate contacts
            indices = np.random.choice(num_surface, size=2, replace=False)
            c0 = surface_points[indices[0], :]
            c1 = surface_points[indices[1], :]

            gripper_distance = np.linalg.norm(c1 - c0)
            if self.gripper.min_width < gripper_distance < self.gripper.max_width:
                # compute centers and axes
                grasp_center = ParallelJawPtGrasp3D.center_from_endpoints(c0, c1)
                grasp_axis = ParallelJawPtGrasp3D.axis_from_endpoints(c0, c1)
                g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center,
                                                                                        grasp_axis,
                                                                                        self.gripper.max_width))
                # keep grasps if the fingers close
                if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:
                    angle_candidates = np.arange(-90, 120, 30)
                    np.random.shuffle(angle_candidates)
                    for grasp_angle in angle_candidates:
                        g.approach_angle_ = grasp_angle
                        # get true contacts (previous is subject to variation)
                        success, contacts = g.close_fingers(graspable, vis=vis)
                        if not success:
                            continue
                        break
                    else:
                        continue
                else:
                    success, contacts = g.close_fingers(graspable, vis=vis)

                if success:
                    grasps.append(g)
            i += 1

        return grasps


class GaussianGraspSampler(GraspSampler):
    """ Sample grasps by sampling a center from a gaussian with mean at the object center of mass
    and grasp axis by sampling the spherical angles uniformly at random.
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, sigma_scale=2.5, **kwargs):
        """
        Returns a list of candidate grasps for graspable object by Gaussian with
        variance specified by principal dimensions.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        sigma_scale : float
            the number of sigmas on the tails of the Gaussian for each dimension
        vis : bool
            visualization

        Returns
        -------
        :obj:`list` of obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        # get object principal axes
        center_of_mass = graspable.mesh.center_of_mass
        principal_dims = graspable.mesh.principal_dims()
        sigma_dims = principal_dims / (2 * sigma_scale)

        # sample centers
        grasp_centers = stats.multivariate_normal.rvs(
            mean=center_of_mass, cov=sigma_dims ** 2, size=num_grasps)

        # samples angles uniformly from sphere
        u = stats.uniform.rvs(size=num_grasps)
        v = stats.uniform.rvs(size=num_grasps)
        thetas = 2 * np.pi * u
        phis = np.arccos(2 * v - 1.0)
        grasp_dirs = np.array([np.sin(phis) * np.cos(thetas), np.sin(phis) * np.sin(thetas), np.cos(phis)])
        grasp_dirs = grasp_dirs.T

        # convert to grasp objects
        grasps = []
        for i in range(num_grasps):
            grasp = ParallelJawPtGrasp3D(
                ParallelJawPtGrasp3D.configuration_from_params(grasp_centers[i, :], grasp_dirs[i, :],
                                                               self.gripper.max_width))

            if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:
                angle_candidates = np.arange(-90, 120, 30)
                np.random.shuffle(angle_candidates)
                for grasp_angle in angle_candidates:
                    grasp.approach_angle_ = grasp_angle
                    # get true contacts (previous is subject to variation)
                    success, contacts = grasp.close_fingers(graspable, vis=vis)
                    if not success:
                        continue
                    break
                else:
                    continue
            else:
                success, contacts = grasp.close_fingers(graspable, vis=vis)

            # add grasp if it has valid contacts
            if success and np.linalg.norm(contacts[0].point - contacts[1].point) > self.min_contact_dist:
                grasps.append(grasp)

        # visualize
        if vis:
            for grasp in grasps:
                plt.clf()
                plt.gcf()
                plt.ion()
                grasp.close_fingers(graspable, vis=vis)
                plt.show(block=False)
                time.sleep(0.5)

            grasp_centers_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_centers.T)
            grasp_centers_grid = grasp_centers_grid.T
            com_grid = graspable.sdf.transform_pt_obj_to_grid(center_of_mass)

            plt.clf()
            ax = plt.gca(projection='3d')
            # graspable.sdf.scatter()
            ax.scatter(grasp_centers_grid[:, 0], grasp_centers_grid[:, 1], grasp_centers_grid[:, 2], s=60, c='m')
            ax.scatter(com_grid[0], com_grid[1], com_grid[2], s=120, c='y')
            ax.set_xlim3d(0, graspable.sdf.dims_[0])
            ax.set_ylim3d(0, graspable.sdf.dims_[1])
            ax.set_zlim3d(0, graspable.sdf.dims_[2])
            plt.show()

        return grasps


class AntipodalGraspSampler(GraspSampler):
    """ Samples antipodal pairs using rejection sampling.
    The proposal sampling ditribution is to choose a random point on
    the object surface, then sample random directions within the friction cone,
    then form a grasp axis along the direction,
    close the fingers, and keep the grasp if the other contact point is also in the friction cone.
    """

    def sample_from_cone(self, n, tx, ty, num_samples=1):
        """ Samples directoins from within the friction cone using uniform sampling.

        Parameters
        ----------
        n : 3x1 normalized :obj:`numpy.ndarray`
            surface normal
        tx : 3x1 normalized :obj:`numpy.ndarray`
            tangent x vector
        ty : 3x1 normalized :obj:`numpy.ndarray`
            tangent y vector
        num_samples : int
            number of directions to sample

        Returns
        -------
        v_samples : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            sampled directions in the friction cone
       """
        v_samples = []
        for i in range(num_samples):
            theta = 2 * np.pi * np.random.rand()
            r = self.friction_coef * np.random.rand()
            v = n + r * np.cos(theta) * tx + r * np.sin(theta) * ty
            v = -v / np.linalg.norm(v)
            v_samples.append(v)
        return v_samples

    def within_cone(self, cone, n, v):
        """
        Checks whether or not a direction is in the friction cone.
        This is equivalent to whether a grasp will slip using a point contact model.

        Parameters
        ----------
        cone : 3xN :obj:`numpy.ndarray`
            supporting vectors of the friction cone
        n : 3x1 :obj:`numpy.ndarray`
            outward pointing surface normal vector at c1
        v : 3x1 :obj:`numpy.ndarray`
            direction vector

        Returns
        -------
        in_cone : bool
            True if alpha is within the cone
        alpha : float
            the angle between the normal and v
        """
        if (v.dot(cone) < 0).any():  # v should point in same direction as cone
            v = -v  # don't worry about sign, we don't know it anyway...
        f = -n / np.linalg.norm(n)
        alpha = np.arccos(f.T.dot(v) / np.linalg.norm(v))
        return alpha <= np.arctan(self.friction_coef), alpha

    def perturb_point(self, x, scale):
        """ Uniform random perturbations to a point """
        x_samp = x + (scale / 2.0) * (np.random.rand(3) - 0.5)
        return x_samp

    def sample_grasps(self, graspable, num_grasps, vis=False, **kwargs):
        """Returns a list of candidate grasps for graspable object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            number of grasps to sample
        vis : bool
            whether or not to visualize progress, for debugging

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            the sampled grasps
        """
        # get surface points
        grasps = []
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        np.random.shuffle(surface_points)
        shuffled_surface_points = surface_points[:min(self.max_num_surface_points_, len(surface_points))]
        logger.info('Num surface: %d' % (len(surface_points)))

        for k, x_surf in enumerate(shuffled_surface_points):
            # print("k:", k, "len(grasps):", len(grasps))
            start_time = time.clock()

            # perturb grasp for num samples
            for i in range(self.num_samples):
                # perturb contact (TODO: sample in tangent plane to surface)
                x1 = self.perturb_point(x_surf, graspable.sdf.resolution)

                # compute friction cone faces
                c1 = Contact3D(graspable, x1, in_direction=None)
                _, tx1, ty1 = c1.tangents()
                cone_succeeded, cone1, n1 = c1.friction_cone(self.num_cone_faces, self.friction_coef)
                if not cone_succeeded:
                    continue
                cone_time = time.clock()

                # sample grasp axes from friction cone
                v_samples = self.sample_from_cone(n1, tx1, ty1, num_samples=1)
                sample_time = time.clock()

                for v in v_samples:
                    if vis:
                        x1_grid = graspable.sdf.transform_pt_obj_to_grid(x1)
                        cone1_grid = graspable.sdf.transform_pt_obj_to_grid(cone1, direction=True)
                        plt.clf()
                        plt.gcf()
                        plt.ion()
                        ax = plt.gca(projection=Axes3D)
                        for j in range(cone1.shape[1]):
                            ax.scatter(x1_grid[0] - cone1_grid[0], x1_grid[1] - cone1_grid[1],
                                       x1_grid[2] - cone1_grid[2], s=50, c='m')

                    # random axis flips since we don't have guarantees on surface normal directoins
                    if random.random() > 0.5:
                        v = -v

                    # start searching for contacts
                    grasp, c1, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(
                        graspable, x1, v, self.gripper.max_width,
                        min_grasp_width_world=self.gripper.min_width, vis=vis)
                    if grasp is None or c2 is None:
                        continue

                    if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:
                        angle_candidates = np.arange(-90, 120, 30)
                        np.random.shuffle(angle_candidates)
                        for grasp_angle in angle_candidates:
                            grasp.approach_angle_ = grasp_angle
                            # get true contacts (previous is subject to variation)
                            success, c = grasp.close_fingers(graspable, vis=vis)
                            if not success:
                                continue
                            break
                        else:
                            continue
                    else:
                        success, c = grasp.close_fingers(graspable, vis=vis)
                        if not success:
                            continue
                    c1 = c[0]
                    c2 = c[1]

                    # make sure grasp is wide enough
                    x2 = c2.point
                    if np.linalg.norm(x1 - x2) < self.min_contact_dist:
                        continue

                    v_true = grasp.axis
                    # compute friction cone for contact 2
                    cone_succeeded, cone2, n2 = c2.friction_cone(self.num_cone_faces, self.friction_coef)
                    if not cone_succeeded:
                        continue

                    if vis:
                        plt.figure()
                        ax = plt.gca(projection='3d')
                        c1_proxy = c1.plot_friction_cone(color='m')
                        c2_proxy = c2.plot_friction_cone(color='y')
                        ax.view_init(elev=5.0, azim=0)
                        plt.show(block=False)
                        time.sleep(0.5)
                        plt.close()  # lol

                    # check friction cone
                    if PointGraspMetrics3D.force_closure(c1, c2, self.friction_coef):
                        grasps.append(grasp)

        # randomly sample max num grasps from total list
        random.shuffle(grasps)
        return grasps


class GpgGraspSampler(GraspSampler):
    """
    Sample grasps by GPG.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=30, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        vis :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 10,  # unit degree
            'range_dtheta': 90,
            'debug_vis': False,
            'r_ball': self.gripper.hand_height,
            'approach_step': 0.005,
            'max_trail_for_r_ball': 3000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        all_points = surface_points
        # construct pynt point cloud and voxel grid
        p_cloud = pcl.PointCloud(surface_points.astype(np.float32))
        voxel = p_cloud.make_voxel_grid_filter()
        voxel.set_leaf_size(*([graspable.sdf.resolution * params['voxel_grid_ratio']] * 3))
        surface_points = voxel.filter().to_array()

        num_surface = surface_points.shape[0]
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # get candidate contacts
            ind = np.random.choice(num_surface, size=1, replace=False)
            selected_surface = surface_points[ind, :].reshape(3)

            # cal major principal curvature
            # r_ball = max(self.gripper.hand_depth, self.gripper.hand_outer_diameter)
            r_ball = params['r_ball']  # FIXME: for some relative small obj, we need to use pre-defined radius
            point_amount = params['num_rball_points']
            max_trial = params['max_trail_for_r_ball']
            # TODO: we can not directly sample from point clouds so we use a relatively small radius.
            ret = self.cal_surface_property(graspable, selected_surface, r_ball,
                                            point_amount, max_trial, vis=params['debug_vis'])
            if ret is None:
                continue
            else:
                old_normal, new_normal, major_pc, minor_pc = ret

            # Judge if the new_normal has the same direction with old_normal, here the correct
            # direction in modified meshpy is point outward.
            if np.dot(old_normal, new_normal) < 0:
                new_normal = -new_normal
                minor_pc = -minor_pc

            for normal_dir in [1.]:  # FIXME: here we can now know the direction of norm, outward
                if params['debug_vis']:
                    # example of show grasp frame
                    self.show_grasp_norm_oneside(selected_surface, new_normal * normal_dir, major_pc * normal_dir,
                                                 minor_pc, scale_factor=0.001)
                    self.show_points(selected_surface, color='g', scale_factor=.002)
                    # self.show_points(all_points)

                # some magic number referred from origin paper
                potential_grasp = []
                for dtheta in np.arange(-params['range_dtheta'],
                                        params['range_dtheta'] + 1,
                                        params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                        (params['num_dy'] + 1) * self.gripper.finger_width,
                                        self.gripper.finger_width):
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc * normal_dir)
                        tmp_grasp_normal = np.dot(rotation, new_normal * normal_dir)
                        tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy
                        # go back a bite after rotation dtheta and translation dy!
                        tmp_grasp_bottom_center = self.gripper.init_bite * (
                                -tmp_grasp_normal * normal_dir) + tmp_grasp_bottom_center

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, graspable,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, graspable,
                                                                       hand_points,
                                                                       "p_bottom")
                        if open_points is True and bottom_points is False:

                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points,
                                                                         "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, graspable,
                                                                          hand_points,
                                                                          "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])
                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        potential_grasp.append(dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)])
                approach_dist = self.gripper.hand_depth  # use gripper depth
                num_approaches = int(approach_dist / params['approach_step'])
                for ptg in potential_grasp:
                    for approach_s in range(num_approaches):
                        tmp_grasp_bottom_center = ptg[1] * approach_s * params['approach_step'] + ptg[0]

                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, graspable, hand_points)
                        if is_collide:
                            # if collide, go back one step to get a collision free hand position
                            tmp_grasp_bottom_center += (-tmp_grasp_normal) * params['approach_step']

                            # final check
                            open_points, _ = self.check_collision_square(tmp_grasp_bottom_center,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points, "p_open")
                            is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                            tmp_major_pc, minor_pc, graspable, hand_points)
                            if open_points and not is_collide:
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                                self.show_points(selected_surface, color='r', scale_factor=.005)
                                if params['debug_vis']:
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, graspable, hand_points,
                                                                "p_open", vis=True)
                            break
                logger.info("processed_potential_grasp %d", len(processed_potential_grasp))

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)
            if not sampled_surface_amount % 20:  # params['debug_vis']:
                # self.show_all_grasps(all_points, processed_potential_grasp)
                # self.show_points(all_points)
                # mlab.show()
                return processed_potential_grasp
            #
            # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            #     tmp_grasp_center,
            #     tmp_major_pc,
            #     self.gripper.max_width))
            # grasps.append(g)

        return processed_potential_grasp


class PointGraspSampler(GraspSampler):
    """
    Sample grasps by PointGraspSampler
    TODO: since gpg sampler changed a lot, this class need to totally rewrite
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=1000, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        vis :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 10,  # unit degree
            'range_dtheta': 90,
            'debug_vis': False,
            'approach_step': 0.005,
            'max_trail_for_r_ball': 3000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        all_points = surface_points
        # construct pynt point cloud and voxel grid
        p_cloud = pcl.PointCloud(surface_points.astype(np.float32))
        voxel = p_cloud.make_voxel_grid_filter()
        voxel.set_leaf_size(*([graspable.sdf.resolution * params['voxel_grid_ratio']] * 3))
        surface_points = voxel.filter().to_array()

        num_surface = surface_points.shape[0]
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # get candidate contacts
            # begin of modification 5: Gaussian over height, select more point in the middle
            # we can use the top part of the point clouds to generate more sample points
            min_height = min(surface_points[:, 2])
            max_height = max(surface_points[:, 2])

            selected_height = min_height + np.random.normal(3 * (max_height - min_height) / 4,
                                                            (max_height - min_height) / 6)
            ind_10 = np.argsort(abs(surface_points[:, 2] - selected_height))[:10]
            ind = ind_10[np.random.choice(len(ind_10), 1)]
            # end of modification 5
            # ind = np.random.choice(num_surface, size=1, replace=False)
            selected_surface = surface_points[ind, :].reshape(3)

            # cal major principal curvature
            r_ball = max(self.gripper.hand_depth, self.gripper.hand_outer_diameter)
            point_amount = params['num_rball_points']
            max_trial = params['max_trail_for_r_ball']
            # TODO: we can not directly sample from point clouds so we use a relatively small radius.
            ret = self.cal_surface_property(graspable, selected_surface, r_ball,
                                            point_amount, max_trial, vis=params['debug_vis'])
            if ret is None:
                continue
            else:
                old_normal, new_normal, major_pc, minor_pc = ret

            for normal_dir in [-1., 1.]:  # FIXME: here we do not know the direction of the object normal
                grasp_bottom_center = self.gripper.init_bite * new_normal * -normal_dir + selected_surface
                new_normal = normal_dir * new_normal
                major_pc = normal_dir * major_pc

                if params['debug_vis']:
                    # example of show grasp frame
                    self.show_grasp_norm_oneside(selected_surface, new_normal, major_pc,
                                                 minor_pc, scale_factor=0.001)
                    self.show_points(selected_surface, color='g', scale_factor=.002)

                # some magic number referred from origin paper
                potential_grasp = []
                extra_potential_grasp = []

                for dtheta in np.arange(-params['range_dtheta'],
                                        params['range_dtheta'] + 1,
                                        params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                        (params['num_dy'] + 1) * self.gripper.finger_width,
                                        self.gripper.finger_width):
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc)
                        tmp_grasp_normal = np.dot(rotation, new_normal)
                        tmp_grasp_bottom_center = grasp_bottom_center + tmp_major_pc * dy

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, graspable,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, graspable,
                                                                       hand_points,
                                                                       "p_bottom")
                        if open_points is True and bottom_points is False:

                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points,
                                                                         "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, graspable,
                                                                          hand_points,
                                                                          "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])
                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        # potential_grasp += dy_potentials
                        potential_grasp.append(dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)])

                    # get more potential_grasp by moving along minor_pc
                    if len(potential_grasp) != 0:
                        self.show_points(selected_surface, color='r', scale_factor=.005)
                        for pt in potential_grasp:
                            for dz in range(-5, 5):
                                new_center = minor_pc * dz * 0.01 + pt[0]
                                extra_potential_grasp.append([new_center, pt[1], pt[2], pt[3]])
                approach_dist = self.gripper.hand_depth  # use gripper depth
                num_approaches = int(approach_dist // params['approach_step'])
                for ptg in extra_potential_grasp:
                    for _ in range(num_approaches):
                        tmp_grasp_bottom_center = ptg[1] * params['approach_step'] + ptg[0]
                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        not_collide = self.check_approach_grasp(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, graspable, hand_points)

                        if not not_collide:
                            # if collide, go back one step to get a collision free hand position
                            tmp_grasp_bottom_center = -ptg[1] * params['approach_step'] + ptg[0]
                            # final check
                            open_points, _ = self.check_collision_square(tmp_grasp_bottom_center,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points, "p_open")
                            not_collide = self.check_approach_grasp(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                    tmp_major_pc, minor_pc, graspable, hand_points)
                            if open_points and not_collide:
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc])

                                self.show_points(selected_surface, color='r', scale_factor=.005)
                                if params['debug_vis']:
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, graspable, hand_points,
                                                                "p_open", vis=True)
                        break
                logger.info("processed_potential_grasp %d", len(processed_potential_grasp))

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)
            if not sampled_surface_amount % 60:  # params['debug_vis']:
                self.show_all_grasps(all_points, processed_potential_grasp)
            #
            # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            #     tmp_grasp_center,
            #     tmp_major_pc,
            #     self.gripper.max_width))
            # grasps.append(g)

        return processed_potential_grasp


class OldPointGraspSampler(GraspSampler):
    """
    Sample grasps by PointGraspSampler
    """

    def show_obj(self, graspable, color='b', clear=False):
        if clear:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()

        # plot the obj
        ax = plt.gca(projection='3d')
        surface = graspable.sdf.surface_points()[0]
        surface = surface[np.random.choice(surface.shape[0], 1000, replace=False)]
        ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], '.',
                   s=np.ones_like(surface[:, 0]) * 0.3, c=color)

    def show_grasp_norm(self, graspable, grasp_center, grasp_bottom_center,
                        grasp_normal, grasp_axis, minor_pc, color='b', clear=False):
        if clear:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()

        ax = plt.gca(projection='3d')
        grasp_center_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_center)
        ax.scatter(grasp_center_grid[0], grasp_center_grid[1], grasp_center_grid[2], marker='s', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_bottom_center)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='x', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center + 0.5 * grasp_axis * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='x', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center - 0.5 * grasp_axis * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='x', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center + 0.5 * minor_pc * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='^', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center - 0.5 * minor_pc * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='^', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center + 0.5 * grasp_normal * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='*', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center - 0.5 * grasp_normal * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='*', c=color)

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=1000, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        vis :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 0.3,
            'range_dtheta': 0.30,
            'max_chain_length': 20,
            'max_retry_times': 100
        }

        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        num_surface = surface_points.shape[0]

        i = 0
        self.grasps = []

        # ____count = 0
        # get all grasps
        while len(self.grasps) < num_grasps and i < max_num_samples:
            # print('sample times:', ____count)
            # ____count += 1
            # begin of modification 5: Gaussian over height
            # we can use the top part of the point clouds to generate more sample points
            # min_height = min(surface_points[:, 2])
            # max_height = max(surface_points[:, 2])
            # selected_height = max_height - abs(np.random.normal(max_height, (max_height - min_height)/3)
            #                                    - max_height)
            # ind_10 = np.argsort(abs(surface_points[:, 2] - selected_height))[:10]
            # ind = ind_10[np.random.choice(len(ind_10), 1)]

            # end of modification 5
            ind = np.random.choice(num_surface, size=1, replace=False)
            grasp_bottom_center = surface_points[ind, :]
            grasp_bottom_center = grasp_bottom_center.reshape(3)

            for ind in range(params['max_chain_length']):
                # if not graspable.sdf.on_surface(graspable.sdf.transform_pt_obj_to_grid(grasp_bottom_center))[0]:
                #     print('first damn it!')
                #     from IPython import embed; embed()
                new_grasp_bottom_center = self.sample_chain(grasp_bottom_center, graspable,
                                                            params, vis)
                if new_grasp_bottom_center is None:
                    i += ind
                    break
                else:
                    grasp_bottom_center = new_grasp_bottom_center
            else:
                i += params['max_chain_length']
            print('Chain broken, length:', ind, 'amount:', len(self.grasps))
        return self.grasps

    def sample_chain(self, grasp_bottom_center, graspable, params, vis):
        grasp_success = False
        grasp_normal = graspable.sdf.surface_normal(
            graspable.sdf.transform_pt_obj_to_grid(grasp_bottom_center))
        for normal_dir in [-1., 1.]:  # FIXME: here we assume normal is outward
            grasp_center = self.gripper.max_depth * normal_dir * grasp_normal + grasp_bottom_center
            r_ball = max(self.gripper.max_depth, self.gripper.max_width)
            # cal major principal curvature
            tmp_count = 0
            M = np.zeros((3, 3))
            while tmp_count < params['num_rball_points']:
                neighbor = grasp_bottom_center + 2 * (np.random.rand(3) - 0.5) * r_ball
                normal = graspable.sdf.surface_normal(graspable.sdf.transform_pt_obj_to_grid(neighbor))
                if normal is None:
                    continue
                normal = normal.reshape(-1, 1)
                M += np.matmul(normal, normal.T)
                tmp_count = tmp_count + 1
            eigval, eigvec = np.linalg.eig(M)  # compared computed normal
            minor_pc = eigvec[np.argmin(eigval)]  # minor principal curvature
            minor_pc /= np.linalg.norm(minor_pc)
            new_normal = eigvec[np.argmax(eigval)]  # estimated surface normal
            new_normal /= np.linalg.norm(new_normal)
            # major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
            # FIXME: We can not get accurate normal, so we use grasp_normal instead
            major_pc = np.cross(minor_pc, grasp_normal)
            if np.linalg.norm(major_pc) != 0:
                major_pc = major_pc / np.linalg.norm(major_pc)
            grasp_axis = major_pc

            g = ParallelJawPtGrasp3D(
                ParallelJawPtGrasp3D.configuration_from_params(
                    grasp_center,
                    grasp_axis,
                    self.gripper.max_width))
            grasp_success, _ = g.close_fingers(graspable, vis=vis)
            if grasp_success:
                self.grasps.append(g)
        if not grasp_success:
            return None

        trial = 0
        next_grasp_bottom_center = None
        while trial < params['max_retry_times'] and next_grasp_bottom_center is None:
            trial += 1
            dy = np.random.uniform(-params['num_dy'] * self.gripper.finger_width,
                                   (params['num_dy']) * self.gripper.finger_width)
            dtheta = np.random.uniform(-params['range_dtheta'], params['range_dtheta'])

            for tmp_normal_dir in [-1., 1.]:
                # get new grasp sample from a chain
                # some magic number referred from origin paper

                # compute centers and axes
                x, y, z = minor_pc
                rotation = RigidTransform.rotation_from_quaternion(
                    np.array([dtheta / 180 * np.pi, x, y, z]))
                tmp_grasp_axis = np.dot(rotation, grasp_axis)
                tmp_grasp_normal = np.dot(rotation, grasp_normal)

                tmp_grasp_bottom_center = grasp_bottom_center + tmp_grasp_axis * dy
                # TODO: find contact
                # FIXME: 0.2 is the same as close_finger()
                approach_dist = 0.2
                approach_dist_grid = graspable.sdf.transform_pt_obj_to_grid(approach_dist)
                num_approach_samples = int(Grasp.samples_per_grid * approach_dist_grid / 2)
                approach_loa = ParallelJawPtGrasp3D.create_line_of_action(tmp_grasp_bottom_center,
                                                                          -tmp_grasp_normal * tmp_normal_dir,
                                                                          approach_dist,
                                                                          graspable,
                                                                          num_approach_samples,
                                                                          min_width=0)
                contact_found, contact = ParallelJawPtGrasp3D.find_contact(approach_loa,
                                                                           graspable, vis=vis)
                if not contact_found:
                    continue
                else:
                    if not graspable.sdf.on_surface(graspable.sdf.transform_pt_obj_to_grid(contact.point))[0]:
                        # print('damn it!')
                        pass
                    else:
                        next_grasp_bottom_center = contact.point
                        break
        print('amount:', len(self.grasps), 'next center:', next_grasp_bottom_center)
        return next_grasp_bottom_center


class GpgGraspSamplerPcl(GraspSampler):
    """
    Sample grasps by GPG with pcl directly.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    """

    def sample_grasps(self, point_cloud, points_for_sample, all_normal, num_grasps=20, max_num_samples=200,
                      show_final_grasp=False,
                      **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        point_cloud :
        all_normal :
        num_grasps : int
            the number of grasps to generate

        show_final_grasp :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 10,  # unit degree
            'range_dtheta': 90,
            'debug_vis': False,
            'r_ball': self.gripper.hand_height,
            'approach_step': 0.005,
            'max_trail_for_r_ball': 1000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points
        all_points = point_cloud.to_array()
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # begin of modification 5: Gaussian over height
            # we can use the top part of the point clouds to generate more sample points
            # min_height = min(all_points[:, 2])
            # max_height = max(all_points[:, 2])
            # selected_height = max_height - abs(np.random.normal(max_height, (max_height - min_height)/3)
            #                                    - max_height)
            # ind_10 = np.argsort(abs(all_points[:, 2] - selected_height))[:10]
            # ind = ind_10[np.random.choice(len(ind_10), 1)]
            # end of modification 5

            # for ros, we neded to judge if the robot is at HOME

            if rospy.get_param("/robot_at_home") == "false":
                robot_at_home = False
            else:
                robot_at_home = True
            if not robot_at_home:
                rospy.loginfo("robot is moving! wait untill it go home. Return empty gpg!")
                return []
            scipy.random.seed()  # important! without this, the worker will get a pseudo-random sequences.
            ind = np.random.choice(points_for_sample.shape[0], size=1, replace=False)
            selected_surface = points_for_sample[ind, :].reshape(3, )
            if show_final_grasp:
                mlab.points3d(selected_surface[0], selected_surface[1], selected_surface[2],
                              color=(1, 0, 0), scale_factor=0.005)

            # cal major principal curvature
            # r_ball = params['r_ball']  # FIXME: for some relative small obj, we need to use pre-defined radius
            r_ball = max(self.gripper.hand_outer_diameter - self.gripper.finger_width, self.gripper.hand_depth,
                         self.gripper.hand_height / 2.0)
            # point_amount = params['num_rball_points']
            # max_trial = params['max_trail_for_r_ball']
            # TODO: we can not directly sample from point clouds so we use a relatively small radius.

            M = np.zeros((3, 3))

            # neighbor = selected_surface + 2 * (np.random.rand(3) - 0.5) * r_ball

            selected_surface_pc = pcl.PointCloud(selected_surface.reshape(1, 3))
            kd = point_cloud.make_kdtree_flann()
            kd_indices, sqr_distances = kd.radius_search_for_cloud(selected_surface_pc, r_ball, 100)
            for _ in range(len(kd_indices[0])):
                if sqr_distances[0, _] != 0:
                    # neighbor = point_cloud[kd_indices]
                    normal = all_normal[kd_indices[0, _]]
                    normal = normal.reshape(-1, 1)
                    if np.linalg.norm(normal) != 0:
                        normal /= np.linalg.norm(normal)
                    M += np.matmul(normal, normal.T)
            if sum(sum(M)) == 0:
                print("M matrix is empty as there is no point near the neighbour.")
                print("Here is a bug, if points amount is too little it will keep trying and never go outside.")
                continue
            else:
                logger.info("Selected a good sample point.")

            eigval, eigvec = np.linalg.eig(M)  # compared computed normal
            minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)  # minor principal curvature !!! Here should use column!
            minor_pc /= np.linalg.norm(minor_pc)
            new_normal = eigvec[:, np.argmax(eigval)].reshape(3)  # estimated surface normal !!! Here should use column!
            new_normal /= np.linalg.norm(new_normal)
            major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
            if np.linalg.norm(major_pc) != 0:
                major_pc = major_pc / np.linalg.norm(major_pc)

            # Judge if the new_normal has the same direction with old_normal, here the correct
            # direction in modified meshpy is point outward.
            if np.dot(all_normal[ind], new_normal) < 0:
                new_normal = -new_normal
                minor_pc = -minor_pc

            for normal_dir in [1]:  # FIXED: we know the direction of norm is outward as we know the camera pos
                if params['debug_vis']:
                    # example of show grasp frame
                    self.show_grasp_norm_oneside(selected_surface, new_normal * normal_dir, major_pc * normal_dir,
                                                 minor_pc, scale_factor=0.001)
                    self.show_points(selected_surface, color='g', scale_factor=.002)
                    self.show_points(all_points)
                    # show real norm direction: if new_norm has very diff than pcl cal norm, then maybe a bug.
                    self.show_line(selected_surface, (selected_surface + all_normal[ind] * 0.05).reshape(3))
                    mlab.show()

                # some magic number referred from origin paper
                potential_grasp = []
                for dtheta in np.arange(-params['range_dtheta'],
                                        params['range_dtheta'] + 1,
                                        params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    dtheta = np.float64(dtheta)
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                        (params['num_dy'] + 1) * self.gripper.finger_width,
                                        self.gripper.finger_width):
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc * normal_dir)
                        tmp_grasp_normal = np.dot(rotation, new_normal * normal_dir)
                        tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy
                        # go back a bite after rotation dtheta and translation dy!
                        tmp_grasp_bottom_center = self.gripper.init_bite * (
                                -tmp_grasp_normal * normal_dir) + tmp_grasp_bottom_center

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, all_points,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, all_points,
                                                                       hand_points,
                                                                       "p_bottom")
                        if open_points is True and bottom_points is False:

                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points,
                                                                         "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, all_points,
                                                                          hand_points,
                                                                          "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])

                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        center_dy = dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)]
                        # we check if the gripper has a potential to collide with the table
                        # by check if the gripper is grasp from a down to top direction
                        finger_top_pos = center_dy[0] + center_dy[1] * self.gripper.hand_depth
                        # [- self.gripper.hand_depth * 0.5] means we grasp objects as a angel larger than 30 degree
                        if finger_top_pos[2] < center_dy[0][2] - self.gripper.hand_depth * 0.5:
                            potential_grasp.append(center_dy)

                approach_dist = self.gripper.hand_depth  # use gripper depth
                num_approaches = int(approach_dist / params['approach_step'])

                for ptg in potential_grasp:
                    for approach_s in range(num_approaches):
                        tmp_grasp_bottom_center = ptg[1] * approach_s * params['approach_step'] + ptg[0]
                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, point_cloud, hand_points)

                        if is_collide:
                            # if collide, go back one step to get a collision free hand position
                            tmp_grasp_bottom_center += (-tmp_grasp_normal) * params['approach_step'] * 3
                            # minus 3 means we want the grasp go back a little bitte more.

                            # here we check if the gripper collide with the table.
                            hand_points_ = self.get_hand_points(tmp_grasp_bottom_center,
                                                                tmp_grasp_normal,
                                                                tmp_major_pc)[1:]
                            min_finger_end = hand_points_[:, 2].min()
                            min_finger_end_pos_ind = np.where(hand_points_[:, 2] == min_finger_end)[0][0]

                            safety_dis_above_table = 0.01
                            if min_finger_end < safety_dis_above_table:
                                min_finger_pos = hand_points_[min_finger_end_pos_ind]  # the lowest point in a gripper
                                x = -min_finger_pos[2] * tmp_grasp_normal[0] / tmp_grasp_normal[2] + min_finger_pos[0]
                                y = -min_finger_pos[2] * tmp_grasp_normal[1] / tmp_grasp_normal[2] + min_finger_pos[1]
                                p_table = np.array([x, y, 0])  # the point that on the table
                                dis_go_back = np.linalg.norm([min_finger_pos, p_table]) + safety_dis_above_table
                                tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center - tmp_grasp_normal * dis_go_back
                            else:
                                # if the grasp is not collide with the table, do not change the grasp
                                tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center

                            # final check
                            _, open_points = self.check_collision_square(tmp_grasp_bottom_center_modify,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points, "p_open")
                            is_collide = self.check_collide(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                            tmp_major_pc, minor_pc, all_points, hand_points)
                            if (len(open_points) > 10) and not is_collide:
                                # here 10 set the minimal points in a grasp, we can set a parameter later
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc,
                                                                  tmp_grasp_bottom_center_modify])
                                if params['debug_vis']:
                                    self.show_points(selected_surface, color='r', scale_factor=.005)
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.check_collision_square(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, all_points, hand_points,
                                                                "p_open", vis=True)
                                break
                logger.info("processed_potential_grasp %d", len(processed_potential_grasp))

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)
            print("current amount of sampled surface:", sampled_surface_amount)
            if params['debug_vis']:  # not sampled_surface_amount % 5:
                if len(all_points) > 10000:
                    pc = pcl.PointCloud(all_points)
                    voxel = pc.make_voxel_grid_filter()
                    voxel.set_leaf_size(0.01, 0.01, 0.01)
                    point_cloud = voxel.filter()
                    all_points = point_cloud.to_array()
                self.show_all_grasps(all_points, processed_potential_grasp)
                self.show_points(all_points, scale_factor=0.008)
                mlab.show()
            print("The grasps number got by modified GPG:", len(processed_potential_grasp))
            if len(processed_potential_grasp) >= num_grasps or sampled_surface_amount >= max_num_samples:
                if show_final_grasp:
                    self.show_all_grasps(all_points, processed_potential_grasp)
                    self.show_points(all_points, scale_factor=0.002)
                    mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))
                    table_points = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * 0.5
                    triangles = [(1, 2, 3), (0, 1, 3)]
                    mlab.triangular_mesh(table_points[:, 0], table_points[:, 1], table_points[:, 2],
                                         triangles, color=(0.8, 0.8, 0.8), opacity=0.5)
                    mlab.show()
                return processed_potential_grasp
            #
            # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            #     tmp_grasp_center,
            #     tmp_major_pc,
            #     self.gripper.max_width))
            # grasps.append(g)

        return processed_potential_grasp


class VacuumGraspSampler(GraspSampler):
    """ Sample grasps by sampling pairs of points on the object surface uniformly at random.
    """
    pass


class DexterousVacuumGrasp(GraspSampler):
    def __init__(self, gripper, config, div_radius=8, div_angle=41, graspable=None):
        self.gripper = gripper
        self._configure(config)
        self.gripper.scale_size = 1.0 / self.scale_obj

        self.gripper_radius = self.gripper.radius * self.gripper.scale_size
        self.gripper_radius_vacuum = self.gripper.radius_vacuum * self.gripper.scale_size
        self.gripper_max_depth = self.gripper.max_depth * self.gripper.scale_size

        self.div_radius = int(div_radius)
        self.div_angle = int(div_angle)
        self.graspable = graspable
        self.normals_obj = None

        self.config_pt_map()

    def estimate_surface_normals_sdf(self):
        """
        FUNCTION: compute surface normal for each vertex by sdf
        :return:
        """
        if self.graspable == None:
            logging.info("Load 3D model for the class 'DexterousVacuumGraps' before estimate the surface normal!")
            return

        self.normals_obj = np.zeros([self.graspable.mesh.vertices.shape[0], 3])
        for i in range(0, self.graspable.mesh.vertices.shape[0]):
            pt_grid = self.graspable.sdf.transform_pt_obj_to_grid(self.graspable.mesh.vertices[i, 0:3])
            self.normals_obj[i, 0:3] = self.graspable.sdf.estimate_normals(pt_grid[i, 0:3])

        return

    def estimate_normals_pcl(self):
        """
        FUNCTION: compute surface normal for each vertex by PCL
        :return:
        """
        if self.graspable == None:
            logging.info("Load 3D model for the class 'DexterousVacuumGraps' before estimate the surface normal!")
            return
        ''''''
        pc_obj = pcl.PointCloud(self.graspable.mesh.vertices.astype(np.float32))
        norm = pc_obj.make_NormalEstimation()
        norm.set_KSearch(30)
        # norm.set_RadiusSearch(0.003)
        normals = norm.compute()
        surface_normal = normals.to_array()
        # np.random.shuffle(surface_normal)

        for cnt_pt in range(0, self.graspable.mesh.vertices.shape[0]):

            tmp_pt = self.graspable.mesh.vertices[cnt_pt, :] + surface_normal[cnt_pt, 0:3] * 0.005
            tmp_pt = self.graspable.sdf.transform_pt_obj_to_grid(tmp_pt)
            dist_1 = self.graspable.sdf[tmp_pt]
            # dist_1 = self.graspable.sdf._signed_distance(tmp_pt)

            tmp_pt = self.graspable.mesh.vertices[cnt_pt, :] - surface_normal[cnt_pt, 0:3] * 0.005
            tmp_pt = self.graspable.sdf.transform_pt_obj_to_grid(tmp_pt)
            dist_2 = self.graspable.sdf[tmp_pt]
            # dist_2 = self.graspable.sdf._signed_distance(tmp_pt)

            if dist_1 < dist_2:
                surface_normal[cnt_pt, 0:3] = -1 * surface_normal[cnt_pt, 0:3]

        # surface_normal = surface_normal[:, 0:3]
        self.normals_obj = surface_normal.astype(np.float64)

    def config_pt_map(self, div_radius=8, div_angle=41):
        """
        FUNCTION: configure the points for grasp quality evaluation
        div_radius:             int:            how many steps along with radius direction
                                                set 21 for visualization set 41 for simulation
        div_angle:              int:            how many steps along with angle direction
        :return:
        """
        self.div_radius = int(div_radius)
        self.div_angle = int(div_angle)

        self.layer_radius = np.linspace(0.0, self.gripper_radius, self.div_radius)
        # self.ind_vacuum_radius = int(float(self.div_radius - 1) * (self.gripper_radius_vacuum / self.gripper_radius))
        # self.layer_radius = np.r_[self.layer_radius, self.layer_radius[1::] + self.gripper_radius]
        self.layer_angle = np.linspace(0.0, 360.0, self.div_angle)

        self.step_radius = self.layer_radius[1] - self.layer_radius[0]
        self.step_angle = self.layer_angle[1] - self.layer_angle[0]

        self.pts_base = np.zeros([3, self.div_radius, self.div_angle])

        for i in range(0, div_radius):
            for j in range(0, div_angle):
                self.pts_base[0, i, j] = self.layer_radius[i] * np.cos(np.deg2rad(self.layer_angle[j]))
                self.pts_base[1, i, j] = self.layer_radius[i] * np.sin(np.deg2rad(self.layer_angle[j]))

        return

    def project_pt_map(self, contact_pt, direction, u1, u2, max_dis_project=None):
        """
        FUNCTION: calculate the point map projected to the contact surface

        :param contact_pt:                  obj. contact point
        :param direction:                   grasp axis
        :param u1:                          x-axis of GCS
        :param u2:                          y-axis of GCS
        :param max_dis_project:             max z value when the point projection is empty
        :return:
        """
        pts_project = np.zeros([3, self.div_radius * 3, self.div_angle])
        '''
             0: empty
             1: pt located in vacuum region
             2: pt located in non-vacuum region
            -1: pt located in non-contact region
        '''
        pt_map_project = np.zeros([self.div_radius * 3, self.div_angle], dtype=np.int)
        pt_map_project[0, :] = np.ones(self.div_angle, dtype=np.int)

        normals_project = np.zeros([3, self.div_radius * 3, self.div_angle])
        normals_project[2, 0, :] = -1.0

        if max_dis_project == None:
            max_dis_project = 3.0 * self.gripper_max_depth

        time_start = time.time()
        '''
        pt_GCS, _, _, _ = \
            DexterousVacuumPoint.crop_surface_grasp(contact_pt, direction, u1, u2,
                                                    2 * self.gripper_radius,
                                                    self.gripper_max_depth,
                                                    flg_exclude_opposite=False)
        time_st_1 = time.time()
        # print("time for pts croping: ", time_st_1 - time_start)

        # ind_sort = pt_GCS[:, 0].argsort()
        # pt_GCS[:, :] = pt_GCS[ind_sort, :]
        # pt_dist = np.zeros(pt_GCS.shape[0])
        # find the min distance to (0, 0, 0)
        min_dist = 99999.0
        for i in range(0, pt_GCS.shape[0]):
            if np.linalg.norm(pt_GCS[i, :]) < min_dist:
                min_dist = np.linalg.norm(pt_GCS[i, :])
        '''
        rad_rot = self.step_radius
        vector_base = np.array([rad_rot, 0.0, 0.0])
        ax_rot = np.array([0.0, 1.0, 0.0])
        min_x_offset = self.step_radius * 0.3
        max_angle_rot = np.arccos(min_x_offset / rad_rot)
        '''define the distance between each rotation according to the threshold of 
            surface judgement (<0.5*surface_thresh)
        '''
        step_dis_rot = self.graspable.sdf.surface_thresh * 0.001
        # step_angle_rot = 2.0 * np.arcsin(0.5 * step_dis_rot / rad_rot)
        for cnt_steps in range(3, 503, 2):
            array_rot = np.linspace(-1.0 * max_angle_rot, 1.0 * max_angle_rot, cnt_steps)
            step_dis = 2.0 * rad_rot * np.sin(0.5 * (array_rot[1] - array_rot[0]))
            if step_dis < step_dis_rot:
                break

        vectors_rot = np.zeros([array_rot.shape[0], 3])
        # vector_rot_horizontal = [, 0.0, 0.0]
        for cnt_rot in range(array_rot.shape[0]):
            vectors_rot[cnt_rot, :] = math_robot.rot_along_axis(ax_rot, -1 * array_rot[cnt_rot], vector_base)

        ax_rot = np.array([0.0, 0.0, 1.0])
        # vectors_offest = np.zeros([array_rot.shape[0], 3])
        # pts_new_GCS = np.zeros([array_rot.shape[0], 3])
        # a = np.c_[self.pt_x_base[:, 1], self.pt_y_base[:, 1], self.pt_z_base[:, 1]

        # find the point when radius = 0.0 at pt(0, 0, ?)
        resolution_zero = 501
        z_zero_pts = np.linspace(-0.003, 0.003, resolution_zero).reshape([resolution_zero, 1])
        pts_zero_GCS = np.c_[
            np.zeros([resolution_zero, 2]), z_zero_pts, np.ones([resolution_zero, 1], dtype=np.float64)]
        pts_zero_obj = math_robot.transfer_CS_reverse(u1, u2, direction, 1.0 * contact_pt.point, pts_zero_GCS)
        pts_zero_grid = self.graspable.sdf.transform_pt_obj_to_grid(pts_zero_obj[:, 0:3].T)
        dists = self.graspable.sdf.get_signed_distances(pts_zero_grid)
        dists = abs(dists)
        ind_pt_nearest = dists.argmin()
        pt_ideal_zero_GCS = pts_zero_GCS[ind_pt_nearest, :]
        pt_ideal_zero_GCS[0:2] = [0.0, 0.0]
        for i in range(0, self.div_angle):
            pts_project[:, 0, i] = pt_ideal_zero_GCS[0: 3]

        time_st_2 = time.time()
        # print("time for pts initialization: ", time_st_2 - time_st_1)
        for cnt_angle in range(0, self.div_angle - 1):

            vectors_offest = math_robot.rot_along_axis(ax_rot, np.deg2rad(self.layer_angle[cnt_angle]), vectors_rot)
            flg_vacuum_region = True
            rad_offset = 0.0
            flg_contact = True
            for cnt_rad in range(1, pts_project.shape[1]):
                pt_org_GCS = np.array([pts_project[0, cnt_rad - 1, cnt_angle],
                                       pts_project[1, cnt_rad - 1, cnt_angle],
                                       pts_project[2, cnt_rad - 1, cnt_angle]])
                # near to be the boundary of gripper
                vector_ref = [pt_org_GCS[0], pt_org_GCS[1], 0.0]
                rad_curr = np.linalg.norm(vector_ref)
                if rad_curr > self.gripper_radius - 0.001:
                    break

                if flg_contact:
                    if rad_offset + self.step_radius > self.gripper_radius_vacuum and flg_vacuum_region == True:
                        pts_new_GCS = pt_org_GCS + vectors_offest / \
                                      self.step_radius * (self.gripper_radius_vacuum - rad_offset)
                        rad_offset = self.gripper_radius_vacuum
                        flg_vacuum_region = False
                    else:
                        pts_new_GCS = pt_org_GCS + vectors_offest
                        rad_offset += self.step_radius

                    pts_new_GCS = np.c_[pts_new_GCS, np.ones([pts_new_GCS.shape[0], 1])]
                    pts_new_obj = math_robot.transfer_CS_reverse(u1, u2, direction, 1.0 * contact_pt.point, pts_new_GCS)
                    # pt_tmp = math_robot.transfer_CS(direction, u1, u2, 1.0 * contact_pt.point, pts_new_obj)
                    pts_new_grid = self.graspable.sdf.transform_pt_obj_to_grid(pts_new_obj[:, 0:3].T)
                    dists = self.graspable.sdf.get_signed_distances(pts_new_grid)

                    dists = abs(dists)
                    ind_pts = dists.argsort()
                    dists[:] = dists[ind_pts]
                    pts_new_grid[:, :] = pts_new_grid[:, ind_pts]
                    pts_new_obj[:, :] = pts_new_obj[ind_pts, :]
                    pts_new_GCS[:, :] = pts_new_GCS[ind_pts, :]

                    flg_onsurface, _ = self.graspable.sdf.on_surface(pts_new_grid[:, 0])
                    if flg_onsurface:
                        pt_ideal_grid = pts_new_grid[:, 0]
                        pt_ideal_obj = pts_new_obj[0, 0:3]
                        pt_ideal_GCS = pts_new_GCS[0, 0:3]

                        normal_idea_obj = self.graspable.sdf.surface_normal(pt_ideal_grid)
                        normal_idea_obj = np.array([normal_idea_obj[0], normal_idea_obj[1], normal_idea_obj[2], 1.0])
                        '''
                        normal_ideal_GCS = math_robot.transfer_CS(u1, u2, direction, 1.0 * contact_pt.point,
                                                                  normal_idea_obj)
                        '''
                        normal_ideal_GCS = math_robot.transfer_CS(u1, u2, direction, np.zeros(3),
                                                                  normal_idea_obj)
                        '''
                        if normal_ideal_GCS[2] > 0.0:
                            normal_ideal_GCS[2] = -1.0 * normal_ideal_GCS[2]
                        '''
                        pts_project[0, cnt_rad, cnt_angle] = pt_ideal_GCS[0]
                        pts_project[1, cnt_rad, cnt_angle] = pt_ideal_GCS[1]
                        pts_project[2, cnt_rad, cnt_angle] = pt_ideal_GCS[2]
                        normals_project[0, cnt_rad, cnt_angle] = normal_ideal_GCS[0]
                        normals_project[1, cnt_rad, cnt_angle] = normal_ideal_GCS[1]
                        normals_project[2, cnt_rad, cnt_angle] = normal_ideal_GCS[2]
                        if rad_offset <= self.gripper_radius_vacuum:
                            pt_map_project[cnt_rad, cnt_angle] = 1
                        else:
                            pt_map_project[cnt_rad, cnt_angle] = 2
                    else:
                        flg_contact = False
                        # make sure new step is not out of max_radius
                        vector_ref = np.array([pt_org_GCS[0], pt_org_GCS[1], 0.0])
                        rad_curr = np.linalg.norm(vector_ref)
                        if self.gripper_radius - rad_curr > 0.001:
                            vector_step = np.array([self.step_radius, 0.0, 0.0])
                        else:
                            vector_step = np.array([self.gripper_radius - rad_curr, 0.0, 0.0])
                        #
                        vectors_step_rot = \
                            math_robot.rot_along_axis(ax_rot, np.deg2rad(self.layer_angle[cnt_angle]), vector_step)

                        pt_new = np.array([pt_org_GCS[0] + vectors_step_rot[0],
                                           pt_org_GCS[1] + vectors_step_rot[1],
                                           max_dis_project])

                        pts_project[:, cnt_rad, cnt_angle] = pt_new

                        pt_map_project[cnt_rad, cnt_angle] = -1
                        normals_project[:, cnt_rad, cnt_angle] = [0.0, 0.0, -1.0]
                else:
                    # make sure new step is not out of max_radius
                    vector_ref = np.array([pt_org_GCS[0], pt_org_GCS[1], 0.0])
                    rad_curr = np.linalg.norm(vector_ref)
                    if self.gripper_radius - rad_curr > 0.001:
                        vector_step = np.array([self.step_radius, 0.0, 0.0])
                    else:
                        vector_step = np.array([self.gripper_radius - rad_curr, 0.0, 0.0])
                    #
                    vectors_step_rot = \
                        math_robot.rot_along_axis(ax_rot, np.deg2rad(self.layer_angle[cnt_angle]), vector_step)

                    pt_new = np.array([pt_org_GCS[0] + vectors_step_rot[0],
                                       pt_org_GCS[1] + vectors_step_rot[1],
                                       max_dis_project])
                    pts_project[:, cnt_rad, cnt_angle] = pt_new

                    pt_map_project[cnt_rad, cnt_angle] = -1
                    normals_project[:, cnt_rad, cnt_angle] = [0.0, 0.0, -1.0]

        time_st_3 = time.time()
        # print("time for pts projection: ", time_st_3 - time_st_2)

        pts_project[0, :, -1] = pts_project[0, :, 0]
        pts_project[1, :, -1] = pts_project[1, :, 0]
        pts_project[2, :, -1] = pts_project[2, :, 0]
        pt_map_project[:, -1] = pt_map_project[:, 0]
        normals_project[0:, :, -1] = normals_project[0, :, 0]
        normals_project[1:, :, -1] = normals_project[1, :, 0]
        normals_project[2:, :, -1] = normals_project[2, :, 0]

        # only return the valid rows
        tmp_sum = np.sum(pt_map_project, axis=1)
        ind_zero = np.squeeze(np.argwhere(tmp_sum[:] == 0))
        if len(ind_zero.shape) > 0:
            ind_zero = ind_zero[ind_zero.argsort()]
            pt_map_project = pt_map_project[0:ind_zero[0], :]
            pts_project = pts_project[:, 0:ind_zero[0], :]
            normals_project = normals_project[:, 0:ind_zero[0], :]

        time_st_4 = time.time()
        # print("time for re-collection: ", time_st_4 - time_st_3)
        return pts_project, pt_map_project, normals_project

    def set_scale_obj(self, scale_obj=1.0):
        """
        NOTE: chang size of object for data augmentation. When the object is amplfied, the gripper is shrinked relatively.
        """
        self.scale_obj = scale_obj
        self.gripper.scale_size = 1.0 / scale_obj
        self.gripper_radius = self.gripper.radius * self.gripper.scale_size
        self.gripper_max_depth = self.gripper.max_depth * self.gripper.scale_size
        self.gripper_radius_vacuum = self.gripper.radius_vacuum * self.gripper.scale_size
        self.config_pt_map()

    def check_distance(self, cur_pos, pos_list, dis_thresh):
        if pos_list is not None:
            for i, (x, y, z) in enumerate(pos_list):
                dis = np.linalg.norm(cur_pos - [x, y, z])
                if dis < dis_thresh:
                    return False
        return True

    def sample_grasps(self, graspable, curr_grasp=0, vis=False, **kwargs):
        """
        FUNCTION: Sample grasps for dexterous vacuum gripper by sampling pairs of points on the object surface.
        Returns a list of candidate grasps for graspable object.
        ------------------------------
        [Parameters]
        ------------------------------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        vis : bool
            whether or not to visualize progress, for debugging
        **kwargs:

        ------------------------------
        [Returns]
        ------------------------------
        :flg_success:
        :new_grasps: new_grasp info:
        """
        if 'vis_surface' in kwargs and kwargs['vis_surface']:
            vis_surface = kwargs['vis_surface']
        else:
            vis_surface = False
        if 'mesh_gripper' in kwargs and kwargs['mesh_gripper']:
            mesh_gripper = kwargs['mesh_gripper']
        else:
            mesh_gripper = None
        if 'multi_approach_angle' in kwargs and kwargs['multi_approach_angle']:
            multi_approach_angle = kwargs['multi_approach_angle']
        else:
            multi_approach_angle = False
        if 'flg_desample_g' in kwargs and kwargs['flg_desample_g']:
            flg_desample_g = kwargs['flg_desample_g']
        else:
            flg_desample_g = False
        if 'dim_g_matrix' in kwargs and kwargs['dim_g_matrix']:
            dim_g_matrix = kwargs['dim_g_matrix']
        else:
            dim_g_matrix = 100
        if 'num_repeat_QP' in kwargs and kwargs['num_repeat_QP']:
            num_repeat_QP = kwargs['num_repeat_QP']
        else:
            num_repeat_QP = 3

        if 'curr_surface_pt' in kwargs and kwargs['curr_surface_pt'] is not None:
            x1 = kwargs['curr_surface_pt']
            # print("sfsfsd", x1)
            # input()
        else:
            x1 = np.array([0.01878754, -0.04554584, 0.09333681])
        ''' For test
        x1 = np.array([-0.045040, 0.030736, 0.063758])
        '''

        logger.info('current_point: %f, %f, %f' % (x1[0], x1[1], x1[2]))
        grasps = []
        # start_time = time.clock()

        c1 = Contact3D(graspable, x1, in_direction=None)
        # dir is towards to object (face inward)
        dir, t1, t2 = c1.tangents()

        # tmp_data = np.copy(c1.graspable.mesh.vertices).astype(np.float32)
        # print("tmp_data:", tmp_data.shape)

        # pc_obj = pcl.PointCloud(tmp_data)
        ''''''
        # print("angle_range:", angle_range)

        if multi_approach_angle:
            tmp_1 = np.linspace(self.angle_range_min, self.angle_range_max, self.num_angle_steps)
            tmp_2 = np.linspace(-self.angle_range_max, -self.angle_range_min, self.num_angle_steps)
            angle_scales = np.r_[tmp_2, 0.0, tmp_1]
            # angle_scales = np.c_[angle_scales, angle_scales]
            angle_scales_it = it.product(angle_scales, repeat=2)
        else:
            angle_scales = np.zeros([1, 1])
            angle_scales_it = it.product(angle_scales, repeat=2)
        # max_z is used to check point clouds qualities
        # max_z = 3 * (self.gripper_radius * 2) * math.tan(math.radians(self.angle_range))

        '''
        https://zhuanlan.zhihu.com/p/56587491
        point P(x0,y0,z0) rotates angle theta along with the axis (x,y,z), than the position of P should be
                              |cos + x^2(1-cos)    xy(1-cos) - zsin    xz(1-cos) + ysin|
        P' = (x0',y0',z0')T = |xy(1-cos) + zsin    cos + y^2(1-cos)    yz(1-cos) - xsin| * (x0,y0,z0)T
                              |zx(1-cos) - ysin    zy(1-cos) + xsin    cos + z^2(1-cos)|
        CAUTION: (x,y,z) must be a normalized vector
        '''

        for i, (an_1, an_2) in enumerate(angle_scales_it):
            print("num_curr_grasp_{}, Rot angles: ({}, {})".format(curr_grasp, an_1, an_2))
            # input()
            ax_z = math_robot.rot_along_axis(1.0 * t1, math.radians(an_1), 1.0 * dir)
            ax_x = math_robot.rot_along_axis(1.0 * t1, math.radians(an_1), 1.0 * t1)
            ax_y = math_robot.rot_along_axis(1.0 * t1, math.radians(an_1), 1.0 * t2)
            ax_z = ax_z / np.linalg.norm(ax_z)
            ax_x = ax_x / np.linalg.norm(ax_x)
            ax_y = ax_y / np.linalg.norm(ax_y)

            ax_z = math_robot.rot_along_axis(1.0 * ax_y, math.radians(an_2), 1.0 * ax_z)
            ax_x = math_robot.rot_along_axis(1.0 * ax_y, math.radians(an_2), 1.0 * ax_x)
            ax_y = math_robot.rot_along_axis(1.0 * ax_y, math.radians(an_2), 1.0 * ax_y)

            time_start = time.time()
            pts_project, map_project, normals_project = self.project_pt_map(c1, ax_z, ax_x, ax_y)
            time_st_1 = time.time()
            # print("time for pt projection: ", time_st_1 - time_start)

            G_matrix, P_air, max_force_vacuum = \
                DexterousVacuumPoint.grasp_matrix(pts_project=pts_project,
                                                  map_project=map_project,
                                                  normals_project=normals_project,
                                                  radius_gripper=self.gripper_radius,
                                                  radius_vacuum=self.gripper_radius_vacuum,
                                                  flg_adv_air_pressure=True,
                                                  type_ap=1)
            # print("G_matrix: ", G_matrix)
            time_st_2 = time.time()
            # print("time for pt G_matrix generation: ", time_st_2 - time_st_1)

            Quality_analyzer = \
                DexterousQuality_Vacuum(grasp_matrix=1.0 * G_matrix,
                                        max_force_vacuum=max_force_vacuum,
                                        radius_gripper=self.gripper_radius,
                                        radius_vacuum=self.gripper_radius_vacuum,
                                        differ=0.3,
                                        total_error=0.05)
            _, sol, dist, quality = \
                Quality_analyzer.analysis_quality(flg_desampling=flg_desample_g,
                                                  num_dim=dim_g_matrix,
                                                  repeat=num_repeat_QP)
            time_st_3 = time.time()
            # print("time for pt Gmatrix generation: ", time_st_3 - time_st_2)

            # print(flg_in)
            # print(sol.message)
            print(sol.fun)
            # print(sol.status)
            # print("E_distance:", dist)
            print("Quality: ", quality)
            # print("num_surface_points:", c1.graspable.surface_points.shape[0])
            str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            # print("x1: ", x1)
            g_info = GraspInfo(name_grasp=str(self.graspable.model_name_ + "_" + str_time),
                               g_rad=1.0 * self.gripper.radius,
                               pos=1.0 * c1.point,
                               dir=1.0 * ax_z,
                               t1=1.0 * ax_x,
                               t2=1.0 * ax_y,
                               quality=1.0 * quality,
                               pts_project=1.0 * pts_project,
                               map_project=1.0 * map_project,
                               normals_project=1.0 * normals_project,
                               grasp_matrix=1.0 * G_matrix,
                               rot_x=0.0, rot_y=0.0)
            print(g_info.get_name_grasp())
            grasps.append(g_info)
            """
            with open('test_save_grasp_info.pickle', 'wb') as f:
                pickle.dump(grasps, f)
            lst_grasp_info = pickle.load(open('test_save_grasp_info.pickle', 'rb'))
            for cnt, info in enumerate(lst_grasp_info):
                print(info.get_name_grasp())
            """
        if grasps == []:
            return False, grasps
        else:
            return True, grasps

    def sample_grasps_test(self, graspable, curr_grasp=0, vis=False, **kwargs):
        """
        FUNCTION: Sample grasps for dexterous vacuum gripper by sampling pairs of points on the object surface.
        Returns a list of candidate grasps for graspable object.
        ------------------------------
        [Parameters]
        ------------------------------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        vis : bool
            whether or not to visualize progress, for debugging
        **kwargs:

        ------------------------------
        [Returns]
        ------------------------------
        :flg_success:
        :new_grasps: new_grasp info:
        """
        if 'vis_surface' in kwargs and kwargs['vis_surface']:
            vis_surface = kwargs['vis_surface']
        else:
            vis_surface = True
        if 'mesh_gripper' in kwargs and kwargs['mesh_gripper']:
            mesh_gripper = kwargs['mesh_gripper']
        else:
            mesh_gripper = None
        if 'flg_desample_g' in kwargs and kwargs['flg_desample_g']:
            flg_desample_g = kwargs['flg_desample_g']
        else:
            flg_desample_g = False
        if 'dim_g_matrix' in kwargs and kwargs['dim_g_matrix']:
            dim_g_matrix = kwargs['dim_g_matrix']
        else:
            dim_g_matrix = 100
        if 'num_repeat_QP' in kwargs and kwargs['num_repeat_QP']:
            num_repeat_QP = kwargs['num_repeat_QP']
        else:
            num_repeat_QP = 3

        if 'curr_surface_pt' in kwargs and kwargs['curr_surface_pt'] is not None:
            x1 = kwargs['curr_surface_pt']
            # print("sfsfsd", x1)
            # input()
        else:
            x1 = np.array([-0.045040, 0.030736, 0.063758])
        ''' For test
        x1 = np.array([-0.045040, 0.030736, 0.063758])
        '''

        logger.info('current_point: %f, %f, %f' % (x1[0], x1[1], x1[2]))
        grasps = []
        # start_time = time.clock()

        c1 = Contact3D(graspable, x1, in_direction=None)
        # dir is towards to object (face inward)
        dir, t1, t2 = c1.tangents()

        # tmp_data = np.copy(c1.graspable.mesh.vertices).astype(np.float32)
        # print("tmp_data:", tmp_data.shape)

        # pc_obj = pcl.PointCloud(tmp_data)
        ''''''
        # print("angle_range:", angle_range)

        angle_scales = np.zeros([1, 1])
        angle_scales_it = it.product(angle_scales, repeat=2)
        # max_z is used to check point clouds qualities
        # max_z = 3 * (self.gripper_radius * 2) * math.tan(math.radians(self.angle_range))

        '''
        https://zhuanlan.zhihu.com/p/56587491
        point P(x0,y0,z0) rotates angle theta along with the axis (x,y,z), than the position of P should be
                              |cos + x^2(1-cos)    xy(1-cos) - zsin    xz(1-cos) + ysin|
        P' = (x0',y0',z0')T = |xy(1-cos) + zsin    cos + y^2(1-cos)    yz(1-cos) - xsin| * (x0,y0,z0)T
                              |zx(1-cos) - ysin    zy(1-cos) + xsin    cos + z^2(1-cos)|
        CAUTION: (x,y,z) must be a normalized vector
        '''

        for i, (an_1, an_2) in enumerate(angle_scales_it):
            print("num_curr_grasp_{}, Rot angles: ({}, {})".format(curr_grasp, an_1, an_2))
            # input()
            ax_z = math_robot.rot_along_axis(1.0 * t1, math.radians(an_1), 1.0 * dir)
            ax_x = math_robot.rot_along_axis(1.0 * t1, math.radians(an_1), 1.0 * t1)
            ax_y = math_robot.rot_along_axis(1.0 * t1, math.radians(an_1), 1.0 * t2)
            ax_z = ax_z / np.linalg.norm(ax_z)
            ax_x = ax_x / np.linalg.norm(ax_x)
            ax_y = ax_y / np.linalg.norm(ax_y)

            ax_z = math_robot.rot_along_axis(1.0 * ax_y, math.radians(an_2), 1.0 * ax_z)
            ax_x = math_robot.rot_along_axis(1.0 * ax_y, math.radians(an_2), 1.0 * ax_x)
            ax_y = math_robot.rot_along_axis(1.0 * ax_y, math.radians(an_2), 1.0 * ax_y)

            time_start = time.time()

            pts_project, map_project, normals_project = self.project_pt_map(c1, ax_z, ax_x, ax_y)
            time_st_1 = time.time()
            # print("time for pt projection: ", time_st_1 - time_start)

            G_matrix, P_air, max_force_vacuum = \
                DexterousVacuumPoint.grasp_matrix(pts_project=pts_project,
                                                  map_project=map_project,
                                                  normals_project=normals_project,
                                                  radius_gripper=self.gripper_radius,
                                                  radius_vacuum=self.gripper_radius_vacuum,
                                                  flg_adv_air_pressure=True,
                                                  type_ap=1)
            # print("G_matrix: ", G_matrix)
            time_st_2 = time.time()
            # print("time for pt G_matrix generation: ", time_st_2 - time_st_1)

            Quality_analyzer = \
                DexterousQuality_Vacuum(grasp_matrix=1.0 * G_matrix,
                                        max_force_vacuum=max_force_vacuum,
                                        radius_gripper=self.gripper_radius,
                                        radius_vacuum=self.gripper_radius_vacuum,
                                        differ=0.3,
                                        total_error=0.05)
            _, sol, dist, quality = \
                Quality_analyzer.analysis_quality(flg_desampling=flg_desample_g,
                                                  num_dim=dim_g_matrix,
                                                  repeat=num_repeat_QP)
            time_st_3 = time.time()
            # print("time for pt Gmatrix generation: ", time_st_3 - time_st_2)

            # print(flg_in)
            # print(sol.message)
            print(sol.fun)
            # print(sol.status)
            # print("E_distance:", dist)
            print("Quality: ", quality)
            # print("num_surface_points:", c1.graspable.surface_points.shape[0])
            str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            # print("x1: ", x1)
            g_info = GraspInfo(name_grasp=str(self.graspable.model_name_ + "_" + str_time),
                               g_rad=1.0 * self.gripper.radius,
                               pos=1.0 * c1.point,
                               dir=1.0 * ax_z,
                               t1=1.0 * ax_x,
                               t2=1.0 * ax_y,
                               quality=1.0 * quality,
                               pts_project=1.0 * pts_project,
                               map_project=1.0 * map_project,
                               normals_project=1.0 * normals_project,
                               grasp_matrix=1.0 * G_matrix,
                               rot_x=0.0, rot_y=0.0)
            print(g_info.get_name_grasp())
            print(g_info.dir_grasp)
            print(g_info.t1_grasp)
            print(g_info.t2_grasp)
            grasps.append(g_info)
            """
            with open('test_save_grasp_info.pickle', 'wb') as f:
                pickle.dump(grasps, f)
            lst_grasp_info = pickle.load(open('test_save_grasp_info.pickle', 'rb'))
            for cnt, info in enumerate(lst_grasp_info):
                print(info.get_name_grasp())
            """

            # display in GCS (gripper coorinates system)
            if vis_surface:
                gl_vis = GL_Visualizer()
                ax_width = 5.0
                x_ax = np.array([0.0, .0, .0, 2.0, .0, .0])
                gl_vis.draw_lines(x_ax, width=ax_width, num_color=1)
                y_ax = np.array([0.0, .0, .0, 0.0, 2.0, .0])
                gl_vis.draw_lines(y_ax, width=ax_width, num_color=2)
                z_ax = np.array([0.0, .0, .0, 0.0, .0, 2.0])
                gl_vis.draw_lines(z_ax, width=ax_width, num_color=4)
                # display obj
                obj_vertices = graspable.mesh.vertices
                obj_normals = graspable.mesh.normals
                obj_vertices = math_robot.transfer_CS(g_info.t1_grasp, g_info.t2_grasp,
                                                      g_info.dir_grasp, g_info.pos_grasp,
                                                      obj_vertices)[:, 0:3]
                obj_normals = math_robot.transfer_CS(g_info.t1_grasp, g_info.t2_grasp,
                                                     g_info.dir_grasp, g_info.pos_grasp,
                                                     obj_normals)[:, 0:3]
                obj_normals = obj_normals / np.linalg.norm(obj_normals, axis=1).\
                    reshape([obj_normals.shape[0], 1])

                gl_vis.display_mesh(vertices=100 * obj_vertices,
                                    triangles=graspable.mesh.triangles,
                                    v_normals=obj_normals)
                # gl_vis.UI_lisener()
                # display grasp center
                gl_vis.draw_spheres(pos_spheres=np.array([0.0, 0.0, -0.5]), radius=0.2, num_color=1)


                if mesh_gripper is not None:
                    # display gripper
                    ''''''
                    offset_gripper = np.array([0., 0., -8.0]).reshape([1, 3])  # cm
                    # gripper_vertices = 0.1 * mesh_gripper.vertices + offset_gripper
                    gripper_vertices = 0.1 * mesh_gripper.vertices * 2.0 + offset_gripper
                    gripper_normals = mesh_gripper.normals
                    gl_vis.display_mesh(vertices=gripper_vertices, triangles=mesh_gripper.triangles,
                                        v_normals=gripper_normals)

                    # display projected map
                    pts_project[2, :, :] = pts_project[2, :, :] - 0.006
                    pts_dis_1 = np.zeros([1, 3])
                    for i in range(0, map_project.shape[0]):
                        for j in range(0, map_project.shape[1]):
                            if map_project[i, j] == 1:
                                pts_dis_1 = np.r_[pts_dis_1, np.array([pts_project[0, i, j],
                                                                       pts_project[1, i, j],
                                                                       pts_project[2, i, j]]).reshape([1, 3])]
                    pts_dis_1 = pts_dis_1[1::, :]
                    gl_vis.draw_spheres(pos_spheres=100. * pts_dis_1, radius=0.1, num_color=2)

                    pts_dis_2 = np.zeros([1, 3])
                    for i in range(0, map_project.shape[0]):
                        for j in range(0, map_project.shape[1]):
                            if map_project[i, j] == 2:
                                pts_dis_2 = np.r_[pts_dis_2, np.array([pts_project[0, i, j],
                                                                       pts_project[1, i, j],
                                                                       pts_project[2, i, j]]).reshape([1, 3])]
                    pts_dis_2 = pts_dis_2[1::, :]
                    gl_vis.draw_spheres(pos_spheres=100. * pts_dis_2, radius=0.1, num_color=3)

                    ''''''
                    # display projected lights
                    offset_gripper[0, 2] = offset_gripper[0, 2] - 1
                    pos_p2p = np.c_[100.0 * pts_dis_1, 90.0 * pts_dis_1+offset_gripper]
                    gl_vis.draw_lines(pos_p2p=pos_p2p, num_color=7)

                    pos_p2p = np.c_[100.0 * pts_dis_2, 90.0 * pts_dis_2+offset_gripper]
                    gl_vis.draw_lines(pos_p2p=pos_p2p, num_color=7)

                    # display mesh lines
                    pts1, pts2 = DexterousVacuumPoint.visualize_points(pts_project, map_project)
                    gl_vis.draw_lines(pos_p2p=100.0*pts1, width=3, num_color=2)
                    gl_vis.draw_lines(pos_p2p=100.0*pts2, width=3, num_color=3)


                print("visualization completed!")
                gl_vis.UI_lisener()

                ''''''
        if grasps == []:
            return False, grasps
        else:
            return True, grasps

    def generate_grasps_dex_vacuum(self, graspable, target_num_grasps=None, grasp_gen_mult=5,
                                   **kwargs):
        """Samples a set of grasps for an object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        vis : bool
            whether show the grasp on picture

        Return
        """
        if 'flg_test' in kwargs and kwargs['flg_test']:
            flg_test = kwargs['flg_test']
        else:
            flg_test = False

        self.graspable = graspable
        # self.estimate_normals_sdf()

        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        # print("obj.sdf.surface_points:", surface_points.shape)
        logger.info('Sampler is Launched...')

        # get num grasps
        if target_num_grasps is None:
            target_num_grasps = self.target_num_grasps
        num_grasps_remaining = target_num_grasps
        grasps = []

        x1 = None
        ''''''
        logger.info('num_grasps_remaining: %d' % (num_grasps_remaining))
        max_loop = int(math.ceil(float(num_grasps_remaining) / 100.))
        while grasp_gen_mult > 0:
            grasp_gen_mult -= 1
            for cnt_loop in range(0, max_loop):
                np.random.shuffle(surface_points)
                ''' for 02 master can'''
                # -0.020132, -0.031535, 0.004600  on the can bottle
                # 0.006333, 0.033850, 0.021725    on the can side
                # -0.045040, 0.030736, 0.063758   on the can side
                # 0.031242, -0.025308, 0.099564
                # -0.04192688, -0.05332974, 0.00148679  # q = 0.507
                ''' for 01 banana'''
                # -0.01628514, 0.042222, 0.00409378
                shuffled_surface_points = surface_points[:min(self.max_num_surface_points_, len(surface_points))]
                if flg_test:
                    print(shuffled_surface_points[0])
                    shuffled_surface_points[0] = [-0.01628514, 0.042222, 0.00409378]

                    x1 = shuffled_surface_points[0]
                    flg_sucess, new_grasps = \
                        self.sample_grasps_test(graspable, curr_surface_pt=x1,
                                           curr_grasp=5, **kwargs)
                lst_grasped = []
                cnt_fail = 0
                for i in range(0, 100):
                    if cnt_fail > 30:  # success is too low, give up this batch
                        break
                    if num_grasps_remaining > 0:
                        x1 = shuffled_surface_points[i]
                        flg_dis = self.check_distance(x1, lst_grasped, self.gripper.radius * 0.3)
                        if flg_dis == False:
                            continue

                        flg_sucess, new_grasps = \
                            self.sample_grasps(graspable, curr_surface_pt=x1,
                                               curr_grasp=num_grasps_remaining, **kwargs)

                        if flg_sucess:
                            lst_grasped.append(np.array([x1[0], x1[1], x1[2]]))
                            grasps = grasps + new_grasps
                            num_grasps_remaining -= 1
                            logger.info('num_grasps_remaining: %d' % (num_grasps_remaining))
                            # print(grasps)
                        else:
                            cnt_fail += 1
                    else:
                        return grasps
                if num_grasps_remaining < 1:
                    return grasps
            if num_grasps_remaining < 1:
                return grasps
        return grasps

    def rot_pt_xy_axes(self, org_pc=np.zeros([1, 3]), rotate_x=0., rotate_y=0.):
        """
        FUNCTION: Rotate points along with x, y axes for data augmentation
        :param org_pc:
        :param rotate_x: metric(degree)
        :param rotate_y:
        :return:
        """
        if not org_pc.shape[1] == 3:
            print("Error from DexterousVacuumGrasp.desampling_pc: The shape of point cloud is not correct.")
            return False, np.zeros([1, 3])
        pcl_vc = np.copy(org_pc[:, 0:3])
        # pcl_vc = pcl_vc.T
        # print(np.min(pcl_vc, axis=0))
        ''''''
        ax_x = np.array([1., 0., 0.])
        ax_y = np.array([0., 1., 0.])
        ax_z = np.array([0., 0., 1.])

        pcl_vc = math_robot.rot_along_axis(ax_x, math.radians(rotate_x), pcl_vc)

        pcl_vc = math_robot.rot_along_axis(ax_y, math.radians(rotate_y), pcl_vc)

        return True, pcl_vc.T

    def random_disturbance(self, size_pc, noise=0.0):
        noise = (np.random.random([size_pc, 3]) * 2.0 - np.ones([size_pc, 3])) * noise
        return noise

    def desampling_pc(self, org_pc=np.zeros([1, 3]), resolution_pc=24, scale_obj=1.0,
                      thresh_pt_subgrid=1,
                      flg_disturb_bd=False, noise_bd=0.0,
                      max_dis_project=None):
        """
        FUNCTION: desampling point cloud based on resolution
        :param org_pc:               n*3 array:  original point cloud
        :param resolution_pc:        int:        resolution of point cloud (e.g.: 24*24) EVEN NUMBER
        :param scale_obj:            float:      re-scale factor for object size
        :param thresh_pt_subgrid:    int:        threshold for each sub grid
        :param flg_disturb_bd:       bool:       add disturbance before desampling
        :param noise_bd:             float:      noise level (metirc: m)
        :return:
            flg_success:    bool
            output_pc:      point cloud after desampling [3, resolution_pc, resolution_pc]
        """
        if not org_pc.shape[1] == 3:
            print("Error from DexterousVacuumGrasp.desampling_pc: The shape of point cloud is not correct.")
            return False, np.zeros([1, 3])
        if max_dis_project == None:
            max_dis_project = 3.0 * self.gripper_radius

        if flg_disturb_bd:
            ns_bd = self.random_disturbance(org_pc.shape[0], noise=noise_bd)
            org_pc[:, 2] = org_pc[:, 2] + ns_bd[:, 2]

        size_grid = 2 * self.gripper_radius / float(resolution_pc)

        output_pc = np.zeros([3, resolution_pc, resolution_pc])
        '''     ( y, x)
                ( 2,-2) ( 2,-1) ( 2, 0) ( 2, 1) ( 2, 2)
                ( 1,-2) ( 1,-1)...
                ( 0,-2)...
                (-1,-2)
                (-2,-2)...                      (-2,-2)
        '''
        for y in range(0, resolution_pc):
            output_pc[1, y, :] = self.gripper_radius - 0.5 * size_grid - size_grid * float(y)
        for x in range(0, resolution_pc):
            output_pc[0, :, x] = -1. * (self.gripper_radius - 0.5 * size_grid) + size_grid * float(x)

        org_pc_pcl = pcl.PointCloud(org_pc.astype(np.float32)[:, 0:3])

        win_project = org_pc_pcl.make_cropbox()
        width_grid = 0.499999 * size_grid

        win_project.set_MinMax(-width_grid, -width_grid, -1.2*self.gripper_radius, 1.0,
                               width_grid, width_grid, 1.2*self.gripper_radius, 1.0)
        '''
        win_project.set_MinMax(-width_grid, -width_grid, -0.01, 1.0,
                               width_grid, width_grid, 0.01, 1.0)
        '''
        cnt_null = 0
        for y in range(0, output_pc.shape[1]):
            for x in range(0, output_pc.shape[2]):

                win_project.set_Translation(output_pc[0, y, x],
                                            output_pc[1, y, x],
                                            0.0)

                cloud_win = win_project.filter()  # very important   not win_project.Filtering

                '''
                obj_tree = cloud_win.make_kdtree()
                eu_cluster = cloud_win.make_EuclideanClusterExtraction()
                eu_cluster.set_ClusterTolerance(width_grid)
                eu_cluster.set_MinClusterSize(1)
                eu_cluster.set_MaxClusterSize(6000)
                eu_cluster.set_SearchMethod(obj_tree)
                cluster_indices = eu_cluster.Extract()
                if len(cluster_indices) > 1:
                    max_z = 1.0
                    for j, indices in enumerate(cluster_indices):
                        cloud_cluster = cloud_win.extract(indices, negative=False)
                        tmp_array = np.asarray(cloud_cluster)
                        height_cluster = np.mean(tmp_array,axis=0)[2]
                        if height_cluster < max_z:
                            max_z = 1.0 * height_cluster
                            cloud_win = cloud_cluster
                '''
                # print(cloud_win.width, cloud_win.size)
                if cloud_win.size < thresh_pt_subgrid:
                    # near to be the boundary of point cloud, some subgrid is almost empty
                    # output_pc[2, y, x] = max_dis_project + np.random.random() * self.gripper_radius
                    output_pc[2, y, x] = 4.0 * self.gripper_radius
                    cnt_null += 1
                    # print("pos: ", output_pc[0, y, x], output_pc[1, y, x])
                    if cnt_null > int(0.9 * (float(resolution_pc) ** 2)):
                        print("{} number of null points are detected".format(cnt_null))
                        return False, np.zeros([1, 3], dtype=np.float32)
                    continue
                tmp_array = (np.asarray(cloud_win)).astype(np.float64)
                # select the point, which is the nearest one with gripper
                # print(tmp_array.shape)
                # tmp_array = tmp_array[tmp_array[:, 2].argsort(), :]
                # output_pc[2, y, x] = tmp_array[0, 2]
                tmp_z = np.mean(tmp_array, axis=0)[2]
                output_pc[2, y, x] = tmp_z

        output_pc = self.scale_obj * output_pc
        rate_filling = 1.0 - float(cnt_null) / float(resolution_pc ** 2)
        output_pc[1, 0, 0] = 1.0 * rate_filling
        # for PyTorch, only float32 is avaliable for GPU acceleration
        return True, output_pc.astype(np.float32)


    def generate_pc_PCL_vis(self, graspable,
                        mesh_gripper=None,
                        resolution=24,
                        flg_disturb_bd=False, noise_bd=0.005,
                        flg_random_rotZ=False):
        """
        FUNCTION: generate point cloud for a list of grasp candidate based on Point Cloud Libaray
        :param graspable:            obj:        `GraspableObject3D` the object to grasp
        :param resolution:           int:        resolution of point cloud
        :param flg_disturb_bd:       bool:       add disturbance before desampling
        :param noise_bd:             float:      noise level (metirc: m)
        :param flg_random_rotZ:      bool:       if rotate the point cloud along with the Z-axis to improve the robustness
        :param mesh_gripper:        mesh obj or None:      check mesh.py
        :return:
        """
        '''
        print("####################################")
        print(graspable.mesh.num_vertices)
        tmp = graspable.mesh.normals
        print(graspable.mesh.normals)
        print("####################################")
        '''
        ax_z = np.array([0., 0., 1.])

        g_info = GraspInfo()
        c1 = Contact3D(graspable, g_info.pos_grasp, in_direction=None)

        '''For test'''
        # 002_master_can
        # g_info.pos_grasp = np.array([-0.045040, 0.030736, 0.063758]) # q = 0.94
        # g_info.pos_grasp = np.array([-0.06216502, 0.01049824, 0.10267749]) # q = 0.7770
        # g_info.pos_grasp = np.array([0.02034432, 0.01516858, 0.00304357]) # q = 0.6297
        # g_info.pos_grasp = np.array([0.01878754, -0.04554584, 0.10734783]) # q = 0.3784
        # g_info.pos_grasp = np.array([-0.04192688, -0.05332974, 0.00148679])  # q = 0.6651

        # 011_banana 
        g_info.pos_grasp = np.array([-0.01628514, 0.042222, 0.00409378]) # q = 0.6691
        c1 = Contact3D(graspable, g_info.pos_grasp, in_direction=None)
        # dir is towards to object (face inward)
        ''''''
        dir, t1, t2 = c1.tangents()
        g_info.t1_grasp = 1.0 * t1
        g_info.t2_grasp = 1.0 * t2
        g_info.dir_grasp = 1.0 * dir

        '''
        g_info.dir_grasp = np.array([-0.87676113,  0.31158753, -0.36633745])# q = 0.3784
        g_info.t1_grasp = np.array([0.38914601,  0.90722539, -0.1597106])
        g_info.t2_grasp = np.array([0.2825868,  -0.2825868,  -0.91667301])
        '''
        '''
        g_info.dir_grasp = np.array([0.56594919, 0.666413,   0.48538152]) # q = 0.6651
        g_info.t1_grasp = np.array([0.79813762, -0.59039949, -0.1200199])
        g_info.t2_grasp = np.array([0.20658619,  0.45532642, -0.8660254])
        '''
        '''
        g_info.dir_grasp = np.array([0.86559391, -0.38507159,  0.32010476]) # q = 0.7770
        g_info.t1_grasp = np.array([0.28056871, 0.90244506, 0.32691606])
        g_info.t2_grasp = np.array([-0.41476305, -0.19316517,  0.88918999])
        '''
        '''
        g_info.dir_grasp = np.array([0.86559391, -0.38507159,  0.32010476]) # q = 0.7770
        g_info.t1_grasp = np.array([0.28056871, 0.90244506, 0.32691606])
        g_info.t2_grasp = np.array([-0.41476305, -0.19316517,  0.88918999])
        '''
        ''''''
        # banana
        g_info.dir_grasp = np.array([0.66853772, -0.32575243, 0.66853772]) # q = 0.64
        g_info.t1_grasp = np.array([0.74367785, 0.29192584, -0.60143375])
        g_info.t2_grasp = np.array([0.55067649e-04, 8.99257838e-01, 4.37418302e-01])


        """ Test end"""
        #####################

        _, surface_obj_GCS, _, _ = \
            DexterousVacuumPoint.crop_surface_grasp(contact_pt=c1,
                                                    direction=g_info.dir_grasp,
                                                    u1=g_info.t1_grasp,
                                                    u2=g_info.t2_grasp,
                                                    width_win=2 * self.gripper_radius,
                                                    depth_win=self.gripper_max_depth,
                                                    flg_exclude_opposite=True)

        surface_obj_GCS = surface_obj_GCS[:, 0:3]
        # rotate
        if flg_random_rotZ:
            rotate_z = 2 * np.pi * np.random.random()
            surface_grasp_GCS = math_robot.rot_along_axis(ax_z, rotate_z, surface_obj_GCS)
        else:
            surface_grasp_GCS = surface_obj_GCS

        flg_success, final_pc = \
            self.desampling_pc(org_pc=surface_grasp_GCS,
                               resolution_pc=resolution,
                               scale_obj=1.0,
                               thresh_pt_subgrid=1,
                               flg_disturb_bd=flg_disturb_bd, noise_bd=noise_bd)
        # re-oder to top view

        min_z = np.min(final_pc[2, :, :])
        final_pc[2, :, :] = final_pc[2, :, :] - min_z
        '''
        max_z = np.max(final_pc[2, :, :])
        final_pc[2, :, :] = max_z - final_pc[2, :, :]
        '''

        gl_vis = GL_Visualizer()
        '''draw x, y, z axis
        ax_width = 5.0
        x_ax = np.array([.0, .0, .0, 1.0, .0, .0])
        gl_vis.draw_lines(x_ax, width=ax_width, num_color=1)
        y_ax = np.array([.0, .0, .0, .0, 1.0, .0])
        gl_vis.draw_lines(y_ax, width=ax_width, num_color=2)
        z_ax = np.array([.0, .0, .0, .0, .0, 1.0])
        gl_vis.draw_lines(z_ax, width=ax_width, num_color=4)
        '''

        gl_vis.display_mesh(100.0 * graspable.mesh.vertices, graspable.mesh.triangles,
                            graspable.mesh.normals)

        # gl_vis.draw_spheres(100.0 * g_info.pos_grasp, radius=0.4, num_color=1)
        '''
        offset_gripper = np.array([0., 0., -5.0]).reshape([1, 3])   # cm
        gripper_vertices = 0.1 * mesh_gripper.vertices + offset_gripper
        gripper_vertices = math_robot.transfer_CS_reverse(g_info.t1_grasp,
                                                  g_info.t2_grasp,
                                                  g_info.dir_grasp,
                                                  100.0*g_info.pos_grasp,
                                                  gripper_vertices)[:, 0:3]

        gripper_normals = mesh_gripper.normals
        gripper_normals = math_robot.transfer_CS_reverse(g_info.t1_grasp,
                                                  g_info.t2_grasp,
                                                  g_info.dir_grasp,
                                                  100.0*g_info.pos_grasp,
                                                  gripper_normals)[:, 0:3]
        gripper_normals = gripper_normals / np.linalg.norm(gripper_normals, axis=1).\
            reshape([gripper_normals.shape[0], 1])
        
        gl_vis.display_mesh(vertices=gripper_vertices, triangles=mesh_gripper.triangles,
                            v_normals=gripper_normals)
        '''

        tmp_pc = np.zeros([final_pc.shape[1] * final_pc.shape[2], 3])
        for y in range(final_pc.shape[1]):
            for x in range(final_pc.shape[2]):
                tmp_pc[final_pc.shape[1] * y + x, :] = np.array([final_pc[0, y, x],
                                                                 final_pc[1, y, x],
                                                                 final_pc[2, y, x]])
        # tmp_pc = tmp_pc + -0.005*np.array([0.0, 0.0, 1.0]).reshape([1, 3])
        tmp_pc[:, 2] = tmp_pc[:, 2] + min_z - 0.000
        tmp_pc = math_robot.transfer_CS_reverse(g_info.t1_grasp,
                                                  g_info.t2_grasp,
                                                  g_info.dir_grasp,
                                                  g_info.pos_grasp,
                                                  tmp_pc)[:, 0:3]
        gl_vis.draw_spheres(100.0*tmp_pc, radius=0.1, num_color=7)
        np.save('test_pc.npy', final_pc)

        print("visualization completed!")
        gl_vis.UI_lisener()

        str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        if flg_success:
            logger.info("{}: Point cloud generation -> success.".format(str_time))
        else:
            logger.info("{}: Point cloud generation -> failed.".format(str_time))
        return True


    def generate_pc_PCL(self, graspable, lst_grasp,
                        path_save='',
                        vis=False,
                        mesh_gripper=None,
                        resolution=24,
                        flg_disturb_bd=False, noise_bd=0.005,
                        flg_random_rotZ=False):
        """
        FUNCTION: generate point cloud for a list of grasp candidate based on Point Cloud Libaray
        :param graspable:            obj:        `GraspableObject3D` the object to grasp
        :param lst_grasp:            list:       list of grasp candidate
        :param path_save:            str:        path to save point cloud
        :param resolution:           int:        resolution of point cloud
        :param flg_disturb_bd:       bool:       add disturbance before desampling
        :param noise_bd:             float:      noise level (metirc: m)
        :param flg_random_rotZ:      bool:       if rotate the point cloud along with the Z-axis to improve the robustness
        :param vis:                 bool:
        :param mesh_gripper:        mesh obj or None:      check mesh.py
        :return:
        """
        '''
        print("####################################")
        print(graspable.mesh.num_vertices)
        tmp = graspable.mesh.normals
        print(graspable.mesh.normals)
        print("####################################")
        '''
        ax_z = np.array([0., 0., 1.])
        for cnt, g_info in enumerate(lst_grasp):
            self.set_scale_obj(g_info.scale_obj)
            c1 = Contact3D(graspable, g_info.pos_grasp, in_direction=None)

            '''For test
            # 002_master_can
            # g_info.pos_grasp = np.array([-0.045040, 0.030736, 0.063758]) # q = 0.94
            # g_info.pos_grasp = np.array([-0.06216502, 0.01049824, 0.10267749]) # q = 0.7770
            # g_info.pos_grasp = np.array([0.02034432, 0.01516858, 0.00304357]) # q = 0.6297
            g_info.pos_grasp = np.array([0.01878754, -0.04554584, 0.10734783]) # q = 0.3784
            # g_info.pos_grasp = np.array([-0.04192688, -0.05332974, 0.00148679])  # q = 0.6651
            # 011_banana 
            # g_info.pos_grasp = np.array([-0.00637234, -0.05888856, 0.01202402]) # q = 0.6691

            c1 = Contact3D(graspable, g_info.pos_grasp, in_direction=None)
            # dir is towards to object (face inward)
            ''''''
            dir, t1, t2 = c1.tangents()
            g_info.t1_grasp = 1.0 * t1
            g_info.t2_grasp = 1.0 * t2
            g_info.dir_grasp = 1.0 * dir
            '''

            '''
            g_info.dir_grasp = np.array([-0.87676113,  0.31158753, -0.36633745])# q = 0.3784
            g_info.t1_grasp = np.array([0.38914601,  0.90722539, -0.1597106])
            g_info.t2_grasp = np.array([0.2825868,  -0.2825868,  -0.91667301])
            '''
            '''
            g_info.dir_grasp = np.array([0.56594919, 0.666413,   0.48538152]) # q = 0.6651
            g_info.t1_grasp = np.array([0.79813762, -0.59039949, -0.1200199])
            g_info.t2_grasp = np.array([0.20658619,  0.45532642, -0.8660254])
            '''
            '''
            g_info.dir_grasp = np.array([0.86559391, -0.38507159,  0.32010476]) # q = 0.7770
            g_info.t1_grasp = np.array([0.28056871, 0.90244506, 0.32691606])
            g_info.t2_grasp = np.array([-0.41476305, -0.19316517,  0.88918999])
            '''
            '''
            g_info.dir_grasp = np.array([0.86559391, -0.38507159,  0.32010476]) # q = 0.7770
            g_info.t1_grasp = np.array([0.28056871, 0.90244506, 0.32691606])
            g_info.t2_grasp = np.array([-0.41476305, -0.19316517,  0.88918999])
            '''

            """ Test end"""
            #####################

            _, surface_obj_GCS, _, _ = \
                DexterousVacuumPoint.crop_surface_grasp(contact_pt=c1,
                                                        direction=g_info.dir_grasp,
                                                        u1=g_info.t1_grasp,
                                                        u2=g_info.t2_grasp,
                                                        width_win=2 * self.gripper_radius,
                                                        depth_win=self.gripper_max_depth,
                                                        flg_exclude_opposite=True)

            surface_obj_GCS = surface_obj_GCS[:, 0:3]
            # rotate
            if flg_random_rotZ:
                rotate_z = 2 * np.pi * np.random.random()
                surface_grasp_GCS = math_robot.rot_along_axis(ax_z, rotate_z, surface_obj_GCS)
            else:
                surface_grasp_GCS = surface_obj_GCS
            '''
            flg_rot_success, surface_grasp_GCS = \
                self.rot_pt_xy_axes(org_pc=surface_obj_GCS,
                                    rotate_x=g_info.rot_x, rotate_y=g_info.rot_y)
            if not flg_rot_success:
                logging.debug('Point cloud generation failed.')
                return False
                surface_grasp_GCS = surface_grasp_GCS[0:3, :].T
            '''

            flg_success, final_pc = \
                self.desampling_pc(org_pc=surface_grasp_GCS,
                                   resolution_pc=resolution,
                                   scale_obj=self.scale_obj,
                                   thresh_pt_subgrid=1,
                                   flg_disturb_bd=flg_disturb_bd, noise_bd=noise_bd)
            # re-oder to top view

            min_z = np.min(final_pc[2, :, :])
            final_pc[2, :, :] = final_pc[2, :, :] - min_z
            '''
            max_z = np.max(final_pc[2, :, :])
            final_pc[2, :, :] = max_z - final_pc[2, :, :]
            '''

            if vis:
                gl_vis = GL_Visualizer()
                '''draw x, y, z axis
                ax_width = 5.0
                x_ax = np.array([.0, .0, .0, 1.0, .0, .0])
                gl_vis.draw_lines(x_ax, width=ax_width, num_color=1)
                y_ax = np.array([.0, .0, .0, .0, 1.0, .0])
                gl_vis.draw_lines(y_ax, width=ax_width, num_color=2)
                z_ax = np.array([.0, .0, .0, .0, .0, 1.0])
                gl_vis.draw_lines(z_ax, width=ax_width, num_color=4)
                '''

                gl_vis.display_mesh(100.0 * graspable.mesh.vertices, graspable.mesh.triangles,
                                    graspable.mesh.normals)

                gl_vis.draw_spheres(100.0 * g_info.pos_grasp, radius=0.4, num_color=1)

                offset_gripper = np.array([0., 0., -5.0]).reshape([1, 3])   # cm
                gripper_vertices = 0.1 * mesh_gripper.vertices + offset_gripper
                gripper_vertices = math_robot.transfer_CS_reverse(g_info.t1_grasp,
                                                          g_info.t2_grasp,
                                                          g_info.dir_grasp,
                                                          100.0*g_info.pos_grasp,
                                                          gripper_vertices)[:, 0:3]

                gripper_normals = mesh_gripper.normals
                gripper_normals = math_robot.transfer_CS_reverse(g_info.t1_grasp,
                                                          g_info.t2_grasp,
                                                          g_info.dir_grasp,
                                                          100.0*g_info.pos_grasp,
                                                          gripper_normals)[:, 0:3]
                gripper_normals = gripper_normals / np.linalg.norm(gripper_normals, axis=1).\
                    reshape([gripper_normals.shape[0], 1])

                gl_vis.display_mesh(vertices=gripper_vertices, triangles=mesh_gripper.triangles,
                                    v_normals=gripper_normals)

                tmp_pc = np.zeros([final_pc.shape[1] * final_pc.shape[2], 3])
                for y in range(final_pc.shape[1]):
                    for x in range(final_pc.shape[2]):
                        tmp_pc[final_pc.shape[1] * y + x, :] = np.array([final_pc[0, y, x],
                                                                         final_pc[1, y, x],
                                                                         final_pc[2, y, x]])
                # tmp_pc = tmp_pc + -0.005*np.array([0.0, 0.0, 1.0]).reshape([1, 3])
                tmp_pc[:, 2] = tmp_pc[:, 2] + min_z - 0.002
                tmp_pc = math_robot.transfer_CS_reverse(g_info.t1_grasp,
                                                          g_info.t2_grasp,
                                                          g_info.dir_grasp,
                                                          g_info.pos_grasp,
                                                          tmp_pc)[:, 0:3]
                gl_vis.draw_spheres(100.0*tmp_pc, radius=0.05, num_color=7)
                np.save('test_pc.npy', final_pc)

                print("visualization completed!")
                gl_vis.UI_lisener()

            str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            if flg_success:
                np.save(path_save + g_info.get_name_grasp() + '.npy', final_pc)
                logger.info("{}: Point cloud generation -> success.".format(str_time))
            else:
                logger.info("{}: Point cloud generation -> failed.".format(str_time))
        return True


    def generate_pc_SDF(self, graspable, lst_grasp,
                        path_save='',
                        vis=False,
                        mesh_gripper=None,
                        resolution=24,
                        flg_disturb_bd=False, noise_bd=0.001):
        """
        FUNCTION: generate point cloud for a list of grasp candidate
        :param graspable:            obj:        `GraspableObject3D` the object to grasp
        :param lst_grasp:            list:       list of grasp candidate
        :param path_save:            str:        path to save point cloud
        :param resolution:           int:        resolution of point cloud
        :param flg_disturb_bd:       bool:       add disturbance before desampling
        :param noise_bd:             float:      noise level (metirc: m)
        :param vis:                 bool:
        :param mesh_gripper:        mesh obj or None:      check mesh.py
        :return:
        """
        '''
        print("####################################")
        print(graspable.mesh.num_vertices)
        tmp = graspable.mesh.normals
        print(graspable.mesh.normals)
        print("####################################")
        '''
        max_dis_project = 3.0 * self.gripper_radius
        size_grid = 2 * self.gripper_radius / float(resolution)
        resolustion_z = 0.001
        step_z = int(1.2 * 2.0 * self.gripper_radius / resolustion_z) + 1
        for cnt, g_info in enumerate(lst_grasp):
            self.set_scale_obj(g_info.scale_obj)

            c1 = Contact3D(graspable, g_info.pos_grasp, in_direction=None)
            output_pc = np.zeros([3, resolution, resolution])
            for y in range(0, resolution):
                output_pc[1, y, :] = self.gripper_radius - 0.5 * size_grid - size_grid * float(y)
            for x in range(0, resolution):
                output_pc[0, :, x] = -1. * (self.gripper_radius - 0.5 * size_grid) + size_grid * float(x)

            u1 = 1.0 * g_info.t1_grasp
            u2 = 1.0* g_info.t2_grasp
            direction = 1.0 * g_info.dir_grasp

            for y in range(0, resolution):
                for x in range(0, resolution):
                    pts_new_GCS = np.ones([step_z, 4], dtype=np.float64)
                    x_pts_base = 1.0 * output_pc[0, y, x] * np.ones(step_z, dtype=np.float64)
                    y_pts_base = 1.0 * output_pc[1, y, x] * np.ones(step_z, dtype=np.float64)
                    z_pts_base = np.linspace(-0.5*self.gripper_radius*1.2,
                                             0.5*self.gripper_radius*1.2,
                                             step_z)
                    pts_new_GCS[:, 0] = x_pts_base[:]
                    pts_new_GCS[:, 1] = y_pts_base[:]
                    pts_new_GCS[:, 2] = z_pts_base[:]

                    pts_new_obj = math_robot.transfer_CS_reverse(u1, u2, direction, 1.0 * c1.point, pts_new_GCS)
                    # pt_tmp = math_robot.transfer_CS(direction, u1, u2, 1.0 * contact_pt.point, pts_new_obj)
                    pts_new_grid = graspable.sdf.transform_pt_obj_to_grid(pts_new_obj[:, 0:3].T)
                    dists = graspable.sdf.get_signed_distances(pts_new_grid)
                    dists = abs(dists)
                    flg_contact = False
                    for i in range(1, dists.shape[0]):
                        if dists[i] > dists[i-1]:
                            if dists[i-1] < graspable.sdf.surface_thresh * 1.5:
                                break
                    if flg_contact:
                        output_pc[2, y, x] = 1.0 * pts_new_GCS[i-1, 2]
                    else:
                        output_pc[2, y, x] = 999.0

            final_pc = output_pc
            # re-oder to top view
            min_z = np.min(final_pc[2, :, :])
            final_pc[2, :, :] = final_pc[2, :, :] - min_z
            ind_max_z = np.argwhere(final_pc[2, :, :] > 100.0)
            if ind_max_z.shape[0] > 0:
                final_pc[2, ind_max_z[:, 0], ind_max_z[:, 1]] = 1.0 * max_dis_project

            if vis:
                gl_vis = GL_Visualizer()
                ax_width = 5.0
                x_ax = np.array([.0, .0, .0, 10.0, .0, .0])
                gl_vis.draw_lines(x_ax, width=ax_width, num_color=1)
                y_ax = np.array([.0, .0, .0, .0, 10.0, .0])
                gl_vis.draw_lines(y_ax, width=ax_width, num_color=2)
                z_ax = np.array([.0, .0, .0, .0, .0, 10.0])
                gl_vis.draw_lines(z_ax, width=ax_width, num_color=4)
                '''
                gl_vis.display_mesh(0.1 * mesh_gripper.vertices, mesh_gripper.triangles,
                                    mesh_gripper.normals)
                '''
                gl_vis.display_mesh(100.0 * graspable.mesh.vertices, graspable.mesh.triangles,
                                    graspable.mesh.normals)

                gl_vis.draw_spheres(100.0 * g_info.pos_grasp, radius=0.4, num_color=1)

                offset_gripper = np.array([0., 0., -5.0]).reshape([1, 3])   # cm
                gripper_vertices = 0.1 * mesh_gripper.vertices + offset_gripper
                gripper_vertices = math_robot.transfer_CS_reverse(g_info.t1_grasp,
                                                          g_info.t2_grasp,
                                                          g_info.dir_grasp,
                                                          100.0*g_info.pos_grasp,
                                                          gripper_vertices)[:, 0:3]

                gripper_normals = mesh_gripper.normals
                gripper_normals = math_robot.transfer_CS_reverse(g_info.t1_grasp,
                                                          g_info.t2_grasp,
                                                          g_info.dir_grasp,
                                                          100.0*g_info.pos_grasp,
                                                          gripper_normals)[:, 0:3]
                gripper_normals = gripper_normals / np.linalg.norm(gripper_normals, axis=1).\
                    reshape([gripper_normals.shape[0], 1])

                gl_vis.display_mesh(vertices=gripper_vertices, triangles=mesh_gripper.triangles,
                                    v_normals=gripper_normals)

                tmp_pc = np.zeros([final_pc.shape[1] * final_pc.shape[2], 3])
                for y in range(final_pc.shape[1]):
                    for x in range(final_pc.shape[2]):
                        tmp_pc[final_pc.shape[1] * y + x, :] = np.array([final_pc[0, y, x],
                                                                         final_pc[1, y, x],
                                                                         final_pc[2, y, x]])
                tmp_pc = tmp_pc + -0.005*np.array([0.0, 0.0, 1.0]).reshape([1, 3])
                tmp_pc = math_robot.transfer_CS_reverse(g_info.t1_grasp,
                                                          g_info.t2_grasp,
                                                          g_info.dir_grasp,
                                                          g_info.pos_grasp,
                                                          tmp_pc)[:, 0:3]
                gl_vis.draw_spheres(100.0*tmp_pc, radius=0.1, num_color=7)


                print("visualization completed!")
                gl_vis.UI_lisener()

            str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

            np.save(path_save + g_info.get_name_grasp() + '.npy', final_pc)
            logger.info("{}: Point cloud generation -> success.".format(str_time))
        return True

    def generate_pc_BVHTree(self, graspable, lst_grasp,
                        path_save='',
                        vis=False,
                        resolution=24,
                        flg_disturb_bd=False, noise_bd=0.005,
                        flg_random_rotZ=False):
        """
        FUNCTION: generate point cloud for a list of grasp candidate based on Point Cloud Libaray
        :param graspable:            obj:        `GraspableObject3D` the object to grasp
        :param lst_grasp:            list:       list of grasp candidate
        :param path_save:            str:        path to save point cloud
        :param resolution:           int:        resolution of point cloud
        :param flg_disturb_bd:       bool:       add disturbance before desampling
        :param noise_bd:             float:      noise level (metirc: m)
        :param flg_random_rotZ:      bool:       if rotate the point cloud along with the Z-axis to improve the robustness
        :return:
        """
        '''
        print("####################################")
        print(graspable.mesh.num_vertices)
        tmp = graspable.mesh.normals
        print(graspable.mesh.normals)
        print("####################################")
        '''
        ax_z = np.array([0., 0., 1.])
        pass

    def generate_pc_mesh_trian_cross(self, graspable, lst_grasp,
                                     path_save='',
                                     vis=False,
                                     resolution=24,
                                     flg_disturb_bd=False, noise_bd=0.005):
        """
        FUNCTION: generate point cloud for a list of grasp candidate
        Note: This method is based on cross intersection of mesh's triangles
        :param graspable:
        :param lst_grasp:
        :param path_save:
        :param vis:
        :param resolution:
        :param flg_disturb_bd:
        :param noise_bd:
        :return:
        """
        pass

######################################################################################
######################################################################################
######################################################################################
'''
    waiting for being completed  
    coded by Hui Zhang (hui.zhang@kuleuven.be  07.10.2021)
'''
class ChameleonTongueGrasp(GraspSampler):
    def __init__(self, gripper, config, graspable=None):
        pass

