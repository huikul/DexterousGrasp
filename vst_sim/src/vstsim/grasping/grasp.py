# -*- coding: utf-8 -*-
# """
# Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational,
# research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
# hereby granted, provided that the above copyright notice, this paragraph and the following two
# paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
# Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
# 7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
# THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
# HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# """
# """
# Grasp class that implements gripper endpoints and grasp functions
# Authors: Jeff Mahler, with contributions from Jacky Liang and Nikhil Sharma
# """
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, norm
import time
import itertools as it
import math
from mpl_toolkits.mplot3d import Axes3D
################################################
from vstsim.grasping import math_robot
################################################
from autolab_core import Point, RigidTransform
from meshpy import Sdf3D, StablePose
from skimage.restoration import denoise_bilateral

from vstsim import abstractstatic
from vstsim.grasping import Contact3D, GraspableObject3D
from vstsim.constants import NO_CONTACT_DIST

import cvxopt as cvx
from scipy.optimize import minimize
import copy
# turn off output logging
cvx.solvers.options['show_progress'] = False

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# class Grasp(object, metaclass=ABCMeta):
class Grasp(object):
    """ Abstract grasp class.

    Attributes
    ----------
    configuration : :obj:`numpy.ndarray`
        vector specifying the parameters of the grasp (e.g. hand pose, opening width, joint angles, etc)
    frame : :obj:`str`
        string name of grasp reference frame (defaults to obj)
    """
    __metaclass__ = ABCMeta
    samples_per_grid = 2  # global resolution for line of action

    @abstractmethod
    def close_fingers(self, obj):
        """ Finds the contact points by closing on the given object.
        
        Parameters
        ----------
        obj : :obj:`GraspableObject3D`
            object to find contacts on
        """
        pass

    @abstractmethod
    def configuration(self):
        """ Returns the numpy array representing the hand configuration """
        pass

    @abstractmethod
    def frame(self):
        """ Returns the string name of the grasp reference frame  """
        pass

    @abstractstatic
    def params_from_configuration(configuration):
        """ Convert configuration vector to a set of params for the class """
        pass

    @abstractstatic
    def configuration_from_params(*params):
        """ Convert param list to a configuration vector for the class """
        pass


# class PointGrasp(Grasp, metaclass=ABCMeta):
class PointGrasp(Grasp):
    """ Abstract grasp class for grasps with a point contact model.

    Attributes
    ----------
    configuration : :obj:`numpy.ndarray`
        vector specifying the parameters of the grasp (e.g. hand pose, opening width, joint angles, etc)
    frame : :obj:`str`
        string name of grasp reference frame (defaults to obj)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_line_of_action(g, axis, width, obj, num_samples):
        """ Creates a line of action, or the points in space that the grasp traces out, from a point g in world coordinates on an object.

        Returns
        -------
        bool
            whether or not successful
        :obj:`list` of :obj:`numpy.ndarray`
            points in 3D space along the line of action
        """
        pass

    # NOTE: implementation of close_fingers must return success, array of contacts (one per column)


class ParallelJawPtGrasp3D(PointGrasp):
    """ Parallel Jaw point grasps in 3D space.
    """

    def __init__(self, configuration, frame='object', grasp_id=None):
        # get parameters from configuration array
        grasp_center, grasp_axis, grasp_width, grasp_angle, jaw_width, min_grasp_width = \
            ParallelJawPtGrasp3D.params_from_configuration(configuration)

        self.center_ = grasp_center
        self.axis_ = grasp_axis / np.linalg.norm(grasp_axis)
        self.max_grasp_width_ = grasp_width
        self.jaw_width_ = jaw_width
        self.min_grasp_width_ = min_grasp_width
        self.approach_angle_ = grasp_angle
        self.frame_ = frame
        self.grasp_id_ = grasp_id

    @property
    def center(self):
        """ :obj:`numpy.ndarray` : 3-vector specifying the center of the jaws """
        return self.center_

    @center.setter
    def center(self, x):
        self.center_ = x

    @property
    def axis(self):
        """ :obj:`numpy.ndarray` : normalized 3-vector specifying the line between the jaws """
        return self.axis_

    @property
    def open_width(self):
        """ float : maximum opening width of the jaws """
        return self.max_grasp_width_

    @property
    def close_width(self):
        """ float : minimum opening width of the jaws """
        return self.min_grasp_width_

    @property
    def jaw_width(self):
        """ float : width of the jaws in the tangent plane to the grasp axis """
        return self.jaw_width_

    @property
    def approach_angle(self):
        """ float : approach angle of the grasp """
        return self.approach_angle_

    @property
    def configuration(self):
        """ :obj:`numpy.ndarray` : vector specifying the parameters of the grasp as follows
        (grasp_center, grasp_axis, grasp_angle, grasp_width, jaw_width) """
        return ParallelJawPtGrasp3D.configuration_from_params(self.center_, self.axis_, self.max_grasp_width_,
                                                              self.approach_angle_, self.jaw_width_,
                                                              self.min_grasp_width_)

    @property
    def frame(self):
        """ :obj:`str` : name of grasp reference frame """
        return self.frame_

    @property
    def id(self):
        """ int : id of grasp """
        return self.grasp_id_

    @frame.setter
    def frame(self, f):
        self.frame_ = f

    @approach_angle.setter
    def approach_angle(self, angle):
        """ Set the grasp approach angle """
        self.approach_angle_ = angle

    @property
    def endpoints(self):
        """
        Returns
        -------
        :obj:`numpy.ndarray`
            location of jaws in 3D space at max opening width """
        return self.center_ - (self.max_grasp_width_ / 2.0) * self.axis_, self.center_ + (
                self.max_grasp_width_ / 2.0) * self.axis_,

    @staticmethod
    def distance(g1, g2, alpha=0.05):
        """ Evaluates the distance between two grasps.

        Parameters
        ----------
        g1 : :obj:`ParallelJawPtGrasp3D`
            the first grasp to use
        g2 : :obj:`ParallelJawPtGrasp3D`
            the second grasp to use
        alpha : float
            parameter weighting rotational versus spatial distance

        Returns
        -------
        float
            distance between grasps g1 and g2
        """
        center_dist = np.linalg.norm(g1.center - g2.center)
        axis_dist = (2.0 / np.pi) * np.arccos(np.abs(g1.axis.dot(g2.axis)))
        return center_dist + alpha * axis_dist

    @staticmethod
    def configuration_from_params(center, axis, width, angle=0, jaw_width=0, min_width=0):
        """ Converts grasp parameters to a configuration vector. """
        if np.abs(np.linalg.norm(axis) - 1.0) > 1e-5:
            raise ValueError('Illegal grasp axis. Must be norm one')
        configuration = np.zeros(10)
        configuration[0:3] = center
        configuration[3:6] = axis
        configuration[6] = width
        configuration[7] = angle
        configuration[8] = jaw_width
        configuration[9] = min_width
        return configuration

    @staticmethod
    def params_from_configuration(configuration):
        """ Converts configuration vector into grasp parameters.
        
        Returns
        -------
        grasp_center : :obj:`numpy.ndarray`
            center of grasp in 3D space
        grasp_axis : :obj:`numpy.ndarray`
            normalized axis of grasp in 3D space
        max_width : float
            maximum opening width of jaws
        angle : float
            approach angle
        jaw_width : float
            width of jaws
        min_width : float
            minimum closing width of jaws
        """
        if not isinstance(configuration, np.ndarray) or (configuration.shape[0] != 9 and configuration.shape[0] != 10):
            raise ValueError('Configuration must be numpy ndarray of size 9 or 10')
        if configuration.shape[0] == 9:
            min_grasp_width = 0
        else:
            min_grasp_width = configuration[9]
        if np.abs(np.linalg.norm(configuration[3:6]) - 1.0) > 1e-5:
            raise ValueError('Illegal grasp axis. Must be norm one')
        return configuration[0:3], configuration[3:6], configuration[6], configuration[7], configuration[
            8], min_grasp_width

    @staticmethod
    def center_from_endpoints(g1, g2):
        """ Grasp center from endpoints as np 3-arrays """
        grasp_center = (g1 + g2) / 2
        return grasp_center

    @staticmethod
    def axis_from_endpoints(g1, g2):
        """ Normalized axis of grasp from endpoints as np 3-arrays """
        grasp_axis = g2 - g1
        if np.linalg.norm(grasp_axis) == 0:
            return grasp_axis
        return grasp_axis / np.linalg.norm(grasp_axis)

    @staticmethod
    def width_from_endpoints(g1, g2):
        """ Width of grasp from endpoints """
        grasp_axis = g2 - g1
        return np.linalg.norm(grasp_axis)

    @staticmethod
    def grasp_from_endpoints(g1, g2, width=None, approach_angle=0, close_width=0):
        """ Create a grasp from given endpoints in 3D space, making the axis the line between the points.

        Parameters
        ---------
        g1 : :obj:`numpy.ndarray`
            location of the first jaw
        g2 : :obj:`numpy.ndarray`
            location of the second jaw
        width : float
            maximum opening width of jaws
        approach_angle : float
            approach angle of grasp
        close_width : float
            width of gripper when fully closed
        """
        x = ParallelJawPtGrasp3D.center_from_endpoints(g1, g2)
        v = ParallelJawPtGrasp3D.axis_from_endpoints(g1, g2)
        if width is None:
            width = ParallelJawPtGrasp3D.width_from_endpoints(g1, g2)
        return ParallelJawPtGrasp3D(
            ParallelJawPtGrasp3D.configuration_from_params(x, v, width, min_width=close_width, angle=approach_angle))

    @property
    def unrotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. X axis points out of the
        gripper palm along the 0-degree approach direction, Y axis points between the jaws, and the Z axs is orthogonal.

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        grasp_axis_y = self.axis
        grasp_axis_x = np.array([grasp_axis_y[1], -grasp_axis_y[0], 0])
        if np.linalg.norm(grasp_axis_x) == 0:
            grasp_axis_x = np.array([1, 0, 0])
        grasp_axis_x = grasp_axis_x / norm(grasp_axis_x)
        grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)

        R = np.c_[grasp_axis_x, np.c_[grasp_axis_y, grasp_axis_z]]
        return R

    @property
    def rotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. X axis points out of the
        gripper palm along the grasp approach angle, Y axis points between the jaws, and the Z axs is orthogonal.

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        R = ParallelJawPtGrasp3D._get_rotation_matrix_y(self.approach_angle)
        R = self.unrotated_full_axis.dot(R)
        return R

    @property
    def T_grasp_obj(self):
        """ Rigid transformation from grasp frame to object frame.
        Rotation matrix is X-axis along approach direction, Y axis pointing between the jaws, and Z-axis orthogonal.
        Translation vector is the grasp center.

        Returns
        -------
        :obj:`RigidTransform`
            transformation from grasp to object coordinates
        """
        T_grasp_obj = RigidTransform(self.rotated_full_axis, self.center, from_frame='grasp', to_frame='obj')
        return T_grasp_obj

    @staticmethod
    def _get_rotation_matrix_y(theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.c_[[cos_t, 0, sin_t], np.c_[[0, 1, 0], [-sin_t, 0, cos_t]]]
        return R

    def gripper_pose(self, gripper=None):
        """ Returns the RigidTransformation from the gripper frame to the object frame when the gripper is executing the
        given grasp.
        Differs from the grasp reference frame because different robots use different conventions for the gripper
        reference frame.
        
        Parameters
        ----------
        gripper : :obj:`RobotGripper`
            gripper to get the pose for

        Returns
        -------
        :obj:`RigidTransform`
            transformation from gripper frame to object frame
        """
        if gripper is None:
            T_gripper_grasp = RigidTransform(from_frame='gripper', to_frame='grasp')
        else:
            T_gripper_grasp = gripper.T_grasp_gripper

        T_gripper_obj = self.T_grasp_obj * T_gripper_grasp
        return T_gripper_obj

    def grasp_angles_from_stp_z(self, stable_pose):
        """ Get angles of the the grasp from the table plane:
        1) the angle between the grasp axis and table normal
        2) the angle between the grasp approach axis and the table normal
        
        Parameters
        ----------
        stable_pose : :obj:`StablePose` or :obj:`RigidTransform`
            the stable pose to compute the angles for

        Returns
        -------
        psi : float
            grasp y axis rotation from z axis in stable pose
        phi : float
            grasp x axis rotation from z axis in stable pose
        """
        T_grasp_obj = self.T_grasp_obj

        if isinstance(stable_pose, StablePose):
            R_stp_obj = stable_pose.r
        else:
            R_stp_obj = stable_pose.rotation
        T_stp_obj = RigidTransform(R_stp_obj, from_frame='obj', to_frame='stp')

        T_stp_grasp = T_stp_obj * T_grasp_obj

        stp_z = np.array([0, 0, 1])
        grasp_axis_angle = np.arccos(stp_z.dot(T_stp_grasp.y_axis))
        grasp_approach_angle = np.arccos(abs(stp_z.dot(T_stp_grasp.x_axis)))
        nu = stp_z.dot(T_stp_grasp.z_axis)

        return grasp_axis_angle, grasp_approach_angle, nu

    def close_fingers(self, obj, vis=False, check_approach=True, approach_dist=1.0):
        """ Steps along grasp axis to find the locations of contact with an object

        Parameters
        ----------
        obj : :obj:`GraspableObject3D`
            object to close fingers on
        vis : bool
            whether or not to plot the line of action and contact points
        check_approach : bool
            whether or not to check if the contact points can be reached
        approach_dist : float
            how far back to check the approach distance, only if checking the approach is set
        
        Returns
        -------
        success : bool
            whether or not contacts were found
        c1 : :obj:`Contact3D`
            the contact point for jaw 1
        c2 : :obj:`Contact3D`
            the contact point for jaw 2
        """
        if vis:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()
        # compute num samples to use based on sdf resolution
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(self.max_grasp_width_)
        num_samples = int(Grasp.samples_per_grid * float(grasp_width_grid) / 2)  # at least 1 sample per grid

        # get grasp endpoints in sdf frame
        g1_world, g2_world = self.endpoints

        # check for contact along approach
        if check_approach:
            approach_dist_grid = obj.sdf.transform_pt_obj_to_grid(approach_dist)
            num_approach_samples = int(Grasp.samples_per_grid * approach_dist_grid / 2)  # at least 1 sample per grid
            approach_axis = self.rotated_full_axis[:, 0]
            approach_loa1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, -approach_axis, approach_dist, obj,
                                                                       num_approach_samples, min_width=0)
            approach_loa2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -approach_axis, approach_dist, obj,
                                                                       num_approach_samples, min_width=0)
            c1_found, _ = ParallelJawPtGrasp3D.find_contact(approach_loa1, obj, vis=vis, strict=True)
            c2_found, _ = ParallelJawPtGrasp3D.find_contact(approach_loa2, obj, vis=vis, strict=True)
            approach_collision = c1_found or c2_found
            if approach_collision:
                plt.clf()
                return False, None

        # get line of action            
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, self.axis_, self.open_width, obj,
                                                                     num_samples, min_width=self.close_width)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -self.axis_, self.open_width, obj,
                                                                     num_samples, min_width=self.close_width)

        if vis:
            ax = plt.gca(projection='3d')
            surface = obj.sdf.surface_points()[0]
            surface = surface[np.random.choice(surface.shape[0], 1000, replace=False)]
            ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], '.',
                       s=np.ones_like(surface[:, 0]) * 0.3, c='b')

        # find contacts
        c1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=vis)
        c2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=vis)

        if vis:
            ax = plt.gca(projection='3d')
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.draw()

        contacts_found = c1_found and c2_found
        return contacts_found, [c1, c2]

    def vis_grasp(self, obj, *args, **kwargs):
        if 'keep' not in kwargs or not kwargs['keep']:
            plt.clf()

        ax = plt.gca(projection='3d')
        if 'show_obj' in kwargs and kwargs['show_obj']:
            # plot the obj
            surface = obj.sdf.surface_points()[0]
            surface = surface[np.random.choice(surface.shape[0], 1000, replace=False)]
            ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], '.',
                       s=np.ones_like(surface[:, 0]) * 0.3, c='b')

        # plot the center of grasp using grid
        grasp_center_grid = obj.sdf.transform_pt_obj_to_grid(self.center)
        ax.scatter(grasp_center_grid[0], grasp_center_grid[1], grasp_center_grid[2], marker='x', c='r')

        # compute num samples to use based on sdf resolution
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(self.max_grasp_width_)
        num_samples = int(Grasp.samples_per_grid * float(grasp_width_grid) / 2)  # at least 1 sample per grid

        # get grasp endpoints in sdf frame
        g1_world, g2_world = self.endpoints

        # check for contact along approach
        approach_dist = 0.1
        approach_dist_grid = obj.sdf.transform_pt_obj_to_grid(approach_dist)
        num_approach_samples = int(approach_dist_grid / 2)  # at least 1 sample per grid
        approach_axis = self.rotated_full_axis[:, 0]
        approach_loa1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, -approach_axis, approach_dist, obj,
                                                                   num_approach_samples, min_width=0)
        approach_loa2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -approach_axis, approach_dist, obj,
                                                                   num_approach_samples, min_width=0)
        end1, end2 = approach_loa1[-1], approach_loa2[-1]
        begin1, begin2 = approach_loa1[0], approach_loa2[0]
        ax.plot([end1[0], end2[0]], [end1[1], end2[1]], [end1[2], end2[2]], 'r-', linewidth=5)
        ax.plot([end1[0], begin1[0]], [end1[1], begin1[1]], [end1[2], begin1[2]], 'r-', linewidth=5)
        ax.plot([begin2[0], end2[0]], [begin2[1], end2[1]], [begin2[2], end2[2]], 'r-', linewidth=5)
        c1_found, _ = ParallelJawPtGrasp3D.find_contact(approach_loa1, obj, vis=False, strict=True)
        c2_found, _ = ParallelJawPtGrasp3D.find_contact(approach_loa2, obj, vis=False, strict=True)
        approach_collision = c1_found or c2_found
        if approach_collision:
            plt.clf()
            return False

        # get line of action
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, self.axis_, self.open_width, obj,
                                                                     num_samples, min_width=self.close_width)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -self.axis_, self.open_width, obj,
                                                                     num_samples, min_width=self.close_width)

        # find contacts
        c1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=False)
        c2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=False)
        begin1, begin2 = line_of_action1[0], line_of_action2[0]
        end1, end2 = obj.sdf.transform_pt_obj_to_grid(c1.point), obj.sdf.transform_pt_obj_to_grid(c2.point)
        print(end1, end2)
        ax.plot([end1[0], begin1[0]], [end1[1], begin1[1]], [end1[2], begin1[2]], 'r-', linewidth=5)
        ax.plot([begin2[0], end2[0]], [begin2[1], end2[1]], [begin2[2], end2[2]], 'r-', linewidth=5)
        ax.scatter(end1[0], end1[1], end1[2], s=80, c='g')
        ax.scatter(end2[0], end2[1], end2[2], s=80, c='g')

        ax.set_xlim3d(0, obj.sdf.dims_[0])
        ax.set_ylim3d(0, obj.sdf.dims_[1])
        ax.set_zlim3d(0, obj.sdf.dims_[2])
        plt.title(','.join([str(i) for i in args]))
        plt.draw()

        contacts_found = c1_found and c2_found
        return contacts_found

    @staticmethod
    def create_line_of_action(g, axis, width, obj, num_samples, min_width=0, convert_grid=True):
        """
        Creates a straight line of action, or list of grid points, from a given point and direction in world or grid coords

        Parameters
        ----------
        g : 3x1 :obj:`numpy.ndarray`
            start point to create the line of action
        axis : normalized 3x1 :obj:`numpy.ndarray`
            normalized numpy 3 array of grasp direction
        width : float
            the grasp width
        num_samples : int
            number of discrete points along the line of action
        convert_grid : bool
            whether or not the points are specified in world coords

        Returns
        -------
        line_of_action : :obj:`list` of 3x1 :obj:`numpy.ndarrays`
            coordinates to pass through in 3D space for contact checking
        """
        num_samples = max(num_samples, 3)  # always at least 3 samples
        line_of_action = [g + t * axis for t in
                          np.linspace(0, float(width) / 2 - float(min_width) / 2, num=num_samples)]
        if convert_grid:
            as_array = np.array(line_of_action).T
            transformed = obj.sdf.transform_pt_obj_to_grid(as_array)
            line_of_action = list(transformed.T)
        return line_of_action

    @staticmethod
    def find_contact(line_of_action, obj, vis=False, strict=False):
        """
        Find the point at which a point traveling along a given line of action hits a surface.

        Parameters
        ----------
        line_of_action : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            the points visited as the fingers close (grid coords)
        obj : :obj:`GraspableObject3D`
            to check contacts on
        vis : bool
            whether or not to display the contact check (for debugging)

        Returns
        -------
        contact_found : bool
            whether or not the point contacts the object surface
        contact : :obj:`Contact3D`
            found along line of action (None if contact not found)
        """
        contact_found = False
        pt_zc = None
        pt_zc_world = None
        contact = None
        num_pts = len(line_of_action)
        sdf_here = 0
        sdf_before = 0
        pt_grid = None
        pt_before = None

        # step along line of action, get points on surface when possible
        i = 0
        reason = 0
        while i < num_pts and not contact_found:
            # update loop vars
            pt_before_before = pt_before
            pt_before = pt_grid
            sdf_before_before = sdf_before
            sdf_before = sdf_here
            pt_grid = line_of_action[i]

            # visualize
            '''
            if vis:
                ax = plt.gca(projection='3d')
                ax.scatter(pt_grid[0], pt_grid[1], pt_grid[2], s=10,c='r')
                plt.show(block=False)
            '''
            # check surface point
            on_surface, sdf_here = obj.sdf.on_surface(pt_grid)

            if on_surface:
                contact_found = True
                reason = 0
                if strict:
                    return contact_found, None

                # quadratic approximation to find actual zero crossing
                if i == 0:
                    pt_after = line_of_action[i + 1]
                    sdf_after = obj.sdf[pt_after]
                    pt_after_after = line_of_action[i + 2]
                    sdf_after_after = obj.sdf[pt_after_after]

                    pt_zc = Sdf3D.find_zero_crossing_quadratic(pt_grid, sdf_here, pt_after, sdf_after, pt_after_after,
                                                               sdf_after_after)

                    # contact not yet found if next sdf value is smaller
                    #if pt_zc is None or np.abs(sdf_after) + obj.sdf.resolution < np.abs(sdf_here):
                    if pt_zc is None or np.abs(sdf_after) < np.abs(sdf_here):
                        contact_found = False

                elif i == len(line_of_action) - 1:
                    pt_zc = Sdf3D.find_zero_crossing_quadratic(pt_before_before, sdf_before_before, pt_before,
                                                               sdf_before, pt_grid, sdf_here)

                    if pt_zc is None:
                        contact_found = False
                        reason = 3

                else:
                    pt_after = line_of_action[i + 1]
                    sdf_after = obj.sdf[pt_after]
                    pt_zc = Sdf3D.find_zero_crossing_quadratic(pt_before, sdf_before, pt_grid, sdf_here, pt_after,
                                                               sdf_after)

                    # contact not yet found if next sdf value is smaller
                    '''
                    if pt_zc is None or np.abs(sdf_after) + obj.sdf.surface_thresh < np.abs(sdf_here):
                        contact_found = False
                        reason = 1
                    '''
                    if pt_zc is None:
                        contact_found = False
                        reason = 1
                    #elif np.abs(sdf_after) + obj.sdf.resolution < np.abs(sdf_here):
                    elif np.abs(sdf_after) < np.abs(sdf_here):
                        contact_found = False
                        reason = 2
            i = i + 1

        if contact_found:
            pt_zc_world = obj.sdf.transform_pt_grid_to_obj(pt_zc)
            # visualization
            if vis and contact_found:
                ax = plt.gca(projection='3d')
                ax.scatter(pt_zc_world[0], pt_zc_world[1], pt_zc_world[2], s=20, c='m')
                ax.plot([pt_zc_world[0], obj.sdf.transform_pt_grid_to_obj(line_of_action[0])[0]],
                        [pt_zc_world[1], obj.sdf.transform_pt_grid_to_obj(line_of_action[0])[1]],
                        [pt_zc_world[2], obj.sdf.transform_pt_grid_to_obj(line_of_action[0])[2]],
                        linewidth=1, c='g')
                plt.show(block=False)
            in_direction_grid = line_of_action[-1] - line_of_action[0]
            in_direction_grid = in_direction_grid / np.linalg.norm(in_direction_grid)
            in_direction = obj.sdf.transform_pt_grid_to_obj(in_direction_grid, direction=True)
            contact = Contact3D(obj, pt_zc_world, in_direction=in_direction)
            if contact.normal is None:
                contact_found = False
                print("contact.normal is None!")
                #logger.warning('contact.normal is None!')
        else:
            pass
            #print("reason:", reason)
            #logger.warning('reason:', reason)
        return contact_found, contact

    def _angle_aligned_with_stable_pose(self, stable_pose):
        """
        Returns the y-axis rotation angle that'd allow the current pose to align with stable pose.
        """

        def _argmin(f, a, b, n):
            # finds the argmax x of f(x) in the range [a, b) with n samples
            delta = (b - a) / n
            min_y = f(a)
            min_x = a
            for i in range(1, n):
                x = i * delta
                y = f(x)
                if y <= min_y:
                    min_y = y
                    min_x = x
            return min_x

        def _get_matrix_product_x_axis(grasp_axis, normal):
            def matrix_product(theta):
                R = ParallelJawPtGrasp3D._get_rotation_matrix_y(theta)
                grasp_axis_rotated = np.dot(R, grasp_axis)
                return abs(np.dot(normal, grasp_axis_rotated))

            return matrix_product

        stable_pose_normal = stable_pose.r[2, :]

        theta = _argmin(
            _get_matrix_product_x_axis(np.array([1, 0, 0]), np.dot(inv(self.unrotated_full_axis), stable_pose_normal)),
            0, 2 * np.pi, 1000)
        return theta

    def grasp_y_axis_offset(self, theta):
        """ Return a new grasp with the given approach angle.
        
        Parameters
        ----------
        theta : float
            approach angle for the new grasp

        Returns
        -------
        :obj:`ParallelJawPtGrasp3D`
            grasp with the given approach angle
        """
        new_grasp = deepcopy(self)
        new_grasp.approach_angle = theta + self.approach_angle
        return new_grasp

    def parallel_table(self, stable_pose):
        """
        Returns a grasp with approach_angle set to be perpendicular to the table normal specified in the given stable pose.

        Parameters
        ----------
        stable_pose : :obj:`StablePose`
            the pose specifying the table

        Returns
        -------
        :obj:`ParallelJawPtGrasp3D`
            aligned grasp
        """
        theta = self._angle_aligned_with_stable_pose(stable_pose)
        new_grasp = deepcopy(self)
        new_grasp.approach_angle = theta
        return new_grasp

    def _angle_aligned_with_table(self, table_normal):
        """
        Returns the y-axis rotation angle that'd allow the current pose to align with the table normal.
        """

        def _argmax(f, a, b, n):
            # finds the argmax x of f(x) in the range [a, b) with n samples
            delta = (b - a) / n
            max_y = f(a)
            max_x = a
            for i in range(1, n):
                x = i * delta
                y = f(x)
                if y >= max_y:
                    max_y = y
                    max_x = x
            return max_x

        def _get_matrix_product_x_axis(grasp_axis, normal):
            def matrix_product(theta):
                R = ParallelJawPtGrasp3D._get_rotation_matrix_y(theta)
                grasp_axis_rotated = np.dot(R, grasp_axis)
                return np.dot(normal, grasp_axis_rotated)

            return matrix_product

        theta = _argmax(
            _get_matrix_product_x_axis(np.array([1, 0, 0]), np.dot(inv(self.unrotated_full_axis), -table_normal)), 0,
            2 * np.pi, 64)
        return theta

    def perpendicular_table(self, stable_pose):
        """
        Returns a grasp with approach_angle set to be aligned width the table normal specified in the given stable pose.

        Parameters
        ----------
        stable_pose : :obj:`StablePose` or :obj:`RigidTransform`
            the pose specifying the orientation of the table

        Returns
        -------
        :obj:`ParallelJawPtGrasp3D`
            aligned grasp
        """
        if isinstance(stable_pose, StablePose):
            table_normal = stable_pose.r[2, :]
        else:
            table_normal = stable_pose.rotation[2, :]
        theta = self._angle_aligned_with_table(table_normal)
        new_grasp = deepcopy(self)
        new_grasp.approach_angle = theta
        return new_grasp

    def project_camera(self, T_obj_camera, camera_intr):
        """ Project a grasp for a given gripper into the camera specified by a set of intrinsics.
        
        Parameters
        ----------
        T_obj_camera : :obj:`autolab_core.RigidTransform`
            rigid transformation from the object frame to the camera frame
        camera_intr : :obj:`perception.CameraIntrinsics`
            intrinsics of the camera to use
        """
        # compute pose of grasp in camera frame
        T_grasp_camera = T_obj_camera * self.T_grasp_obj
        y_axis_camera = T_grasp_camera.y_axis[:2]
        if np.linalg.norm(y_axis_camera) > 0:
            y_axis_camera = y_axis_camera / np.linalg.norm(y_axis_camera)

        # compute grasp axis rotation in image space
        rot_z = np.arccos(y_axis_camera[0])
        if y_axis_camera[1] < 0:
            rot_z = -rot_z
        while rot_z < 0:
            rot_z += 2 * np.pi
        while rot_z > 2 * np.pi:
            rot_z -= 2 * np.pi

        # compute grasp center in image space
        t_grasp_camera = T_grasp_camera.translation
        p_grasp_camera = Point(t_grasp_camera, frame=camera_intr.frame)
        u_grasp_camera = camera_intr.project(p_grasp_camera)
        d_grasp_camera = t_grasp_camera[2]
        return Grasp2D(u_grasp_camera, rot_z, d_grasp_camera,
                       width=self.open_width,
                       camera_intr=camera_intr)

    @staticmethod
    def grasp_from_contact_and_axis_on_grid(obj, grasp_c1_world, grasp_axis_world, grasp_width_world, grasp_angle=0,
                                            jaw_width_world=0,
                                            min_grasp_width_world=0, vis=False, backup=0.5):
        """
        Creates a grasp from a single contact point in grid coordinates and direction in grid coordinates.
        
        Parameters
        ----------
        obj : :obj:`GraspableObject3D`
            object to create grasp for
        grasp_c1_grid : 3x1 :obj:`numpy.ndarray`
            contact point 1 in world
        grasp_axis : normalized 3x1 :obj:`numpy.ndarray`
           normalized direction of the grasp in world
        grasp_width_world : float
            grasp_width in world coords
        jaw_width_world : float
            width of jaws in world coords
        min_grasp_width_world : float
            min closing width of jaws
        vis : bool
            whether or not to visualize the grasp
        
        Returns
        -------
        g : :obj:`ParallelJawGrasp3D`
            grasp created by finding the second contact
        c1 : :obj:`Contact3D`
            first contact point on the object
        c2 : :obj:`Contact3D`
            second contact point on the object
        """
        # transform to grid basis
        grasp_axis_world = grasp_axis_world / np.linalg.norm(grasp_axis_world)
        grasp_axis_grid = obj.sdf.transform_pt_obj_to_grid(grasp_axis_world, direction=True)
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(grasp_width_world)
        min_grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(min_grasp_width_world)
        grasp_c1_grid = obj.sdf.transform_pt_obj_to_grid(
            grasp_c1_world) - backup * grasp_axis_grid  # subtract to find true point
        num_samples = int(2 * grasp_width_grid)  # at least 2 samples per grid
        g2 = grasp_c1_grid + (grasp_width_grid - backup) * grasp_axis_grid

        # get line of action
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(grasp_c1_grid, grasp_axis_grid, grasp_width_grid,
                                                                     obj, num_samples,
                                                                     min_width=min_grasp_width_grid, convert_grid=False)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2, -grasp_axis_grid, 2 * grasp_width_grid, obj,
                                                                     num_samples,
                                                                     min_width=0, convert_grid=False)
        if vis:
            obj.sdf.scatter()
            ax = plt.gca(projection='3d')
            ax.scatter(grasp_c1_grid[0] - grasp_axis_grid[0], grasp_c1_grid[1] - grasp_axis_grid[1],
                       grasp_c1_grid[2] - grasp_axis_grid[2], c='r')
            ax.scatter(grasp_c1_grid[0], grasp_c1_grid[1], grasp_c1_grid[2], s=80, c='b')

        # compute the contact points on the object
        contact1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=vis)
        contact2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=vis)

        if vis:
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.draw()
        if not contact1_found or not contact2_found or np.linalg.norm(c1.point - c2.point) <= min_grasp_width_world:
            logging.debug('No contacts found for grasp')
            return None, None, None

        # create grasp
        grasp_center = ParallelJawPtGrasp3D.center_from_endpoints(c1.point, c2.point)
        grasp_axis = ParallelJawPtGrasp3D.axis_from_endpoints(c1.point, c2.point)
        configuration = ParallelJawPtGrasp3D.configuration_from_params(grasp_center, grasp_axis, grasp_width_world,
                                                                       grasp_angle, jaw_width_world)
        return ParallelJawPtGrasp3D(configuration), c1, c2  # relative to object

    def surface_information(self, graspable, width=2e-2, num_steps=21, direction=None):
        """ Return the patch surface information at the contacts that this grasp makes on a graspable.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            object to get surface information for
        width : float
            width of the window in obj frame
        num_steps : int
            number of steps

        Returns
        -------
        :obj:`list` of :obj:`SurfaceWindow`
             surface patches, one for each contact
        """
        return graspable.surface_information(self, width, num_steps, direction1=self.axis_, direction2=-self.axis_)


class VacuumPoint(Grasp):
    """ Defines a vacuum target point and axis in 3D space (5 DOF)
    """

    def __init__(self, radius, graspble):
        #center, axis = VacuumPoint.params_from_configuration(configuration)
        self._center = center
        self._radius = radius
        self._graspble = graspble

    @property
    def center(self):
        return self._center

    @staticmethod
    def find_projection(obj, curr_loc, direction, max_projection, num_samples, vis=False):
        """Finds the point of contact when shooting a direction ray from curr_loc.
        Params:
            curr_loc - numpy 3 array of the starting point in obj frame
            direction - normalized numpy 3 array, direction to look for contact
            max_projection - float maximum amount to search forward for a contact (meters)
            num_samples - float number of samples when finding contacts
        Returns:
            found - True if projection contact is found
            projection_contact - Contact3D instance
        """
        # get start of projection
        line_of_action = ParallelJawPtGrasp3D.create_line_of_action(
            curr_loc, direction, max_projection, obj, num_samples)
        found, projection_contact = ParallelJawPtGrasp3D.find_contact(
            line_of_action, obj, vis=vis)
        '''
        if vis:
            ax = plt.gca(projection = '3d')
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
        '''
        return found, projection_contact

    @staticmethod
    def compute_contactSurface_projection(contact_pt, direction, u1, u2, width=3e-2, num_steps=31,
                                          max_projection=0.06, back_up=0.02, samples_per_grid=2.0,
                                          num_cone_faces=8,
                                          sigma_spatial=1,
                                          vis = False, vis_cone = False,
                                          compute_weighted_covariance=False,
                                          disc=False, num_radial_steps=5, debug_objs=None):
        """Compute the projection window onto the virtual camera defined by self.point, normal,
            u1, u2 and back up.
        Params:
            u1, u2 - orthogonal numpy 3 arrays

            width - float width of the window in obj frame
            num_steps - int number of steps
            max_projection - float maximum amount to search forward for a
                contact (meters)

            back_up - amount in meters to back up before projecting
            samples_per_grid - float number of samples per grid when finding contacts
            sigma - bandwidth of gaussian filter on window
            direction - dir to do the projection along
            compute_weighted_covariance - whether to return the weighted
               covariance matrix, along with the window
        Returns:
            window - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        if direction is None:  # normal and tangents not found
            #raise ValueError('Direction could not be computed')
            logger.warning('Failed to calculate tangents...Current point is skiped')
            return False, []
        # number of samples used when looking for contacts
        no_contact = NO_CONTACT_DIST # NO_CONTACT_DIST = 0.0
        num_samples = int(samples_per_grid * max_projection / contact_pt.graspable.sdf.surface_thresh)
        #num_samples = int (max_projection / )
        window = np.zeros(num_steps ** 2)

        res = width / num_steps

        #print("num_sample: ", num_samples, "   res: ", res)

        scales = np.linspace(-width / 2.0 + res / 2.0, width / 2.0 - res / 2.0, num_steps)
        scales_it = it.product(scales, repeat=2) # (1,1) (1,2) (1,3)...  not (1,1) (2,1) (3,1)...
        coors_x = np.zeros([num_steps ** 2, 1])
        coors_y = np.zeros([num_steps ** 2, 1])
        win_normals = np.zeros([num_steps ** 2, 3])
        window_info = None

        cone_coors = np.zeros([num_cone_faces + 1, 3])
        cone_coors[8, :] = contact_pt.point
        cone_normal = np.zeros([num_cone_faces + 1, 3])
        cone_normal[8, :] = contact_pt.normal
        cone_info = None

        # start computing weighted covariance matrix
        if compute_weighted_covariance:
            cov = np.zeros((3, 3))
            cov_weight = 0

        viewpoint = contact_pt.point - direction * back_up
        #print(contact_pt.point, np.shape(contact_pt.point))
        #print(viewpoint, np.shape(viewpoint))
        #input()
        if vis:
            # ax = plt.gca(projection='3d')
            # contact_pt.graspable_.sdf.scatter() # scatter 散点图
            #ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
            # ax = plt.gca(projection='3d')
            fig = plt.figure()
            # fig.set_size_inches(3.32, 2.49)  # width, height
            # fig.set_dpi(300)
            ax = Axes3D(fig)

            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            ax.set_xlim3d(contact_pt.graspable.min_pos[0] - back_up - 0.01,
                          contact_pt.graspable.max_pos[0] + back_up + 0.01)
            ax.set_ylim3d(contact_pt.graspable.min_pos[1] - back_up - 0.01,
                          contact_pt.graspable.max_pos[1] + back_up + 0.01)
            ax.set_zlim3d(contact_pt.graspable.min_pos[2] - back_up - 0.01,
                          contact_pt.graspable.max_pos[2] + back_up + 0.01)

            ax.scatter(contact_pt.graspable.surface_points[:, 0],
                       contact_pt.graspable.surface_points[:, 1],
                       contact_pt.graspable.surface_points[:, 2], s=1, c='b')  # 绘制数据点
            ax.scatter(contact_pt.point[0], contact_pt.point[1], contact_pt.point[2], s=30, c='g')
            ax.scatter(viewpoint[0], viewpoint[1], viewpoint[2], s=3, c='r')
            plt.show(block=False)

            ''''''
        #print("signed_dis:", contact_pt.graspable.sdf[contact_pt.graspable.sdf.transform_pt_obj_to_grid(contact_pt.point)])
        #plt.show()
        cnt_vacancy = 0
        for i, (c1, c2) in enumerate(scales_it):
            coors_x[i] = c1
            coors_y[i] = c2
            curr_loc = viewpoint + c1 * u1 + c2 * u2
            #print("viewpoint:", c1, c2)
            #curr_loc = viewpoint
            #curr_loc_grid = contact_pt.graspable.sdf.transform_pt_obj_to_grid(curr_loc)
            #print("signed_dis:", contact_pt.graspable.sdf[curr_loc_grid])
            '''
            if self.graspable.sdf.is_out_of_bounds(curr_loc_grid):
                window[i] = no_contact
                continue
            '''
            if vis:
                ax.scatter(curr_loc[0], curr_loc[1], curr_loc[2], s=10, c='y')
                plt.show(block=False)

            found, projection_contact = VacuumPoint.find_projection(contact_pt.graspable,
                curr_loc, direction, max_projection, num_samples, vis= vis and not vis_cone)
            #.show(block=False)
            #return True, found, projection_contact
            if found:
                logging.debug('%d found.' %(i))
                ''''''
                #sign = direction.dot(projection_contact.point - curr_loc) # decide the sign of distance
                #projection = (sign / abs(sign)) * np.linalg.norm(projection_contact.point - curr_loc)
                projection = np.linalg.norm(projection_contact.point - curr_loc)
                #projection = min(projection, max_projection)
                win_normals[i,:] = projection_contact.normal

            else:
                logging.debug('%d not found.' % (i))
                projection = no_contact
                cnt_vacancy = cnt_vacancy + 1.

            window[i] = projection
        logger.info('vacancy rate: %f' % (cnt_vacancy / (1.0 * num_steps**2)))
        #print("vacancy rate:", cnt_vacancy / (1.0 * num_steps**2))
        #input()

        if cnt_vacancy / (1.0 * num_steps**2) > 0.05:
            '''omit this frame due to serious vacancy points'''
            #if vis or vis_cone:
            #    plt.show()
            return False, window_info, cone_info
        elif cnt_vacancy > 0.:
            '''amend this frame if possible'''
            avg_normals = np.mean(win_normals, axis=0) # average normals
            avg_normals = avg_normals / np.linalg.norm(avg_normals)
            avg_dis = np.mean(window) / (cnt_vacancy / (1.0 * num_steps**2))

            for i in range(0, win_normals.shape[0]):
                if np.linalg.norm(win_normals[i, :]) < 1e-2:
                    win_normals[i, :] = avg_normals # fill the vacancies with average normal
                    window[i] = avg_dis
            #print("avg_normals:", avg_normals, np.linalg.norm(avg_normals))
            logger.info('avg_normals: %f, %f, %f. %f' % (avg_normals[0], avg_normals[1], avg_normals[2],
                                                         avg_dis))
        # find convex combinations of tangent vectors
        cnt_vacancy = 0.
        for j in range(0, num_cone_faces):
            tmp_x = width * 0.5 * np.cos(2 * np.pi * (float(j) / num_cone_faces))
            tmp_y = width * 0.5 * np.sin(2 * np.pi * (float(j) / num_cone_faces))
            curr_loc = viewpoint + tmp_x * u1 + tmp_y * u2

            found, projection_contact = VacuumPoint.find_projection(contact_pt.graspable,
                curr_loc, direction, max_projection, num_samples, vis=False)
            if found:
                logging.debug('%d cone face found.' %(i))
                cone_normal[j, :] = projection_contact.normal
                cone_coors[j, :] = projection_contact.point
                # this part of visualization is not reliable enough
                if vis and vis_cone:
                    ax = plt.gca(projection='3d')
                    ax.plot([cone_coors[j, 0],  viewpoint[0]],
                            [cone_coors[j, 1],  viewpoint[1]],
                            [cone_coors[j, 2],  viewpoint[2]],
                            linewidth=2, c='r')
                    if j > 0:
                        ax.plot([cone_coors[j - 1, 0], cone_coors[j, 0]],
                                [cone_coors[j - 1, 1], cone_coors[j, 1]],
                                [cone_coors[j - 1, 2], cone_coors[j, 2]],
                                linewidth=2, c='lightseagreen')
            else:
                logging.debug('%d cone face not found.' % (i))
                cnt_vacancy = cnt_vacancy + 1.
        if vis and vis_cone:
            ax.plot([cone_coors[0, 0], cone_coors[7, 0]],
                    [cone_coors[0, 1], cone_coors[7, 1]],
                    [cone_coors[0, 2], cone_coors[7, 2]],
                    linewidth=2, c='lightseagreen')
        if cnt_vacancy > 0.:
            logger.info('imperfect cone.')
            avg_normals = np.mean(cone_normal, axis=0)  # average normals
            avg_normals = avg_normals / np.linalg.norm(avg_normals)
            for j in range(0, num_cone_faces):
                if np.linalg.norm(cone_normal[j, :]) < 1e-2:
                    cone_normal[j, :] = avg_normals
        cone_info = np.c_[cone_coors, cone_normal]
        #input()
        if not disc:
            window = window.reshape((num_steps, num_steps)) # transpose to make x-axis along columns
            # print("after reshape", window[num_steps-1, 0])
            # apply bilateral filter
            if sigma_spatial > 0.0:
                window_min_val = np.min(window)
                window_pos = window - window_min_val # normorlization
                # window_pos_blur = denoise_bilateral(window_pos, sigma_spatial=sigma_spatial, mode='edge')
                # window = window_pos_blur + window_min_val
                window = denoise_bilateral(window_pos, sigma_spatial=sigma_spatial, mode='edge')
        #coors_x = np.reshape(coors_x.reshape((num_steps, num_steps)).T, [1, num_steps **2])
        #coors_y = np.reshape(coors_y.reshape((num_steps, num_steps)).T, [1, num_steps **2])
        window_info = np.c_[coors_x, coors_y, window.reshape(num_steps ** 2, 1), win_normals]

        #print(window_info)
        #if vis or vis_cone:
        #    plt.show()
        return True, window_info, cone_info,

    @staticmethod
    def friction_cone(direction = None, window_info = None, cone_info = None, friction_coef=0.5):
        """Computes the friction cone with given coordinate, point clouds.
        Params:
            direction - approching direction
            window_info - point clouds with surface normal
            cone_info - cone with coors and surface normal
        Returns:
            BOOL - False/True
            _friction_cone - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        num_cone_faces = np.shape(cone_info)[0] - 1
        _friction_cone = np.zeros((3, num_cone_faces))
        # check whether contact would slip, which is whether or not the tangent force is always
        # greater than the frictional force
        surf_avg_norm = np.mean(window_info, axis=0)[3:]
        surf_in_norm = -surf_avg_norm / np.linalg.norm(surf_avg_norm)
        ''''''
        normal_force_mag = 1.
        normal_force_mag = max(np.dot(direction, surf_in_norm), 0.)
        #print(direction, surf_in_norm, normal_force_mag)
        #print(direction[0]*surf_in_norm[0] + direction[1]*surf_in_norm[1] + direction[2]*surf_in_norm[2])
        #print(np.shape(direction), np.shape(surf_in_norm))
        #input()
        _, t1, t2 = VacuumPoint.tangents(direction=surf_in_norm)

        tan_force_x = np.dot(direction, t1)
        tan_force_y = np.dot(direction, t2)
        tan_force_mag = np.sqrt(tan_force_x ** 2 + tan_force_y ** 2)
        friction_force_mag = friction_coef * normal_force_mag
        '''
        if friction_force_mag < tan_force_mag:
            logging.warning('Contact would slip')
            return False, _friction_cone
        '''
        # set up friction cone
        force = direction

        # find convex combinations of tangent vectors
        for j in range(num_cone_faces):
            tan_vec = t1 * np.cos(2 * np.pi * (float(j) / num_cone_faces)) + t2 * np.sin(
                2 * np.pi * (float(j) / num_cone_faces))
            normal_force_mag = max(np.dot(direction, -cone_info[j, 3:]), 0.)
            _friction_cone[:, j] = force + normal_force_mag * friction_coef * tan_vec*10.
            #_friction_cone[:, j] = force + friction_coef * tan_vec
        return True, _friction_cone

    @staticmethod
    def tangents(direction, align_axes=True, max_samples=1000):
        """Returns the direction vector and tangent vectors at a contact point.
        The direction vector defaults to the *inward-facing* normal vector at
        this contact.
        The direction and tangent vectors for a right handed coordinate frame.

        Parameters
        ----------
        direction : 3x1 :obj:`numpy.ndarray`
            direction to find orthogonal plane for
        align_axes : bool
            whether or not to align the tangent plane to the object reference frame
        max_samples : int
            number of samples to use in discrete optimization for alignment of reference frame

        Returns
        -------
        direction : normalized 3x1 :obj:`numpy.ndarray`
            direction to find orthogonal plane for
        t1 : normalized 3x1 :obj:`numpy.ndarray`
            first tangent vector, x axis
        t2 : normalized 3x1 :obj:`numpy.ndarray`
            second tangent vector, y axis
        """

        # transform to
        direction = direction.reshape((3, 1))  # make 2D for SVD

        # get orthogonal plane
        U, _, _ = np.linalg.svd(direction)

        # U[:, 1:] spans the tanget plane at the contact
        x, y = U[:, 1], U[:, 2]

        # make sure t1 and t2 obey right hand rule
        z_hat = np.cross(x, y)
        if z_hat.dot(direction) < 0:
            y = -y
        v = x
        w = y

        # redefine tangent x axis to automatically align with the object x axis
        if align_axes:
            max_ip = 0
            max_theta = 0
            target = np.array([1, 0, 0])
            theta = 0
            d_theta = 2 * np.pi / float(max_samples)
            for i in range(max_samples):
                v = np.cos(theta) * x + np.sin(theta) * y
                if v.dot(target) > max_ip:
                    max_ip = v.dot(target)
                    max_theta = theta
                theta = theta + d_theta

            v = np.cos(max_theta) * x + np.sin(max_theta) * y
            w = np.cross(direction.ravel(), v)
        return np.squeeze(direction), v, w

    @staticmethod
    def torques(forces, contact_pt):
        """
        Get the torques that can be applied by a set of force vectors at the contact point.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            the forces applied at the contact
        contact_pt: obj:`contacts`
        Returns
        -------
        success : bool
            whether or not computation was successful
        torques : 3xN :obj:`numpy.ndarray`
            the torques that can be applied by given forces at the contact
        """
        as_grid = contact_pt.graspable.sdf.transform_pt_obj_to_grid(contact_pt.point)
        on_surface, _ = contact_pt.graspable.sdf.on_surface(as_grid)
        if not on_surface:
            logging.debug('Contact point not on surface')
            return False, None

        num_forces = forces.shape[1]
        torques = np.zeros([3, num_forces])
        moment_arm = contact_pt.graspable.moment_arm(contact_pt.point)
        for i in range(num_forces):
            torques[:, i] = np.cross(moment_arm, forces[:, i])

        return True, torques

    @staticmethod
    def transfer_vector_to_newXYZ(vector, axis_z=None, axis_x=None, axis_y=None):
        """ transfer the vector to a defined coordinates
        :param vector:
        :param axis_z: orthogonal normalized vectors to define the new coordinate
        :param axis_x:
        :param axis_y:
        :return: False/True, new matrix
        """
        out_vector = np.zeros(np.shape(vector))
        if axis_z is None:
            logger.info('define at least 1 axis for transformation.')
            return False, vector
        if axis_x is None and axis_y is None:
            _, axis_x, axis_y = VacuumPoint.tangents(axis_z)
        #print(vector)
        for i in range(0, np.shape(vector)[1]):
            """ |v1|*cos(theta) = v1 * axis_1 / |axis_1|
            """
            #print(vector[:, i])
            #print(vector[0:3, i])
            #input()
            tx = np.dot(vector[0:3, i], axis_x) / np.linalg.norm(axis_x)
            ty = np.dot(vector[0:3, i], axis_y) / np.linalg.norm(axis_y)
            tz = np.dot(vector[0:3, i], axis_z) / np.linalg.norm(axis_z)
            ta = np.dot(vector[3:, i].reshape(1,3), axis_x) / np.linalg.norm(axis_x)
            tb = np.dot(vector[3:, i].reshape(1,3), axis_y) / np.linalg.norm(axis_y)
            tc = np.dot(vector[3:, i].reshape(1,3), axis_z) / np.linalg.norm(axis_z)
            out_vector[0:3, i] = [tx, ty, tz]
            out_vector[3:, i] = [ta, tb, tc]
        return True, out_vector

    @staticmethod
    def grasp_matrix(contact_pt, direction = None, window_info = None, cone_info = None,
                     friction_coef = 0.5, u1=None, u2=None, approching_perspective=True,
                     new_x=None, new_y=None, new_z=None):
        """Computes the friction cone with given coordinate, point clouds.
        Params:
            contact_pt: obj:`contacts`
            direction - approching direction
            window_info - point clouds with surface normals
            cone_info - cone with coors and surface mormals
            friction_coef - friction coefficient
            approching_perspective - True/False if transform all of data into camera perspective
            u1, u2 - orthogonal axis for approching perspective [u1, u2, direction]
        Returns:
            flg_Gmatrix - False/True if grasp matrix is generated
            flg_trans - False/True if grasp matrix is transformed
            grasp_matrix - 6 * N array   N = faces number of the cone
        """
        grasp_matrix = None
        flg_trans = False
        flg_Gmatrix = False
        if np.linalg.norm(direction) - 1. > 1e-5:
            logger.warning('normalized deirection vector is needed')
            return flg_Gmatrix, flg_trans, grasp_matrix
        flg_fric, forces = VacuumPoint.friction_cone(direction=direction,
                                                     window_info=window_info, cone_info=cone_info,
                                                     friction_coef=friction_coef)
        if flg_fric:
            #print(forces)
            #print("===============")
            flg_torques, torques = VacuumPoint.torques(forces=forces, contact_pt=contact_pt)
            if flg_torques:
                #print(torques)
                #print("===============")
                flg_Gmatrix = True
                grasp_matrix =  np.r_[forces, torques]
                if approching_perspective:
                    if np.linalg.norm(u1) - 1. > 1e-5 or np.linalg.norm(u2) - 1. > 1e-5:
                        logger.warning('transformation failed, normalized orthogonal vectors are needed')
                        return flg_Gmatrix, flg_trans, grasp_matrix
                    else:
                        #print(grasp_matrix)
                        #print(np.sum(grasp_matrix, axis=1))
                        flg_trans, out_matrix = VacuumPoint.transfer_vector_to_newXYZ(grasp_matrix,
                                                                                      axis_z=new_z,
                                                                                      axis_x=new_x,
                                                                                      axis_y=new_y)
                        return flg_Gmatrix, flg_trans, out_matrix
                else:
                    return flg_Gmatrix, flg_trans, grasp_matrix
            else:
                return flg_Gmatrix, flg_trans, grasp_matrix
        else:
            return flg_Gmatrix, flg_trans, grasp_matrix

    @staticmethod
    def grasp_quality_qp(grasp_matrix,
                         stable_wrench,
                         fric_coef=0.5, elastic_limit=0.005, radius=0.015, max_force=250,
                         differ=0.2,
                         wrench_regularizer=1e-10):

        """ Checks force closure by solving a quadratic program (whether or not zero is in the convex hull)

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        Returns
        -------

        """
        min_norm, wrenchs = VacuumPoint.min_norm_vector_in_facet(facet=grasp_matrix,
                                                                 offset=stable_wrench,
                                                                 differ=differ,
                                                                 wrench_regularizer=wrench_regularizer)
        return min_norm, wrenchs

    @staticmethod
    def min_norm_vector_in_facet(facet,
                                 offset,
                                 differ=0.2,
                                 fric_coef=0.5, elastic_limit=0.005, radius=0.015, max_force=250,
                                 wrench_regularizer=1e-10):
        """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.
        Parameters
        ----------
        facet : 6xN :obj:`numpy.ndarray`
            vectors forming the facet
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        Returns
        -------
        float
            minimum norm of any point in the convex hull of the facet
        Nx1 :obj:`numpy.ndarray`
            vector of coefficients that achieves the minimum
        """
        [dof, dim] = facet.shape

        # create alpha weights for vertices of facet
        G = facet.T.dot(facet)
        grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h
        P = cvx.matrix(2 * grasp_matrix, tc='d')  # quadratic cost for Euclidean dist
        q = cvx.matrix(np.dot(facet.T,offset), tc='d')
        #G = cvx.matrix(-np.eye(dim))  # greater than zero constraint
        #h = cvx.matrix(np.zeros((dim, 1)))

        tmp_matrix = np.copy(facet)
        #tmp_matrix[2, :] = -1* tmp_matrix[2, :]
        #print(np.shape(tmp_matrix))
        #input()
        '''
        condition_1 = np.c_[tmp_matrix, np.zeros([dof,dim]), np.zeros([dof,dim])]
        tmp_matrix[2, :] = np.zeros([1, dim])
        condition_2 = np.c_[np.zeros([dof,dim]), -tmp_matrix, np.zeros([dof,dim])]
        condition_3 = np.c_[np.zeros([dim,dim]), np.zeros([dim,dim]), -np.eye(dim)]
        '''
        condition_1 = np.copy(tmp_matrix)
        #tmp_matrix[2, :] = np.zeros([1, dim])
        condition_2 = -tmp_matrix
        #condition_3 = -np.eye(dim)
        condition_4 = -np.eye(dim)
        condition_5 = np.eye(dim)
        G = cvx.matrix(np.r_[condition_1, condition_2, condition_4, condition_5], tc='d')
        #G = cvx.matrix(condition_4)

        max_fx = fric_coef * max_force / (3. ** 0.5) #
        max_fy = max_fx
        max_fz = 9. * max_force
        #max_tx = np.pi * radius * elastic_limit / (2. ** 0.5)
        #max_ty = np.pi * radius * elastic_limit / (2. ** 0.5)
        #max_tz = radius * fric_coef / max_force / (3. ** 0.5)
        max_tx = 1.
        max_ty = 1.
        max_tz = 1.
        constraint_1 = np.array([max_fx, max_fy, max_fz, max_tx, max_ty, max_tz]).reshape(6, 1)
        constraint_2 = np.array([max_fx, max_fy, 0., max_tx, max_ty, max_tz]).reshape(6, 1)
        #constraint_3 = np.zeros([dim, 1])
        avg_weight = 1./dim
        min_weight = avg_weight * (1. - differ)
        max_weight = avg_weight * (1. + differ)
        constraint_4 = np.array(min_weight * np.ones([dim, 1])).reshape(dim, 1)
        constraint_5 = np.array(max_weight * np.ones([dim, 1])).reshape(dim, 1)
        h = cvx.matrix(np.r_[constraint_1, constraint_2, constraint_4, constraint_5], tc='d')
        #h = cvx.matrix(constraint_4)
        #print(np.shape(h))
        #input()

        A = cvx.matrix(np.ones((1, dim)), tc='d')  # sum constraint to enforce convex
        b = cvx.matrix(np.ones(1), tc='d')  # combinations of vertices
        sol = cvx.solvers.qp(P, q, G, h, A, b)
        v = np.array(sol['x'])
        min_norm = np.sqrt(sol['primal objective'])

        return abs(min_norm), v

class GraspQuality_Vacuum():

    def __init__(self, grasp_matrix=None, stable_wrench=None,
                 fric_coef=0.5, elastic_limit=0.005, radius=0.015, max_force=250,
                 differ=0.2, wrench_regularizer=1e-10):
        self.grasp_matrix_ = grasp_matrix
        self.stable_wrench_ = stable_wrench
        self.fric_coef_ = fric_coef
        self.elastic_limit_ = elastic_limit
        self.radius_ = radius
        self.max_force_ = max_force
        self.differ_ = differ
        self.wrench_regularizer_ = wrench_regularizer
        [dof, dim] = np.shape(grasp_matrix)
        start_params = 1.0 / dim
        self.start_wrench_ = np.array(start_params * np.ones([dim, 1])).reshape(dim, 1)

    def set_grasp_matrix(self, input):
        self.grasp_matrix_ = input
    def set_stable_wrench(self, input):
        self.stable_wrench_ = input
    def set_fric_coef(self, input):
        self.fric_coef_ = input
    def set_elastic_limit(self, input):
        self.elastic_limit_ = input
    def set_radius(self, input):
        self.radius_ = input
    def set_max_force(self, input):
        self.max_force_ = input
    def set_differ(self, input):
        self.differ_ = input
    def set_wrench_regularizer(self, input):
        self.wrench_regularizer_ = input

    @property
    def grasp_matrix(self):
        return self.grasp_matrix_
    @property
    def stable_wrench(self):
        return self.stable_wrench_
    @property
    def fric_coef(self):
        return self.fric_coef_
    @property
    def elastic_limit(self):
        return self.elastic_limit_
    @property
    def radius(self):
        return self.radius_
    @property
    def max_force(self):
        return self.max_force_
    @property
    def differ(self):
        return self.differ_
    @property
    def wrench_regularizer(self):
        return self.wrench_regularizer_

    def objective(self, w):
        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h
        facet = self.grasp_matrix[:3,:]
        offset = (self.stable_wrench)[:3]
        dives = facet.dot(w) - offset
        dis = np.linalg.norm(dives) ** 2
        return dis

    def ineq_constraint_min_0(self, w):
        cond = (self.grasp_matrix)[0,:]
        min_fx = -self.fric_coef * self.max_force / (3. ** 0.5) #
        return np.linalg.norm(cond.dot(w) - min_fx)
    def ineq_constraint_min_1(self, w):
        cond = (self.grasp_matrix)[1,:]
        min_fy = -self.fric_coef * self.max_force / (3. ** 0.5)  #
        return np.linalg.norm(cond.dot(w) - min_fy)
    def ineq_constraint_min_2(self, w):
        cond = (self.grasp_matrix)[2,:]
        min_fz = 1.
        return np.linalg.norm(cond.dot(w) - min_fz)
    def ineq_constraint_max_0(self, w):
        cond = (self.grasp_matrix)[0,:]
        max_fx = self.fric_coef * self.max_force / (3. ** 0.5) #
        return np.linalg.norm(max_fx - cond.dot(w))
    def ineq_constraint_max_1(self, w):
        cond = (self.grasp_matrix)[1,:]
        max_fy = self.fric_coef * self.max_force / (3. ** 0.5)  #
        return np.linalg.norm(max_fy - cond.dot(w))
    def ineq_constraint_max_2(self, w):
        cond = (self.grasp_matrix)[2,:]
        max_fz = 0.7067647816774781
        out = lambda w: np.linalg.norm(cond.dot(w)- max_fz)
        return out
        #return np.linalg.norm(cond.dot(w)-max_fz)

    def eq_constraint_1(self, w):
        eq = 2.0
        for i in range(0, np.size(w)):
            eq = eq - w[i]
        return eq

    def evaluation_qp(self):
        avg_weight = 1.0 / np.shape(self.grasp_matrix)[1]
        b = [avg_weight * (1.-self.differ), avg_weight * (1.+self.differ)]
        bs = [b, b, b, b, b, b, b, b]
        con1 = {'type': 'ineq', 'fun': self.ineq_constraint_min_0}
        con2 = {'type': 'ineq', 'fun': self.ineq_constraint_min_1}
        con3 = {'type': 'ineq', 'fun': self.ineq_constraint_min_2}
        con4 = {'type': 'ineq', 'fun': self.ineq_constraint_max_0}
        con5 = {'type': 'ineq', 'fun': self.ineq_constraint_max_1}
        con6 = {'type': 'ineq', 'fun': self.ineq_constraint_max_2}
        con7 = {'type': 'eq', 'fun': self.eq_constraint_1}
        #cons = [ con3, con6, con7]
        #cons = [ con7]
        cond = (self.grasp_matrix)[2, :]
        max_fx = self.fric_coef * self.max_force / (3. ** 0.5)
        max_fy = self.fric_coef * self.max_force / (3. ** 0.5)

        ''''''
        cons = ({'type': 'ineq', 'fun': lambda w: (-self.grasp_matrix)[0, :].dot(w) + max_fx}, \
                {'type': 'ineq', 'fun': lambda w: (self.grasp_matrix)[0, :].dot(w) - max_fx}, \
                {'type': 'ineq', 'fun': lambda w: (-self.grasp_matrix)[1, :].dot(w) + max_fy}, \
                {'type': 'ineq', 'fun': lambda w: (self.grasp_matrix)[1, :].dot(w) - max_fy}, \
                {'type': 'ineq', 'fun': lambda w: (self.grasp_matrix)[2, :].dot(w) - 0.5}, \
                {'type': 'ineq', 'fun': lambda w: -np.sum(w) + (1.-self.differ)},\
                {'type': 'ineq', 'fun': lambda w: np.sum(w) - (1.+self.differ)})

        sol = minimize(self.objective, self.start_wrench_, method='SLSQP', bounds=bs, constraints=cons)
        #print(sol)
        #print(self.objective(sol.x))
        return sol

    def analysis_quality(self):
        sol = self.evaluation_qp()
        max_fx = self.fric_coef * self.max_force / (3. ** 0.5)
        max_fy = self.fric_coef * self.max_force / (3. ** 0.5)
        min_fx = -self.fric_coef * self.max_force / (3. ** 0.5)
        min_fy = -self.fric_coef * self.max_force / (3. ** 0.5)
        min_fz = 0.5
        max_sum = 1. + self.differ
        min_sum = 1. - self.differ
        extra_dis = 0.

        cur_sum = np.sum(sol.x)
        if cur_sum < min_sum:
            extra_dis = extra_dis + 10 * np.absolute((cur_sum - min_sum) / (max_sum - min_sum))
        elif cur_sum > max_sum:
            extra_dis = extra_dis + 10 * np.absolute((cur_sum - max_sum) / (max_sum - min_sum))

        cur_fx = self.grasp_matrix[0, :].dot(sol.x)
        if cur_fx > max_fx:
            extra_dis = extra_dis + 10 * np.absolute((cur_fx - max_fx) / (max_fx - min_fx))
        elif cur_fx < min_fx:
            extra_dis = extra_dis + 10 * np.absolute((cur_fx - min_fx) / (max_fx - min_fx))

        cur_fy = self.grasp_matrix[1, :].dot(sol.x)
        if cur_fy > max_fy:
            extra_dis = extra_dis + 10 * np.absolute((cur_fy - max_fy) / (max_fy - min_fy))
        elif cur_fy < min_fx:
            extra_dis = extra_dis + 10 * np.absolute((cur_fy - min_fy) / (max_fy - min_fy))

        return True, sol, sol.fun + extra_dis


class DexterousVacuumPoint(Grasp):
    """ Defines a grasp point and axis for dexterous vacuum gripper in 3D space (5 DOF)
    """
    def __init__(self, center, radius, graspble):
        """
        :param center:
        :param radius:
        :param graspble:
        """
        #center, axis = VacuumPoint.params_from_configuration(configuration)
        self.center = center
        self.radius = radius
        self.graspble = graspble


    @property
    def center(self):
        return self.center

    @staticmethod
    def find_projection(obj, curr_loc, direction, max_projection, num_samples, vis=False):
        """Finds the point of contact when shooting a direction ray from curr_loc.
        Params:
            curr_loc - numpy 3 array of the starting point in obj frame
            direction - normalized numpy 3 array, direction to look for contact
            max_projection - float maximum amount to search forward for a contact (meters)
            num_samples - float number of samples when finding contacts
        Returns:
            found - True if projection contact is found
            projection_contact - Contact3D instance
        """
        # get start of projection
        line_of_action = ParallelJawPtGrasp3D.create_line_of_action(
            curr_loc, direction, max_projection, obj, num_samples)
        found, projection_contact = ParallelJawPtGrasp3D.find_contact(
            line_of_action, obj, vis=vis)
        '''
        if vis:
            ax = plt.gca(projection = '3d')
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
        '''
        return found, projection_contact

    @staticmethod
    def crop_surface_grasp(contact_pt, direction, u1, u2, width_win, depth_win,
                           flg_exclude_opposite=True):
        """
        FUNCTION: Crop out point around gripper and transform them from WCS to GCS

        :param contact_pt:                  obj. contact point
        :param direction:                   grasp axis
        :param u1:                          x-axis of GCS
        :param u2:                          y-axis of GCS
        :param width_win:                   width of crop window
        :param depth_win:                   depth of crop window
        :param flg_exclude_opposite:        whether excludet the opposite points
        :return:
            surface point of object after rotation in GCS
            surface point of grasp cropped from object (GCS)
        """
        '''
        https://blog.csdn.net/jc_laoshu/article/details/69657579
        |x'|   |u_x  u_y  u_z  -(x0*u_x + y0*u_y + z0*u_z)|   |x|
        |y'| = |v_x  v_y  v_z  -(x0*v_x + y0*v_y + z0*v_z)| * |y|
        |z'|   |n_x  n_y  n_z  -(x0*n_x + y0*n_y + z0*n_z)|   |z|
        |1 |   | 0    0    0                1             |   |1|
        '''
        pt_zero = 1.0 * contact_pt.point
        surface_obj = np.ones([contact_pt.graspable.mesh.vertices.shape[0], 4], dtype=np.float64)
        surface_obj[:, 0:3] = np.copy(contact_pt.graspable.mesh.vertices)
        # surface_obj = np.ones([contact_pt.graspable.surface_points.shape[0], 4], dtype=np.float64)
        # surface_obj[:, 0:3] = np.copy(contact_pt.graspable.surface_points)

        surface_obj = math_robot.transfer_CS(u1, u2, direction, pt_zero, surface_obj)

        normals_obj = np.ones([contact_pt.graspable.mesh.normals.shape[0], 4], dtype=np.float64)
        normals_obj[:, 0:3] = np.copy(contact_pt.graspable.mesh.normals)
        # normals_obj = math_robot.transfer_CS(u1, u2, direction, pt_zero, normals_obj)
        normals_obj = math_robot.transfer_CS(u1, u2, direction, np.zeros(3), normals_obj)

        x1 = surface_obj[:, 0] > -0.5 * width_win - 1e-5
        x2 = surface_obj[:, 0] < 0.5 * width_win + 1e-5
        y1 = surface_obj[:, 1] > -0.5 * width_win
        y2 = surface_obj[:, 1] < 0.5 * width_win
        z1 = surface_obj[:, 2] > -2.0 * depth_win
        z2 = surface_obj[:, 2] < depth_win

        pt_surface_crop = np.logical_and(x1, x2)
        pt_surface_crop = np.logical_and(pt_surface_crop, y1)
        pt_surface_crop = np.logical_and(pt_surface_crop, y2)
        pt_surface_crop = np.logical_and(pt_surface_crop, z1)
        pt_surface_crop = np.logical_and(pt_surface_crop, z2)
        ind_surface_crop = np.argwhere(pt_surface_crop[:]==True)
        ind_surface_crop = np.squeeze(ind_surface_crop)
        # ind = surface_obj[:, 0].argsort()
        surface_crop = 1.0 * surface_obj[ind_surface_crop, :]
        normals_crop = 1.0 * normals_obj[ind_surface_crop, :]
        # exclude the opposite points
        if flg_exclude_opposite:
            ind_surface = np.argwhere(normals_crop[:, 2] < 0.0)
            ind_surface = np.squeeze(ind_surface)
            surface_crop = 1.0 * surface_crop[ind_surface, :]
            normals_crop = 1.0 * normals_crop[ind_surface, :]

        '''
        change the style to depth camera  * -1 - min_z
        '''
        # surface_grasp[:, 2] = -1 * surface_grasp[:, 2]
        # surface_grasp[:, 2] -= np.min(surface_grasp[:, 2])
        # print("test")
        # input()
        return surface_obj[:, 0:3], surface_crop[:, 0:3], normals_obj[:, 0:3], normals_crop[:, 0:3]

    @staticmethod
    def visualize_points(pts_project=None, map_project=None):
        """
        CUATION: Do not use this function to estimation grasp quality. It is only used for visualization
        :param pts_project:             3*c*r float array:          points in polar coodinate system
        :param map_project:             c*r int array:              0: none
                                                                    1: vacuum region
                                                                    2: contact but not
                                                                    3: non-contact
        :return:
               pts_p2p:                 n*6 float array:     start/end point of each line
        """
        pts1_p2p = []
        pts2_p2p = []
        for cnt_rad in range(0, map_project.shape[0]-1):
            for cnt_ang in range(0, map_project.shape[1]-1):
                # for the 1st row, map_project[0, :] = 1.0
                if cnt_rad == 1:
                    # Positions:            v_3
                    #               v_1     v_2
                    v1 = np.array([pts_project[0, cnt_rad, cnt_ang],
                                   pts_project[1, cnt_rad, cnt_ang],
                                   pts_project[2, cnt_rad, cnt_ang]])
                    v2 = np.array([pts_project[0, cnt_rad + 1, cnt_ang],
                                   pts_project[1, cnt_rad + 1, cnt_ang],
                                   pts_project[2, cnt_rad + 1, cnt_ang]])
                    v3 = np.array([pts_project[0, cnt_rad + 1, cnt_ang + 1],
                                   pts_project[1, cnt_rad + 1, cnt_ang + 1],
                                   pts_project[2, cnt_rad + 1, cnt_ang + 1]])
                    #       1
                    #   1   1
                    if map_project[cnt_rad + 1, cnt_ang] == 1 and map_project[cnt_rad + 1, cnt_ang + 1] == 1:
                        pts1_p2p.append(np.r_[v1, v2])
                        pts1_p2p.append(np.r_[v1, v3])
                        pts1_p2p.append(np.r_[v2, v3])
                    #       1              2
                    #   1   2    or    1   1
                    elif map_project[cnt_rad + 1, cnt_ang] == 2 and map_project[cnt_rad + 1, cnt_ang + 1] == 1:
                        # area_sub = 0.5 * area_sub
                        pts1_p2p.append(np.r_[v1, v2])
                        pts1_p2p.append(np.r_[v1, v3])
                        pts1_p2p.append(np.r_[v2, v3])
                    elif map_project[cnt_rad + 1, cnt_ang] == 1 and map_project[cnt_rad + 1, cnt_ang + 1] == 2:
                        pts1_p2p.append(np.r_[v1, v2])
                        pts1_p2p.append(np.r_[v1, v3])
                        pts1_p2p.append(np.r_[v2, v3])
                    #       2              2
                    #   1   2    or    1   3    or ...
                    else:
                        continue
                else:
                    # Positions:    v_3     v_4
                    #               v_1     v_2
                    v1 = np.array([pts_project[0, cnt_rad, cnt_ang],
                                   pts_project[1, cnt_rad, cnt_ang],
                                   pts_project[2, cnt_rad, cnt_ang]])
                    v2 = np.array([pts_project[0, cnt_rad + 1, cnt_ang],
                                   pts_project[1, cnt_rad + 1, cnt_ang],
                                   pts_project[2, cnt_rad + 1, cnt_ang]])
                    v3 = np.array([pts_project[0, cnt_rad, cnt_ang + 1],
                                   pts_project[1, cnt_rad, cnt_ang + 1],
                                   pts_project[2, cnt_rad, cnt_ang + 1]])
                    v4 = np.array([pts_project[0, cnt_rad + 1, cnt_ang + 1],
                                   pts_project[1, cnt_rad + 1, cnt_ang + 1],
                                   pts_project[2, cnt_rad + 1, cnt_ang + 1]])

                    # triangle 1: v_1 -> v_2 -> v_4
                    if map_project[cnt_rad, cnt_ang] > 0 and \
                            map_project[cnt_rad + 1, cnt_ang] > 0 and \
                            map_project[cnt_rad + 1, cnt_ang + 1] > 0:
                        #     1
                        # 1   1
                        if map_project[cnt_rad, cnt_ang] == 1 and \
                                map_project[cnt_rad + 1, cnt_ang] == 1 and \
                                map_project[cnt_rad + 1, cnt_ang + 1] == 1:
                            pts1_p2p.append(np.r_[v1, v2])
                            pts1_p2p.append(np.r_[v1, v4])
                            pts1_p2p.append(np.r_[v2, v4])
                        else:
                            pts2_p2p.append(np.r_[v1, v2])
                            pts2_p2p.append(np.r_[v1, v4])
                            pts2_p2p.append(np.r_[v2, v4])

                    # triangle 2: v_1 -> v_3 -> v_4
                    if map_project[cnt_rad, cnt_ang] > 0 and \
                            map_project[cnt_rad, cnt_ang + 1] > 0 and \
                            map_project[cnt_rad + 1, cnt_ang + 1] > 0:
                        # 1   1
                        # 1
                        if map_project[cnt_rad, cnt_ang] == 1 and \
                                map_project[cnt_rad, cnt_ang + 1] == 1 and \
                                map_project[cnt_rad + 1, cnt_ang + 1] == 1:
                            pts1_p2p.append(np.r_[v1, v3])
                            pts1_p2p.append(np.r_[v1, v4])
                            pts1_p2p.append(np.r_[v3, v4])
                        else:
                            pts2_p2p.append(np.r_[v1, v3])
                            pts2_p2p.append(np.r_[v1, v4])
                            pts2_p2p.append(np.r_[v3, v4])
        pts1_p2p = np.array(pts1_p2p)
        pts2_p2p = np.array(pts2_p2p)
        return pts1_p2p, pts2_p2p

    @staticmethod
    def grasp_matrix(pts_project=None, map_project=None, normals_project=None,
                     radius_gripper=0.02, radius_vacuum=0.01, P_air=None, coef_fric=0.5,
                     flg_adv_air_pressure=False, type_ap=1):
        """
        FUNCTION:
        :param pts_project:             3*c*r float array:          points in polar coodinate system
        :param map_project:             c*r int array:              0: none
                                                                    1: vacuum region
                                                                    2: contact but not in vacuum region
                                                                    3: non-contact
        :param normals_project          3*c*r float array:          surface normals of the points
        :param radius_gripper           float:
        :param radius_vacuum            float:                      radius of the region where air flow into the gripper
        :param pressure_air             float:                      air pressure between contact surface
        :param coef_fric                float:                      coefficient friction between the contact surface
        :param flg_adv_air_pressur      bool:                       if use a differenct way to calculate ap
        :param type_ap                  int:                        type 1, 2, 3, ...
        :return:
        """
        # Fv_max = P_max * np.pi * radius_gripper ** 2

        Area_vacuum = 0.0
        Area_rest = 0.0
        G_matrix = np.zeros([4, 1]).reshape([4, 1])
        for cnt_rad in range(0, map_project.shape[0]-1):
            for cnt_ang in range(0, map_project.shape[1]-1):
                # for the 1st row, map_project[0, :] = 1.0
                if cnt_rad == 1:
                    # Positions:            v_3
                    #               v_1     v_2
                    v1 = np.array([pts_project[0, cnt_rad, cnt_ang],
                                   pts_project[1, cnt_rad, cnt_ang],
                                   pts_project[2, cnt_rad, cnt_ang]])
                    v2 = np.array([pts_project[0, cnt_rad + 1, cnt_ang],
                                   pts_project[1, cnt_rad + 1, cnt_ang],
                                   pts_project[2, cnt_rad + 1, cnt_ang]])
                    v3 = np.array([pts_project[0, cnt_rad + 1, cnt_ang + 1],
                                   pts_project[1, cnt_rad + 1, cnt_ang + 1],
                                   pts_project[2, cnt_rad + 1, cnt_ang + 1]])
                    area_sub = math_robot.calculate_area_triangle(v1, v2, v3)
                    #       1
                    #   1   1
                    if map_project[cnt_rad + 1, cnt_ang] == 1 and map_project[cnt_rad + 1, cnt_ang + 1] == 1:
                        pass
                    #       1              2
                    #   1   2    or    1   1
                    elif map_project[cnt_rad + 1, cnt_ang] == 2 and map_project[cnt_rad + 1, cnt_ang + 1] == 1:
                        area_sub = 0.5 * area_sub
                    elif map_project[cnt_rad + 1, cnt_ang] == 1 and map_project[cnt_rad + 1, cnt_ang + 1] == 2:
                        area_sub = 0.5 * area_sub
                    #       2              2
                    #   1   2    or    1   3    or ...
                    else:
                        continue
                    Area_vacuum += area_sub
                    nv_1 = np.array([normals_project[0, cnt_rad, cnt_ang],
                                     normals_project[1, cnt_rad, cnt_ang],
                                     normals_project[2, cnt_rad, cnt_ang]]).reshape([1, 3])
                    nv_2 = np.array([normals_project[0, cnt_rad + 1, cnt_ang],
                                     normals_project[1, cnt_rad + 1, cnt_ang],
                                     normals_project[2, cnt_rad + 1, cnt_ang]]).reshape([1, 3])
                    nv_3 = np.array([normals_project[0, cnt_rad + 1, cnt_ang + 1],
                                     normals_project[1, cnt_rad + 1, cnt_ang + 1],
                                     normals_project[2, cnt_rad + 1, cnt_ang + 1]]).reshape([1, 3])
                    normal_tri = -1.0 * np.mean(np.r_[nv_1, nv_2, nv_3], axis=0)
                    normal_tri = normal_tri / np.linalg.norm(normal_tri)
                    n_x = np.array([normal_tri[0], 0.0, 0.0])
                    n_y = np.array([0.0, normal_tri[1], 0.0])
                    n_z = np.array([0.0, 0.0, normal_tri[2]])
                    f_x = area_sub * n_x + coef_fric * area_sub * (n_x / np.linalg.norm(n_x + n_y))
                    f_y = area_sub * n_y + coef_fric * area_sub * (n_y / np.linalg.norm(n_x + n_y))
                    f_z = area_sub * n_z
                    p_center = np.mean(np.array([v1, v2, v3]), axis=0)
                    p_arm = np.array([p_center[0], p_center[1], 0.0])
                    t_z = np.cross(p_arm, f_x + f_y)
                    G_matrix = np.c_[G_matrix, np.array([f_x[0], f_y[1], f_z[2], t_z[2]])]
                else:
                    # Positions:    v_3     v_4
                    #               v_1     v_2
                    v1 = np.array([pts_project[0, cnt_rad, cnt_ang],
                                   pts_project[1, cnt_rad, cnt_ang],
                                   pts_project[2, cnt_rad, cnt_ang]])
                    v2 = np.array([pts_project[0, cnt_rad + 1, cnt_ang],
                                   pts_project[1, cnt_rad + 1, cnt_ang],
                                   pts_project[2, cnt_rad + 1, cnt_ang]])
                    v3 = np.array([pts_project[0, cnt_rad, cnt_ang + 1],
                                   pts_project[1, cnt_rad, cnt_ang + 1],
                                   pts_project[2, cnt_rad, cnt_ang + 1]])
                    v4 = np.array([pts_project[0, cnt_rad + 1, cnt_ang + 1],
                                   pts_project[1, cnt_rad + 1, cnt_ang + 1],
                                   pts_project[2, cnt_rad + 1, cnt_ang + 1]])

                    # triangle 1: v_1 -> v_2 -> v_4
                    if map_project[cnt_rad, cnt_ang] > 0 and \
                            map_project[cnt_rad + 1, cnt_ang] > 0 and \
                            map_project[cnt_rad + 1, cnt_ang + 1] > 0:
                        nv_1 = np.array([normals_project[0, cnt_rad, cnt_ang],
                                         normals_project[1, cnt_rad, cnt_ang],
                                         normals_project[2, cnt_rad, cnt_ang]]).reshape([1, 3])
                        nv_2 = np.array([normals_project[0, cnt_rad + 1, cnt_ang],
                                         normals_project[1, cnt_rad + 1, cnt_ang],
                                         normals_project[2, cnt_rad + 1, cnt_ang]]).reshape([1, 3])
                        nv_4 = np.array([normals_project[0, cnt_rad + 1, cnt_ang + 1],
                                         normals_project[1, cnt_rad + 1, cnt_ang + 1],
                                         normals_project[2, cnt_rad + 1, cnt_ang + 1]]).reshape([1, 3])
                        area_sub = math_robot.calculate_area_triangle(v1, v2, v4)
                        #     1
                        # 1   1
                        if map_project[cnt_rad, cnt_ang] == 1 and \
                                map_project[cnt_rad + 1, cnt_ang] == 1 and \
                                map_project[cnt_rad + 1, cnt_ang + 1] == 1:
                            Area_vacuum += 1.0 * area_sub
                        else:
                            Area_rest += 1.0 * area_sub

                        normal_tri = -1 * np.mean(np.r_[nv_1, nv_2, nv_4], axis=0)
                        normal_tri = normal_tri / np.linalg.norm(normal_tri)
                        n_x = np.array([normal_tri[0], 0.0, 0.0])
                        n_y = np.array([0.0, normal_tri[1], 0.0])
                        n_z = np.array([0.0, 0.0, normal_tri[2]])
                        f_x = area_sub * n_x + coef_fric * area_sub * (n_x / np.linalg.norm(n_x + n_y))
                        f_y = area_sub * n_y + coef_fric * area_sub * (n_y / np.linalg.norm(n_x + n_y))
                        f_z = area_sub * n_z
                        p_center = np.mean(np.array([v1, v2, v4]), axis=0)
                        p_arm = np.array([p_center[0], p_center[1], 0.0])
                        t_z = np.cross(p_arm, f_x + f_y)
                        G_matrix = np.c_[G_matrix, np.array([f_x[0], f_y[1], f_z[2], t_z[2]])]
                    # triangle 2: v_1 -> v_3 -> v_4
                    if map_project[cnt_rad, cnt_ang] > 0 and \
                            map_project[cnt_rad, cnt_ang + 1] > 0 and \
                            map_project[cnt_rad + 1, cnt_ang + 1] > 0:
                        nv_1 = np.array([normals_project[0, cnt_rad, cnt_ang],
                                         normals_project[1, cnt_rad, cnt_ang],
                                         normals_project[2, cnt_rad, cnt_ang]]).reshape([1, 3])
                        nv_3 = np.array([normals_project[0, cnt_rad, cnt_ang + 1],
                                         normals_project[1, cnt_rad, cnt_ang + 1],
                                         normals_project[2, cnt_rad, cnt_ang + 1]]).reshape([1, 3])
                        nv_4 = np.array([normals_project[0, cnt_rad + 1, cnt_ang + 1],
                                         normals_project[1, cnt_rad + 1, cnt_ang + 1],
                                         normals_project[2, cnt_rad + 1, cnt_ang + 1]]).reshape([1, 3])
                        area_sub = math_robot.calculate_area_triangle(v1, v3, v4)
                        # 1
                        # 1   1
                        if map_project[cnt_rad, cnt_ang] == 1 and \
                                map_project[cnt_rad, cnt_ang + 1] == 1 and \
                                map_project[cnt_rad + 1, cnt_ang + 1] == 1:
                            Area_vacuum += 1.0 * area_sub
                        else:
                            Area_rest += 1.0 * area_sub

                        normal_tri = -1 * np.mean(np.r_[nv_1, nv_3, nv_4], axis=0)
                        normal_tri = normal_tri / np.linalg.norm(normal_tri)
                        n_x = np.array([normal_tri[0], 0.0, 0.0])
                        n_y = np.array([0.0, normal_tri[1], 0.0])
                        n_z = np.array([0.0, 0.0, normal_tri[2]])
                        f_x = area_sub * n_x + coef_fric * area_sub * (n_x / np.linalg.norm(n_x + n_y))
                        f_y = area_sub * n_y + coef_fric * area_sub * (n_y / np.linalg.norm(n_x + n_y))
                        f_z = area_sub * n_z
                        p_center = np.mean(np.array([v1, v3, v4]), axis=0)
                        p_arm = np.array([p_center[0], p_center[1], 0.0])
                        t_z = np.cross(p_arm, f_x + f_y)
                        G_matrix = np.c_[G_matrix, np.array([f_x[0], f_y[1], f_z[2], t_z[2]])]


        Area_force_max = 1.0 * (Area_vacuum + Area_rest)
        if P_air == None:
            # make sure max suction force is 1
            P_air = 1.0 / Area_force_max
        Area_vacuum_max = np.pi * radius_vacuum ** 2
        if Area_vacuum > Area_vacuum_max:
            Area_vacuum = 1.0 * Area_vacuum_max
        P_real = P_air * (Area_vacuum / Area_vacuum_max)
        if flg_adv_air_pressure:
            if type_ap == 1:
                k_Area = Area_vacuum / Area_vacuum_max
                P_real = P_air * (k_Area * k_Area)
            elif type_ap == 2:
                pass
            else:
                pass

        G_matrix = P_real * G_matrix
        return G_matrix[:, 1::], P_air, Area_force_max * P_air

class DexterousQuality_Vacuum():
    def __init__(self, grasp_matrix=np.zeros([4, 1]), stable_wrench=None, max_force_vacuum=1.0,
                 coef_fric=0.5, radius_gripper=0.025, radius_vacuum=0.02,
                 differ=0.2, total_error=0.1):
        self.__grasp_matrix = 1.0 * grasp_matrix
        self.__grasp_matrix_QP = 1.0 * grasp_matrix

        self.__coef_fric = 1.0 * coef_fric
        self.__radius_gripper = 1.0 * radius_gripper
        self.__radius_vacuum = 1.0 * radius_vacuum
        self.__differ = 1.0 * differ
        [dof, dim] = np.shape(grasp_matrix)
        # start_params = 1.0 / dim
        self.__start_wrench = np.array(np.ones([dim, 1]), dtype=np.float64).reshape(dim, 1)
        self.__max_force_vacuum = max_force_vacuum
        self.__total_error = total_error
        if stable_wrench == None:
            self.__stable_wrench = np.array([0.0, 0.0, self.__max_force_vacuum, 0.0])
        else:
            self.__stable_wrench = 1.0 * stable_wrench

    def set_grasp_matrix(self, input):
        self.__grasp_matrix = 1.0 * np.copy(input)

    def set_stable_wrench(self, input):
        self.__stable_wrench = 1.0 * np.copy(input)

    def set_coef_fric(self, input):
        self.__coef_fric = 1.0 * input

    def set_radius_gripper(self, input):
        self.__radius_gripper = 1.0 * input

    def set_radius_vacuum(self, input):
        self.__radius_vacuum = 1.0 * input

    def set_differ(self, input):
        self.__differ = 1.0 * input

    def desample_grasp_matrix(self, dim):
        tmp_matrix = np.copy(self.grasp_matrix)
        tmp_matrix = tmp_matrix.T
        np.random.shuffle(tmp_matrix)
        tmp_matrix = tmp_matrix.T
        if dim < self.grasp_matrix.shape[1]:
            self.__grasp_matrix_QP = np.zeros([4, dim])
            ind_c = np.linspace(0, self.grasp_matrix.shape[1], dim + 1).astype(np.int)
            for i in range(0, dim):
                self.__grasp_matrix_QP[:, i] = np.sum(tmp_matrix[:, ind_c[i]:ind_c[i+1]], axis=1)
            self.__start_wrench = np.array(np.ones([dim, 1]), dtype=np.float64).reshape(dim, 1)
        else:
            dim = self.grasp_matrix.shape[1]
            self.__grasp_matrix_QP = np.copy(tmp_matrix)
            self.__start_wrench = np.array(np.ones([dim, 1]), dtype=np.float64).reshape(dim, 1)

    @property
    def grasp_matrix(self):
        return 1.0 * self.__grasp_matrix

    @property
    def stable_wrench(self):
        return 1.0 * self.__stable_wrench

    @property
    def coef_fric(self):
        return 1.0 * self.__coef_fric

    @property
    def radius_gripper(self):
        return 1.0 * self.__radius_gripper

    @property
    def radius_vacuum(self):
        return 1.0 * self.__radius_vacuum

    @property
    def differ(self):
        return 1.0 * self.__differ

    @property
    def start_wrench(self):
        return 1.0 * self.__start_wrench

    @property
    def max_force_vacuum(self):
        return 1.0 * self.__max_force_vacuum

    @property
    def total_error(self):
        return 1.0 * self.__total_error

    @property
    def g_matrix_QP(self):
        return 1.0 * self.__grasp_matrix_QP

    def objective(self, w):
        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h
        facet = 1.0 * self.g_matrix_QP
        offset = 1.0 * self.__stable_wrench
        dives = facet.dot(w) - offset
        dis = np.linalg.norm(dives) ** 2
        return dis

    @property
    def evaluation_qp(self):
        avg_weight = 1.0
        b = [avg_weight * (1. - self.differ), avg_weight * (1. + self.differ)]
        bs = []
        # bs = [b, b, b, b, b, b, b, b]
        for i in range(0, np.shape(self.g_matrix_QP)[1]):
            bs.append(list.copy(b))
        sum_w = float(self.g_matrix_QP.shape[0])

        max_fx = self.coef_fric * self.max_force_vacuum / (3. ** 0.5)
        max_fy = self.coef_fric * self.max_force_vacuum / (3. ** 0.5)
        max_fz = 1.0 * self.max_force_vacuum
        max_tz = self.radius_gripper * self.coef_fric * self.max_force_vacuum / (3. ** 0.5)
        ''''''
        cons = [{'type': 'ineq', 'fun': lambda w: (-self.g_matrix_QP)[0, :].dot(w) + max_fx}, \
                {'type': 'ineq', 'fun': lambda w: (self.g_matrix_QP)[0, :].dot(w) - max_fx}, \
                {'type': 'ineq', 'fun': lambda w: (-self.g_matrix_QP)[1, :].dot(w) + max_fy}, \
                {'type': 'ineq', 'fun': lambda w: (self.g_matrix_QP)[1, :].dot(w) - max_fy}, \
                {'type': 'ineq', 'fun': lambda w: (-self.g_matrix_QP)[2, :].dot(w) + 0}, \
                {'type': 'ineq', 'fun': lambda w: (self.g_matrix_QP)[2, :].dot(w) - max_fz}, \
                {'type': 'ineq', 'fun': lambda w: (-self.g_matrix_QP)[3, :].dot(w) + max_tz}, \
                {'type': 'ineq', 'fun': lambda w: (self.g_matrix_QP)[3, :].dot(w) - max_tz}, \
                {'type': 'ineq', 'fun': lambda w: -np.sum(w) + (1. - self.total_error) * sum_w}, \
                {'type': 'ineq', 'fun': lambda w: np.sum(w) - (1. + self.total_error) * sum_w}]
        '''
        sol = minimize(self.objective, self.start_wrench, method='SLSQP', bounds=bs, constraints=cons,
                       options={'maxiter':150})
        '''
        sol = minimize(self.objective, self.start_wrench, method='SLSQP', bounds=bs, constraints=cons)

        # print(sol)
        # print(self.objective(sol.x))
        return sol

    def analysis_quality(self, flg_desampling=False, num_dim=100, repeat=1):
        """
        :param flg_desampling:      bool:           if desample the grasp matrix to speed up
        :param num_dim:             int:            dimension after desampling
        :param repeat:              int:            repeat times to get best result
        :return:
        """

        logging.info('SLSQP start')
        tmp_dists = np.zeros(repeat)
        lst_sol = []
        for i in range(0, repeat):
            if flg_desampling:
                self.desample_grasp_matrix(num_dim)
            sol = self.evaluation_qp
            tmp_dists[i] = 1.0 * sol.fun
            lst_sol.append(sol)
        ind_min = np.argmin(tmp_dists)
        '''
        max_fx = self.coef_fric * self.max_force_vacuum / (3. ** 0.5)
        max_fy = self.coef_fric * self.max_force_vacuum / (3. ** 0.5)
        min_fx = -self.coef_fric * self.max_force_vacuum / (3. ** 0.5)
        min_fy = -self.coef_fric * self.max_force_vacuum / (3. ** 0.5)
        min_fz = 0.0
        '''
        sum_w = float(self.g_matrix_QP.shape[1])
        max_sum = (1. + self.total_error) * sum_w
        min_sum = (1. - self.total_error) * sum_w
        cur_sum = np.sum(lst_sol[ind_min].x)
        dist = 1.0 * lst_sol[ind_min].fun
        if min_sum < cur_sum < max_sum:
            flg_in = True
        else:
            dist += 4 + np.random.rand()
            flg_in = False
        quality = np.exp(-1.0 * dist)

        return flg_in, lst_sol[ind_min], 1.0*dist, 1.0*quality



class ChameleonTongueContact(Grasp):
    def __init__(self, center, radius, graspble):
        pass


class ChameleonTongue_Quality():
    def __init__(self):
        pass


