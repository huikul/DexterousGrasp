# -*- coding: utf-8 -*-
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
Encapsulates data and operations on a 2D or 3D object that can be grasped
Author: Jeff Mahler
"""
"""
Modified by: Hui Zhang
Email      : hui.zhang@kuleuven.be
Date       : 26/02/2020 15:56    
"""
from abc import ABCMeta, abstractmethod

import copy
import logging
import numpy as np

import meshpy.mesh as m
import meshpy.sdf as s

import IPython
import matplotlib.pyplot as plt

from autolab_core import RigidTransform, SimilarityTransform
try:
    import pcl
except ImportError:
    logging.warning('Failed to import pcl!')

# class GraspableObject(metaclass=ABCMeta):
class GraspableObject:
    """ Encapsulates geometric structures for computing contact in grasping.
    
    Attributes
    ----------
    sdf : :obj:`Sdf3D`
        signed distance field, for quickly computing contact points
    mesh : :obj:`Mesh3D`
        3D triangular mesh to specify object geometry, should match SDF
    key : :obj:`str`
        object identifier, usually given from the database
    model_name : :obj:`str`
        name of the object mesh as a .obj file, for use in collision checking
    mass : float
        mass of the object
    convex_pieces : :obj:`list` of :obj:`Mesh3D`
        convex decomposition of the object geom for collision checking
    """
    __metaclass__ = ABCMeta

    def __init__(self, sdf, mesh, key='', model_name='', mass=1.0, convex_pieces=None):
        self.sdf_ = sdf
        self.mesh_ = mesh

        self.key_ = key
        self.model_name_ = model_name # for OpenRave usage, gross!
        self.mass_ = mass
        self.convex_pieces_ = convex_pieces
        ''''''
        surface_points, _ = self.sdf.surface_points(grid_basis=False)
        np.random.shuffle(surface_points)
        # surface_points = surface_points[:min(2000, len(surface_points))]
        # tmp_array = np.ones([surface_points.shape[0], 4], dtype=np.float64)
        tmp_array = np.ones(surface_points.shape, dtype=np.float64)
        for i, curr_point in enumerate(surface_points):
            tmp_array[i, 0:3] = curr_point
        self.surface_points_ = tmp_array

        self.min_pos_ = np.min(tmp_array, axis=0)
        self.max_pos_ = np.max(tmp_array, axis=0)
        # print("min_pos_: ", self.min_pos_)
        # print("max_pos_: ", self.max_pos_)
        self.size_ = np.max(self.max_pos_ - self.min_pos_)
        # print("size_: ", self.size_)
    @property
    def min_pos(self):
        return self.min_pos_
    @property
    def max_pos(self):
        return self.max_pos_
    @property
    def size(self):
        return self.size_
    @property
    def surface_points(self):
        return self.surface_points_
    @property
    def sdf(self):
        return self.sdf_

    @property
    def mesh(self):
        return self.mesh_

    @property
    def mass(self):
        return self.mass_

    @property
    def key(self):
        return self.key_

    @property
    def model_name(self):
        return self.model_name_

    @property
    def convex_pieces(self):
        return self.convex_pieces_

class GraspableObject3D(GraspableObject):
    """ 3D Graspable object for computing contact in grasping.
    
    Attributes
    ----------
    sdf : :obj:`Sdf3D`
        signed distance field, for quickly computing contact points
    mesh : :obj:`Mesh3D`
        3D triangular mesh to specify object geometry, should match SDF
    key : :obj:`str`
        object identifier, usually given from the database
    model_name : :obj:`str`
        name of the object mesh as a .obj file, for use in collision checking
    mass : float
        mass of the object
    convex_pieces : :obj:`list` of :obj:`Mesh3D`
        convex decomposition of the object geom for collision checking
    """
    def __init__(self, sdf, mesh, key='',
                 model_name='', mass=1.0,
                 convex_pieces=None):
        if not isinstance(sdf, s.Sdf3D):
            raise ValueError('Must initialize 3D graspable object with 3D sdf')
        if not isinstance(mesh, m.Mesh3D):
            raise ValueError('Must initialize 3D graspable object with 3D mesh')

        GraspableObject.__init__(self, sdf, mesh, key=key,
                                 model_name=model_name, mass=mass,
                                 convex_pieces=convex_pieces)

    def moment_arm(self, x):
        """ Computes the moment arm to a point x.

        Parameters
        ----------
        x : 3x1 :obj:`numpy.ndarray`
            point to get moment arm for
        
        Returns
        -------
        3x1 :obj:`numpy.ndarray`
        """
        return x - self.mesh.center_of_mass

    def rescale(self, scale):
        """ Rescales uniformly by a given factor.

        Parameters
        ----------
        scale : float
            the amount to scale the object

        Returns
        -------
        :obj:`GraspableObject3D`
            the graspable object rescaled by the given factor
        """
        stf = SimilarityTransform(scale=scale)
        sdf_rescaled = self.sdf_.rescale(scale)
        mesh_rescaled = self.mesh_.transform(stf)
        convex_pieces_rescaled = None
        if self.convex_pieces_ is not None:
            convex_pieces_rescaled = []
            for convex_piece in self.convex_pieces_:
                convex_piece_rescaled = convex_piece.transform(stf)
                convex_pieces_rescaled.append(convex_piece_rescaled)
        return GraspableObject3D(sdf_rescaled, mesh_rescaled, key=self.key,
                                 model_name=self.model_name, mass=self.mass,
                                 convex_pieces=convex_pieces_rescaled)

    def transform(self, delta_T):
        """ Transform by a delta transform.


        Parameters
        ----------
        delta_T : :obj:`RigidTransform`
            the transformation from the current reference frame to the alternate reference frame
        
        Returns
        -------
        :obj:`GraspableObject3D`
             graspable object trasnformed by the delta
        """
        sdf_tf = self.sdf_.transform(delta_T)
        mesh_tf = self.mesh_.transform(delta_T)
        convex_pieces_tf = None
        if self.convex_pieces_ is not None:
            convex_pieces_tf = []
            for convex_piece in self.convex_pieces_:
                convex_piece_tf = convex_piece.transform(delta_T)
                convex_pieces_tf.append(convex_piece_tf)
        return GraspableObject3D(sdf_tf, mesh_tf, key=self.key,
                                 model_name=self.model_name, mass=self.mass,
                                 convex_pieces=convex_pieces_tf)

    def surface_information(self, grasp, width, num_steps, plot=False, direction1=None, direction2=None):
        """ Returns the patches on this object for a given grasp.

        Parameters
        ----------
        grasp : :obj:`ParallelJawPtGrasp3D`
            grasp to get the patch information for
        width : float
            width of jaw opening
        num_steps : int
            number of steps
        plot : bool
            whether to plot the intermediate computation, for debugging
        direction1 : normalized 3x1 :obj:`numpy.ndarray`
            direction along which to compute the surface information for the first jaw, if None then defaults to grasp axis
        direction2 : normalized 3x1 :obj:`numpy.ndarray`
            direction along which to compute the surface information for the second jaw, if None then defaults to grasp axis
       
        Returns
        -------
        :obj:`list` of :obj:`SurfaceWindow`
             surface patches, one for each contact
       """
        contacts_found, contacts = grasp.close_fingers(self)#, vis=True)
        if not contacts_found:
            raise ValueError('Failed to find contacts')
        contact1, contact2 = contacts

        if plot:
            plt.figure()
            contact1.plot_friction_cone()
            contact2.plot_friction_cone()

            ax = plt.gca(projection = '3d')
            ax.set_xlim3d(0, self.sdf.dims_[0])
            ax.set_ylim3d(0, self.sdf.dims_[1])
            ax.set_zlim3d(0, self.sdf.dims_[2])

        window1 = contact1.surface_information(width, num_steps, direction=direction1)
        window2 = contact2.surface_information(width, num_steps, direction=direction2)
        return window1, window2, contact1, contact2

    '''
    def _find_projection(self, curr_loc, direction, max_projection, back_up, num_samples, vis=False):
        """Finds the point of contact when shooting a direction ray from curr_loc.
        Params:
            curr_loc - numpy 3 array of the starting point in obj frame
            direction - normalized numpy 3 array, direction to look for contact
            max_projection - float maximum amount to search forward for a contact (meters)
            back_up - float amount to back up before finding a contact (meters)
            num_samples - float number of samples when finding contacts
        Returns:
            found - True if projection contact is found
            projection_contact - Contact3D instance
        """
        # get start of projection
        projection_start = curr_loc - back_up * direction
        line_of_action = ParallelJawPtGrasp3D.create_line_of_action(
            projection_start, direction, (max_projection + back_up), self, num_samples
        )
        found, projection_contact = ParallelJawPtGrasp3D.find_contact(
            line_of_action, self, vis=vis
        )

        if vis:
            ax = plt.gca(projection = '3d')
            ax.set_xlim3d(0, self.sdf.dims_[0])
            ax.set_ylim3d(0, self.sdf.dims_[1])
            ax.set_zlim3d(0, self.sdf.dims_[2])

        return found, projection_contact
    '''

