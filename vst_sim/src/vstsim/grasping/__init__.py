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
Modified by: Hui Zhang
Email      : hui.zhang@kuleuven.be
Date       : 23/02/2020   09:53
"""

from vstsim.grasping.contacts import Contact3D, SurfaceWindow
from vstsim.grasping.graspable_object import GraspableObject, GraspableObject3D
from vstsim.grasping.grasp import Grasp, PointGrasp, ParallelJawPtGrasp3D, \
     VacuumPoint, GraspQuality_Vacuum, DexterousVacuumPoint, DexterousQuality_Vacuum, \
     ChameleonTongueContact, ChameleonTongue_Quality
######################################################################
from vstsim.grasping.gripper import RobotGripper
from vstsim.grasping.grasp_quality_config import GraspQualityConfig, QuasiStaticGraspQualityConfig, \
    RobustQuasiStaticGraspQualityConfig, GraspQualityConfigFactory
from vstsim.grasping.quality import PointGraspMetrics3D
from vstsim.grasping.random_variables import GraspableObjectPoseGaussianRV, ParallelJawGraspPoseGaussianRV, \
    ParamsGaussianRV
from vstsim.grasping.robust_grasp_quality import QuasiStaticGraspQualityRV
from vstsim.grasping.grasp_quality_function import GraspQualityResult, GraspQualityFunction, \
    QuasiStaticQualityFunction, RobustQuasiStaticQualityFunction, GraspQualityFunctionFactory

try:
    from vstsim.grasping.collision_checker import OpenRaveCollisionChecker, GraspCollisionChecker
except Exception:
    print('Unable to import OpenRaveCollisionChecker and GraspCollisionChecker! Likely due to missing '
          'OpenRave dependency.')
    print('Install OpenRave 0.9 from source if required. Instructions can be found at '
          'http://openrave.org/docs/latest_stable/coreapihtml/installation_linux.html')

"""
CAUTION: The order of importing is very important
"""
from vstsim.grasping.grasp_info import GraspInfo, GraspInfo_TongueGrasp

from vstsim.grasping.grasp_sampler import GraspSampler, UniformGraspSampler, GaussianGraspSampler, \
    AntipodalGraspSampler, GpgGraspSampler, PointGraspSampler, GpgGraspSamplerPcl, VacuumGraspSampler, \
    DexterousVacuumGrasp, ChameleonTongueGrasp



__all__ = ['Contact3D', 'GraspableObject', 'GraspableObject3D', 'ParallelJawPtGrasp3D',
           ###############################
           'VacuumPoint', 'GraspQuality_Vacuum',
           'DexterousVacuumPoint', 'DexterousQuality_Vacuum',
           'ChameleonTongueContact', 'ChameleonTongue_Quality',
           'GraspInfo', 'GraspInfo_TongueGrasp',
           ###############################
           'Grasp', 'PointGrasp', 'RobotGripper', 'PointGraspMetrics3D',
           'GraspQualityConfig', 'QuasiStaticGraspQualityConfig', 'RobustQuasiStaticGraspQualityConfig',
           'GraspQualityConfigFactory',
           'GraspSampler', 'UniformGraspSampler', 'GaussianGraspSampler', 'AntipodalGraspSampler',
           'GpgGraspSampler', 'PointGraspSampler', 'GpgGraspSamplerPcl',
           ###########################################
           'VacuumGraspSampler', 'DexterousVacuumGrasp', 'ChameleonTongueGrasp', 
           ###############################################
           'GraspableObjectPoseGaussianRV', 'ParallelJawGraspPoseGaussianRV', 'ParamsGaussianRV',
           'QuasiStaticGraspQualityRV', 'RobustPointGraspMetrics3D',
           'GraspQualityResult', 'GraspQualityFunction', 'QuasiStaticQualityFunction',
           'RobustQuasiStaticQualityFunction', 'GraspQualityFunctionFactory',
           'OpenRaveCollisionChecker', 'GraspCollisionChecker', ]

# module name spoofing for correct imports
from vstsim.grasping import grasp
import sys
sys.modules['vstsim.grasp'] = grasp
