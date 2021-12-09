#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hui Zhang
# E-mail     : hui.zhang@kuleuven.be
# Description:
# Date       : 08/10/2020 15:46
# File Name  : math_robot.py

from scipy.spatial.transform import Rotation
import numpy as np
import math
"""
    This file defines some common functions for robotic systems
"""


def rot_along_axis(a_x=np.array([1.0, 0.0, 0.0]), angle=0.0,
                   pts=np.ones([1, 3])):
    """
    FUNCTION: rotate a point or vector along with any vector
    CAUTION: rotation direction -> right hand principle & anticlockwise
    :param a_x:         3*1 float array:        the rotation axis
    :param angle:       float:                  rotation angle (metric:rad)
    :param pts:         n*3 float array:        the rotated points
    :return:
    """
    '''
    https://en.wikipedia.org/wiki/Rotation_matrix
    https://zhuanlan.zhihu.com/p/56587491
    point P(x0,y0,z0) rotates angle theta along with the axis (x,y,z), than the position of P should be
                          |cos + x^2(1-cos)    xy(1-cos) - zsin    xz(1-cos) + ysin|
    P' = (x0',y0',z0')T = |xy(1-cos) + zsin    cos + y^2(1-cos)    yz(1-cos) - xsin| * (x0,y0,z0)T
                          |zx(1-cos) - ysin    zy(1-cos) + xsin    cos + z^2(1-cos)|
    CAUTION: (x,y,z) must be a normalized vector
    '''
    a_x = a_x / np.linalg.norm(a_x)
    cos = math.cos(angle)
    sin = math.sin(angle)
    [x, y, z] = a_x

    tmp_x = np.array([cos + x ** 2 * (1 - cos), x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin]).reshape(1, 3)
    tmp_y = np.array([x * y * (1 - cos) + z * sin, cos + y ** 2 * (1 - cos), y * z * (1 - cos) - x * sin]).reshape(1, 3)
    tmp_z = np.array([z * x * (1 - cos) - y * sin, z * y * (1 - cos) + x * sin, cos + z ** 2 * (1 - cos)]).reshape(1, 3)
    rot_matrix_t1 = np.r_[tmp_x, tmp_y, tmp_z]

    pts_ar = np.dot(rot_matrix_t1, pts.T).T

    return pts_ar


def transfer_CS(a_x=np.array([1.0, 0.0, 0.0]),
                a_y=np.array([0.0, 1.0, 0.0]),
                a_z=np.array([0.0, 0.0, 1.0]),
                o_n=np.zeros(3), pts=np.ones([1, 4])):
    """
    FUNCTION: Transfer points from coordinate system A to coordinate system B
    :param a_x:         3*1 float array:        x-axis of the new CS    MUST right-hand principle
    :param o_n:         3*1 float array:        zero point of the new CS
    :param pts:         n*4 or n*3 float array:        the rotated points
    :return:
                        n*4 float array:
    """
    '''
    https://blog.csdn.net/jc_laoshu/article/details/69657579
    point P(x,y,z) rotates angle theta along with the axis (x,y,z), than the position of P should be
                         |u_x    u_y    u_z    -1 * (x0 * u_x + y0 * u_y + z0 * u_z)|
    P' = (x',y',z',1)T = |v_x    v_z    v_z    -1 * (x0 * v_x + y0 * v_y + z0 * v_z)| * (x,y,z,1)T
                         |n_x    n_y    n_z    -1 * (x0 * n_x + y0 * n_y + z0 * n_z)|
                         |0      0       0      1                                   |
    '''
    a_x = a_x / np.linalg.norm(a_x)
    a_y = a_y / np.linalg.norm(a_y)
    a_z = a_z / np.linalg.norm(a_z)
    if pts.size <= 4:
        tmp_pt = np.ones(4)
        tmp_pt[0:3] = pts[0:3]
        pts = tmp_pt
    else:
        if pts.shape[1] == 3:
            tmp_array = np.ones([pts.shape[0], 1])
            pts = np.c_[pts, tmp_array]

    T_A2B = np.zeros([4, 4])
    T_A2B[0, 0:3] = 1.0 * a_x
    T_A2B[1, 0:3] = 1.0 * a_y
    T_A2B[2, 0:3] = 1.0 * a_z

    [x0, y0, z0] = o_n
    [u_x, u_y, u_z] = a_x
    [v_x, v_y, v_z] = a_y
    [n_x, n_y, n_z] = a_z
    T_A2B[0, 3] = -1 * (x0 * u_x + y0 * u_y + z0 * u_z)
    T_A2B[1, 3] = -1 * (x0 * v_x + y0 * v_y + z0 * v_z)
    T_A2B[2, 3] = -1 * (x0 * n_x + y0 * n_y + z0 * n_z)
    T_A2B[3, 3] = 1.0

    pts_as = T_A2B.dot(pts.T).T
    return pts_as


def transfer_CS_reverse(a_x=np.array([1.0, 0.0, 0.0]),
                        a_y=np.array([0.0, 1.0, 0.0]),
                        a_z=np.array([0.0, 0.0, 1.0]),
                        o_n=np.zeros(3), pts=np.ones([1, 4])):
    """
    FUNCTION: This is the reverse operation of the function "transfer_CS"
    :param a_x:         3*1 float array:        x-axis of the new CS
    :param o_n:         3*1 float array:        zero point of the new CS
    :param pts:         n*4 or n*3 float array:        the rotated points
    :return:
                        n*4 float array:
    """

    a_x = a_x / np.linalg.norm(a_x)
    a_y = a_y / np.linalg.norm(a_y)
    a_z = a_z / np.linalg.norm(a_z)
    if pts.size <= 4:
        tmp_pt = np.ones(4)
        tmp_pt[0:3] = pts[0:3]
        pts = tmp_pt
    else:
        if pts.shape[1] == 3:
            tmp_array = np.ones([pts.shape[0], 1])
            pts = np.c_[pts, tmp_array]

    T_A2B = np.zeros([4, 4])
    T_A2B[0, 0:3] = 1.0 * a_x
    T_A2B[1, 0:3] = 1.0 * a_y
    T_A2B[2, 0:3] = 1.0 * a_z

    [x0, y0, z0] = o_n
    [u_x, u_y, u_z] = a_x
    [v_x, v_y, v_z] = a_y
    [n_x, n_y, n_z] = a_z
    T_A2B[0, 3] = -1 * (x0 * u_x + y0 * u_y + z0 * u_z)
    T_A2B[1, 3] = -1 * (x0 * v_x + y0 * v_y + z0 * v_z)
    T_A2B[2, 3] = -1 * (x0 * n_x + y0 * n_y + z0 * n_z)
    T_A2B[3, 3] = 1.0

    T_B2A = np.linalg.inv(T_A2B)

    pts_as = T_B2A.dot(pts.T).T
    return pts_as


def calculate_area_triangle(v_1=np.array([1.0, 0.0, 0.0]),
                            v_2=np.array([0.0, 1.0, 0.0]),
                            v_3=np.array([0.0, 0.0, 1.0])):
    """
    FUNCTION: calculate the area of triangle
    :param v_1:         3*1 float array:        vertex 1
    :param v_2:         3*1 float array:
    :param v_3:         3*1 float array:
    :return:
    """
    vc_1 = v_1 - v_2
    vc_2 = v_1 - v_3
    area = 0.5 * np.linalg.norm(np.cross(vc_1, vc_2))

    return area


def calculate_Euler_ZXY_from_vectors(a_x=np.array([1.0, 0.0, 0.0]),
                                     a_y=np.array([0.0, 1.0, 0.0]),
                                     a_z=np.array([0.0, 0.0, 1.0]),
                                     o_n=np.zeros(3)):
    """
    Calculate the Euler angle to transfer into new coordinate system

    ZXY order  -> Tait-Bryan Angle
    :param a_x:         3*1 float array:        x-axis of the new CS
    :param a_y:
    :param a_z:
    :param o_n:         3*1 float array:        zero point of the new CS
    :return:
                        3 float array:          the angle Z, X, Y       metric: radian

    https://en.wikipedia.org/wiki/Euler_angles
    https://zh.wikipedia.org/wiki/%E6%AC%A7%E6%8B%89%E8%A7%92
    https://zhuanlan.zhihu.com/p/45404840
    """
    # step 1: build the RT(Rotation & transformation) matrix:
    """
                         |u_x    u_y    u_z    -1 * (x0 * u_x + y0 * u_y + z0 * u_z)|
    P' = (x',y',z',1)T = |v_x    v_z    v_z    -1 * (x0 * v_x + y0 * v_y + z0 * v_z)| * (x,y,z,1)T
                         |n_x    n_y    n_z    -1 * (x0 * n_x + y0 * n_y + z0 * n_z)|
                         |0      0       0      1                                   |
                            
                         |c1*c3+s1*s2*s3     s1*s2*c3-c1*s3      s1*c2   x0*m00+y0*m01+z0*m02|
                       = |   c2*s3               c2*c3             -s2   x0*m10+y0*m11+z0*m12| * (x,y,z,1)T
                         |c1*s2*s3-s1*c3     s1*s3+c1*s2*c3      c1*c2   x0*m20+y0*m21+z0*m22|
                         |   0                   0                   0          1            |
    m(i, j) is the value on the i-th row, j-th coloum value of RT matrix 
                         
    """
    T_A2B = np.zeros([4, 4])
    T_A2B[0, 0:3] = 1.0 * a_x
    T_A2B[1, 0:3] = 1.0 * a_y
    T_A2B[2, 0:3] = 1.0 * a_z

    [x0, y0, z0] = o_n
    [u_x, u_y, u_z] = a_x
    [v_x, v_y, v_z] = a_y
    [n_x, n_y, n_z] = a_z
    T_A2B[0, 3] = -1 * (x0 * u_x + y0 * u_y + z0 * u_z)
    T_A2B[1, 3] = -1 * (x0 * v_x + y0 * v_y + z0 * v_z)
    T_A2B[2, 3] = -1 * (x0 * n_x + y0 * n_y + z0 * n_z)
    T_A2B[3, 3] = 1.0

    # step 2: if R_x = +_pi/2 degree -> Gimbal Lock 万向锁
    """
    if R_x = -pi/2 -> s2 = -1, c2 = 0
             |c1*c3-s1*s3   -s1*s3-c1*s3    0|   |cos(R_x+R_z)      -sin(R_x+R_z)   0|
       M_R = |     0              0         1| = |      0|                  0       1|
             |-c1*s3-s1*c3   s1*s3-c1*c3    0|   |-sin(R_x+R_z)     -cos(R_x+R_z)   0|
             
       R_y+R_z = arctan(R_y+R_z) = arctan(2.0*m01/m21)
       
       Let R_y = R_z = 0.5 * arctan2(m01/m21)
    #################################################   
    #################################################   
    if R_x = pi/2 -> s2 = 1, c2 = 0
             |c1*c3+s1*s3   s1*s3-c1*s3    0|   |cos(R_x-R_z)      sin(R_x-R_z)   0|
       M_R = |     0              0        1| = |      0|                 0       1|
             |c1*s3-s1*c3   s1*s3+c1*c3    0|   |-sin(R_x-R_z)     cos(R_x-R_z)   0|
             
       R_y-R_z = arctan(R_y-R_z) = arctan2(m01/m21)
       
       Let R_y = arctan(m01/m21), R_z = 0   
    #################################################   
    #################################################      
    common case:
        R_y = arctan2(m02, m22)
        R_x = arcsin(-m12)
        R_z = arctan2(m10, m11)
    """
    if T_A2B[1, 2] == 1.0:      # sin() == -1.0
        R_x = -0.5 * np.pi
        if T_A2B[2, 1] == 0.0 and T_A2B[0, 1] > 0.0:
            R_y = 0.25 * np.pi
            R_z = 0.25 * np.pi
        elif T_A2B[2, 1] == 0.0 and T_A2B[0, 1] < 0.0:
            R_y = -0.25 * np.pi
            R_z = -0.25 * np.pi
        else:
            R_y = 0.5 * np.arctan2(T_A2B[0, 1], T_A2B[2, 1])
            R_z = 0.5 * np.arctan2(T_A2B[0, 1], T_A2B[2, 1])
        return np.array([R_z, R_x, R_y])

    elif T_A2B[1, 2] == -1.0:   # sin() == 1.0
        R_x = 0.5 * np.pi
        if T_A2B[2, 1] == 0.0 and T_A2B[0, 1] > 0.0:
            R_y = 0.5 * np.pi
            R_z = 0.0
        elif T_A2B[2, 1] == 0.0 and T_A2B[0, 1] < 0.0:
            R_y = -0.5 * np.pi
            R_z = 0.0
        else:
            R_y = 0.5 * np.arctan2(T_A2B[0, 1], T_A2B[2, 1])
            R_z = 0.0
        return np.array([R_z, R_x, R_y])

    else:
        R_x = np.arcsin(-1.0 * T_A2B[1, 2])

        if T_A2B[2, 2] == 0.0 and T_A2B[0, 2] < 0.0:
            R_y = -0.5 * np.pi
        elif T_A2B[2, 2] == 0.0 and T_A2B[0, 2] > 0.0:
            R_y = 0.5 * np.pi
        else:
            R_y = np.arctan2(T_A2B[0, 2], T_A2B[2, 2])

        if T_A2B[1, 1] == 0.0 and T_A2B[1, 0] < 0.0:
            R_z = -0.5 * np.pi
        elif T_A2B[1, 1] == 0.0 and T_A2B[1, 0] > 0.0:
            R_z = 0.5 * np.pi
        else:
            R_z = np.arctan2(T_A2B[1, 0],T_A2B[1, 1])

        return np.array([R_z, R_x, R_y])


def calculate_Euler_from_vectors(a_x=np.array([1.0, 0.0, 0.0]),
                                 a_y=np.array([0.0, 1.0, 0.0]),
                                 a_z=np.array([0.0, 0.0, 1.0]),
                                 o_n=np.zeros(3), order='xyz'):
    """
    Calculate the Euler angle to transfer into new coordinate system

    ZXY order  -> Tait-Bryan Angle
    :param a_x:         3*1 float array:        x-axis of the new CS
    :param a_y:
    :param a_z:
    :param o_n:         3*1 float array:        zero point of the new CS
    :param order:       str:                    order of rotation axes: 'xyz', 'zxy', 'yzx', ...
    :return:
                        3 float array:          the angle Z, X, Y       metric: radian

    https://en.wikipedia.org/wiki/Euler_angles
    https://zh.wikipedia.org/wiki/%E6%AC%A7%E6%8B%89%E8%A7%92
    https://zhuanlan.zhihu.com/p/45404840
    """
    # step 1: build the RT(Rotation & transformation) matrix:
    """
                         |u_x    u_y    u_z    -1 * (x0 * u_x + y0 * u_y + z0 * u_z)|
    P' = (x',y',z',1)T = |v_x    v_z    v_z    -1 * (x0 * v_x + y0 * v_y + z0 * v_z)| * (x,y,z,1)T
                         |n_x    n_y    n_z    -1 * (x0 * n_x + y0 * n_y + z0 * n_z)|
                         |0      0       0      1                                   |
    """
    T_A2B = np.zeros([4, 4])
    T_A2B[0, 0:3] = 1.0 * a_x
    T_A2B[1, 0:3] = 1.0 * a_y
    T_A2B[2, 0:3] = 1.0 * a_z

    [x0, y0, z0] = o_n
    [u_x, u_y, u_z] = a_x
    [v_x, v_y, v_z] = a_y
    [n_x, n_y, n_z] = a_z
    T_A2B[0, 3] = -1 * (x0 * u_x + y0 * u_y + z0 * u_z)
    T_A2B[1, 3] = -1 * (x0 * v_x + y0 * v_y + z0 * v_z)
    T_A2B[2, 3] = -1 * (x0 * n_x + y0 * n_y + z0 * n_z)
    T_A2B[3, 3] = 1.0

    # step 2: calculate the Euler angle
    r_matrix = Rotation.from_matrix(T_A2B[0:3, 0:3])
    euler = r_matrix.as_euler(order)
    return euler


def RT_matrix_to_Euler_ZXY(M_A2B):
    """
    from scipy.spatial.transform import Rotation
    """
    pass


def Euler_ZXY_to_R_matrix(Euler_ZXY=np.zeros(3)):
    """
    from scipy.spatial.transform import Rotation
    """
    pass


def quaternion_to_R_matrix(Quat=np.array([1.0, 0.0, 0.0, 0.0])):
    """
    from scipy.spatial.transform import Rotation
    """
    pass


def RT_matrix_to_quaternion(M_A2B):
    """
    from scipy.spatial.transform import Rotation
    """
    pass


def Euler_XYZ_to_quaternion(Euler_ZXY=np.zeros(3)):
    """
    from scipy.spatial.transform import Rotation
    """
    pass


def quaternion_to_Euler_XYZ(Quat=np.array([1.0, 0.0, 0.0, 0.0])):
    """
    from scipy.spatial.transform import Rotation
    """
    pass

def orthogonal_normals(direction=np.array([0.0, 0.0, 1.0]), align_axes=True, max_samples=1000):
    """Returns the direction vector and tangent vectors at a contact point.
    The direction vector defaults to the *inward-facing* normal vector at
    this contact.
    The direction and tangent vectors for a right handed coordinate frame.

    Parameters
    ----------
    direction : 3*float         direction to find orthogonal plane for
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
    # (u, s, v)
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

