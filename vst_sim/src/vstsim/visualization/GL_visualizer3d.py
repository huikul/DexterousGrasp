#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hui Zhang
# E-mail     : hui.zhang@kuleuven.be
# Description:
# Date       : 29/01/2021 15:56
# File Name  : GL_visualizer3d.py

import pygame
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np


class GL_Visualizer(object):
    def __init__(self):
        self.color_map = np.zeros([15, 3])
        self.color_map[0, :] = np.array([255, 255, 255]) / 255.0        # White
        self.color_map[1, :] = np.array([255, 0, 0]) / 255.0        # Red
        self.color_map[2, :] = np.array([60, 180, 75]) / 255.0      # Green
        self.color_map[3, :] = np.array([255, 225, 25]) / 255.0     # Yellow
        self.color_map[4, :] = np.array([0, 130, 200]) / 255.0      # Blue
        self.color_map[5, :] = np.array([245, 130, 48]) / 255.0     # Orange
        self.color_map[6, :] = np.array([145, 30, 180]) / 255.0     # Purple
        self.color_map[7, :] = np.array([70, 240, 240]) / 255.0     # Cyan
        self.color_map[8, :] = np.array([240, 50, 230]) / 255.0     # Magenta
        self.color_map[9, :] = np.array([210, 245, 60]) / 255.0     # Lime
        self.color_map[10, :] = np.array([250, 190, 190]) / 255.0    # Pink
        self.color_map[11, :] = np.array([0, 128, 128]) / 255.0     # Teal
        self.color_map[12, :] = np.array([128, 0, 0]) / 255.0       # Maroon
        self.color_map[13, :] = np.array([128, 128, 0]) / 255.0     # Olive
        self.color_map[14, :] = np.array([0, 0, 128]) / 255.0       # Navy

        self.vertices = None
        self.v_normals = None
        self.faces = None
        self.material = None
        self.texcoords = []
        self.gl_list = []
        self.init_GL()

    def init_GL(self, viewport=(1024, 768)):
        """
        initialization
        :return:
        """
        srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        # glClearDepth, 它给深度缓冲指定了一个初始值，缓冲中的每个像素的深度值都是这个，
        # 比如1，这个时候你往里面画一个物体， 由于物体的每个像素的深度值都小于等于1，
        # 所以整个物体都被显示了出来。 如果初始值指定为0， 物体的每个像素的深度值都大于等于0，
        # 所以整个物体都不可见。 如果初始值指定为0.5， 那么物体就只有深度小于0.5的那部分才是可见的
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        # 用来开启深度缓冲区的功能，启动后OPengl就可以跟踪Z轴上的像素，
        # 那么它只有在前面没有东西的情况下才会绘制这个像素，在绘制3d时，最好启用，视觉效果会比较真实
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA)
        # ------------------- Lighting
        # glEnable(GL_LIGHTING) # 如果enbale那么就使用当前的光照参数去推导顶点的颜色
        # glEnable(GL_LIGHT0) # 第一个光源，而GL_LIGHT1表示第二个光源
        #  ------------------- Display List
        self.setLightRes()  # 启用指定光源
        self.setupRC()  # 设置环境光
        self.setCamera()

    def setLightRes(self):
        """
        set the light
        :return:
        """
        lightPosition = np.array([-40, 200, 100, 0.0])  # 平行光源, GL_POSITION属性的最后一个参数为0
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, lightPosition)
        glEnable(GL_LIGHTING)  # 启用光源
        glEnable(GL_LIGHT0)  # 使用指定灯光

    def setupRC(self):
        # 当你想剔除背面的时候，你只需要调用glEnable(GL_CULL_FACE)就可以了，OPENGL状态机会自动按照默认值进行CULL_FACE，
        # 默认是glFrontFace（GL_CCW）  GL_CCW逆时针为正，GL_CW顺时针
        glEnable(GL_DEPTH_TEST)
        glFrontFace(GL_CCW)
        glEnable(GL_CULL_FACE)
        # 启用光照计算
        glEnable(GL_LIGHTING)
        # 指定环境光强度（RGBA）  此时可以控制模型的显示颜色
        ambientLight = np.array([0.0, 0.0, 0.0, 0.0])
        # 设置光照模型，将ambientLight所指定的RGBA强度值应用到环境光
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight)
        # 启用颜色追踪
        # GL_COLOR_MATERIAL使我们可以用颜色来贴物体。如果没有这行代码，纹理将始终保持原来的颜色，glColor3f(r,g,b)就没有用了
        glEnable(GL_COLOR_MATERIAL)
        # 设置多边形正面的环境光和散射光材料属性，追踪glColor
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        # glClearColor(0.0f, 0.0f, 0.5f, 1.0f)
        # GL_AMBIENT表示各种光线照射到该材质上，经过很多次反射后最终遗留在环境中的光线强度（颜色）
        # GL_DIFFUSE表示光线照射到该材质上，经过漫反射后形成的光线强度（颜色）
        # GL_SPECULAR表示光线照射到该材质上，经过镜面反射后形成的光线强度（颜色）
        # 通常，GL_AMBIENT和GL_DIFFUSE都取相同的值，可以达到比较真实的效果
        # 使用GL_AMBIENT_AND_DIFFUSE可以同时设置GL_AMBIENT和GL_DIFFUSE属性
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)  # most obj files expect to be smooth-shaded

    def setCamera(self, viewport=(1024, 768), fov=45.0, near_pt=1.0, far_pt=400.0):
        """
        :param viewport:            scale of window
        :param fov:                 Field of view in vertical direction(radian)
        :param near_pt:             float
        :param far_pt:              float
        :return:
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = viewport
        # gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        gluPerspective(fov, width / float(height), near_pt, far_pt)
        glMatrixMode(GL_MODELVIEW)
        self.init_pygame(viewport=viewport)

    def init_pygame(self, viewport=(1024.0, 768.0)):
        pygame.init()
        srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

    def UI_lisener(self):

        clock = pygame.time.Clock()
        rx, ry, rz = (0, 0, 0)
        tx, ty = (0, 0)
        zpos = 5
        rotate = move = False
        while 1:
            clock.tick(30)
            for e in pygame.event.get():
                if e.type == QUIT:
                    sys.exit()
                elif e.type == KEYDOWN and e.key == K_ESCAPE:
                    sys.exit()
                elif e.type == MOUSEBUTTONDOWN:
                    if e.button == 4:
                        zpos = max(0.1, zpos - 1)
                    elif e.button == 5:
                        zpos += 1
                    elif e.button == 1:
                        rotate = True
                    elif e.button == 3:
                        move = True
                elif e.type == MOUSEBUTTONUP:
                    if e.button == 1:
                        rotate = False
                    elif e.button == 3:
                        move = False
                elif e.type == MOUSEMOTION:
                    i, j = e.rel
                    if rotate:
                        rx += i
                        ry += j
                    if move:
                        tx += i
                        ty -= j
                # press "WSAD" to move the camera
                elif e.type == KEYDOWN and e.key == K_w:
                    ty += 2.0
                elif e.type == KEYDOWN and e.key == K_s:
                    ty-= 2.0
                elif e.type == KEYDOWN and e.key == K_a:
                    tx += 2.0
                elif e.type == KEYDOWN and  e.key == K_d:
                    tx -= 2.0
                # press "UP DOWN LEFT RIGHT" to rotate the camera
                elif e.type == KEYDOWN and e.key == K_UP:
                    ry += 2.0
                elif e.type == KEYDOWN and e.key == K_DOWN:
                    ry-= 2.0
                elif e.type == KEYDOWN and e.key == K_LEFT:
                    rx += 2.0
                elif e.type == KEYDOWN and  e.key == K_RIGHT:
                    rx -= 2.0
                # press "u j h k y i" to rotate the camera
                elif e.type == KEYDOWN and e.key == K_y:
                    ry += 2.0
                elif e.type == KEYDOWN and e.key == K_h:
                    ry-= 2.0
                elif e.type == KEYDOWN and e.key == K_g:
                    rx += 2.0
                elif e.type == KEYDOWN and  e.key == K_j:
                    rx -= 2.0
                elif e.type == KEYDOWN and e.key == K_t:
                    rz += 2.0
                elif e.type == KEYDOWN and e.key == K_u:
                    rz -= 2.0

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            ''''''
            # gluLookAt(5.0, .0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            # RENDER OBJECT
            glTranslate(tx / 20., ty / 20., - zpos)
            glRotate(ry, 1, 0, 0)
            glRotate(rx, 0, 1, 0)
            glRotate(rz, 0, 0, 1)
            for i, gl_list in enumerate(self.gl_list):
                glCallList(gl_list)

            pygame.display.flip()

    def display_obj(self, filename, m_trans=np.zeros([3]), m_rot=np.zeros([3]),
                    num_color=0, m_color=None):
        """
        Display a 3D model from the .obj file
        :param filename:
        :param m_trans:         3 float array
        :param m_rot:           3 float array      rotate angle (degree) along with (z, x, y) axis Tait-Bryan Angle
        :param num_color:       int                serial number of color matrix defined in self.color_map
        :param m_color:         3 float array      rgb -> (0, 1.0)
        :return:
                                bool
        """
        self.vertices = []
        self.v_normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                self.v_normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                continue
                # self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                # self.faces.append(face)
                self.faces.append((face, norms, texcoords, material))
        if len(self.vertices)==0 or len(self.faces)==0:
            return False

        tmp_gl_list = glGenLists(1)
        self.gl_list.append(tmp_gl_list)
        glNewList(tmp_gl_list, GL_COMPILE)
        ''''''
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        glPushMatrix()

        # ZXY Tait-Bryan Angle
        glRotatef(m_rot[0], 0.0, 0.0, 1.0)
        glRotatef(m_rot[1], 1.0, 0.0, 0.0)
        glRotatef(m_rot[2], 0.0, 1.0, 0.0)
        glTranslatef(m_trans[0], m_trans[1], m_trans[2])  # Move to the place

        if m_color is None:
            m_color = np.array(self.color_map[num_color, :])
        glColor3f(m_color[0], m_color[1], m_color[2])
        for face in self.faces:
            vertices, normals, texture_coords, material = face
            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                ''''''
                if normals[i] > 0:
                    a = self.v_normals[normals[i] - 1]
                    glNormal3fv(self.v_normals[normals[i] - 1])
                b = self.vertices[vertices[i] - 1]
                glVertex3fv(np.array(self.vertices[vertices[i] - 1]))
            glEnd()
        ''''''
        glColor3f(1.0, 1.0, 1.0)
        glPopMatrix()
        glDisable(GL_TEXTURE_2D)
        glEndList()
        return True

    def display_mesh(self, vertices=None, triangles=None, v_normals=None,
                     m_trans=np.zeros([3]), m_rot=np.zeros([3]),
                     num_color=0, m_color=None, rescale=1.0):
        """
        Display a mesh:         metric: cm
        :param vertices:        n*3 float array
        :param triangles:       m*3 int array
        :param v_normals:       n*3 float array     the normal of each vertex
        :param m_trans:         3 float array
        :param m_rot:           3 float array      rotate angle (degree) along with (z, x, y) axis Tait-Bryan Angle
        :param num_color:       int                serial number of color matrix defined in self.color_map
        :param m_color:         3 float array      rgb -> (0, 1.0)
        :param rescale:         float  rescale the 3D mesh
        :return:
                                bool
        """
        if vertices is not None:
            self.vertices = None
            self.v_normals = None
            self.faces = None
            self.material = None
            self.vertices = rescale * vertices
            if triangles is None:
                raise Exception("Please input the indexes of triangles for the mesh!")
            self.faces = 1 * triangles
            try:
                self.v_normals = 1.0 * v_normals
            except IOError:
                print("Failed to import the normals of vertices")
        else:
            return False

        tmp_gl_list = glGenLists(1)
        self.gl_list.append(tmp_gl_list)
        glNewList(tmp_gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)

        glPushMatrix()
        # ZXY Tait-Bryan Angle
        glRotatef(m_rot[0], 0.0, 0.0, 1.0)
        glRotatef(m_rot[1], 1.0, 0.0, 0.0)
        glRotatef(m_rot[2], 0.0, 1.0, 0.0)
        glTranslatef(m_trans[0], m_trans[1], m_trans[2])  # Move to the place
        ''''''
        if m_color is None:
            m_color = np.array(self.color_map[num_color, :])
        glColor3f(m_color[0], m_color[1], m_color[2])

        for i in range(self.faces.shape[0]):
            glBegin(GL_POLYGON)
            for j in range(self.faces.shape[1]):
                if self.v_normals is not None:
                    # a = self.v_normals[self.faces[i, j], :].tolist()
                    glNormal3fv(self.v_normals[self.faces[i, j], :])
                '''
                if self.texcoords is not None:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                '''
                glVertex3fv(self.vertices[self.faces[i, j], :])
            glEnd()
        glColor3f(1.0, 1.0, 1.0)
        glPopMatrix()
        glDisable(GL_TEXTURE_2D)
        glEndList()
        return True

    def draw_spheres(self, pos_spheres, radius=0.2, num_color=0, m_color=None):
        """
        Draw a set of spheres
        :param pos_spheres:         n*3 float array    metric: cm
        :param radius:
        :param num_color:       int                serial number of color matrix defined in self.color_map
        :param m_color:         3 float array      rgb -> (0, 1.0)
        :return:
        """
        pos_spheres = pos_spheres.reshape(pos_spheres.size//3, 3)
        sphere = gluNewQuadric()
        if m_color is None:
            m_color = np.array(self.color_map[num_color, :])

        tmp_gl_list = glGenLists(1)
        self.gl_list.append(tmp_gl_list)
        glNewList(tmp_gl_list, GL_COMPILE)

        for i in range(pos_spheres.shape[0]):
            glPushMatrix()
            glColor3f(m_color[0], m_color[1], m_color[2])
            glTranslatef(pos_spheres[i, 0], pos_spheres[i, 1], pos_spheres[i, 2])  # Move to the place
            gluSphere(sphere, radius, 32, 16)  # Draw sphere
            glPopMatrix()
        glColor3f(1.0, 1.0, 1.0)
        glEndList()

    def draw_lines(self, pos_p2p, width=1.0, num_color=0, m_color=None):
        """
        Draw a set of lines
        :param width:           width of lines
        :param pos_p2p:         n*6 float array     start/end point
        :param num_color:
        :param m_color:
        :return:
        """
        pos_p2p = pos_p2p.reshape(pos_p2p.size//6, 6)
        if m_color is None:
            m_color = np.array(self.color_map[num_color, :])

        tmp_gl_list = glGenLists(1)
        self.gl_list.append(tmp_gl_list)

        glNewList(tmp_gl_list, GL_COMPILE)
        glLineWidth(width)
        glColor3f(m_color[0], m_color[1], m_color[2])
        for i in range(pos_p2p.shape[0]):
            glBegin(GL_LINES)
            glVertex3f(pos_p2p[i, 0], pos_p2p[i, 1], pos_p2p[i, 2])
            glVertex3f(pos_p2p[i, 3], pos_p2p[i, 4], pos_p2p[i, 5])
            glEnd()
        glColor3f(1.0, 1.0, 1.0)
        glEndList()

    def draw_points(self, pts, size=1.0, num_color=0, m_color=None):
        """
        Draw a set of points
        :param size:            float
        :param pts:             n*3 float array     coordinates of points
        :param num_color:
        :param m_color:
        :return:
        """
        pts = pts.reshape(pts.size//6, 6)
        if m_color is None:
            m_color = np.array(self.color_map[num_color, :])

        tmp_gl_list = glGenLists(1)
        self.gl_list.append(tmp_gl_list)

        glNewList(tmp_gl_list, GL_COMPILE)
        glPointSize(size)
        glColor3f(m_color[0], m_color[1], m_color[2])
        for i in range(pts.shape[0]):
            glBegin(GL_POINTS)
            glVertex3f(pts[i, 0], pts[i, 1], pts[i, 2])
            glEnd()
        glColor3f(1.0, 1.0, 1.0)
        glEndList()

