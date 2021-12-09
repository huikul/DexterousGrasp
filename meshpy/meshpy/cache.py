# python-pcl

def _pcl_visualization(point_cloud):
    visual = pcl.pcl_visualization.PCLVisualizering()
    visual.AddCoordinateSystem(0.01, 0)
    visual.AddPointCloud(point_cloud)
    v = True
    while v:
        visual.SpinOnce()
        v = not (visual.WasStopped())


#self.graspable.pcl_surface.
test_pcl = pcl.PointCloud()
box_filter = self.graspable.pcl_surface.make_cropbox()
box_filter.set_MinMax(-width / 2, -width / 2, -1., 0., width / 2, width / 2, 1., 0.)
box_filter.set_Translation(self.point[0], self.point[1], self.point[2])
[rx, ry, rz]= self._compute_rotation_angle_XYZ(direction)
box_filter.set_Rotation(rx, ry, rz)

outcloud = pcl.PointCloud()
outcloud = box_filter.filter()
print(outcloud.size)
#fl = self.graspable.pcl_surface.make_voxel_grid_filter()
#fl.set_leaf_size (0.01, 0.01, 0.01)
#p = multiprocessing.Process(target=_pcl_visualization(outcloud))
#p.start()
#p.join()
visual = pcl.pcl_visualization.CloudViewing()
visual_1 = pcl.pcl_visualization.PCLVisualizering()
# PointXYZ

visual.ShowMonochromeCloud(self.graspable.pcl_surface, b'cloud')
visual_1.AddCoordinateSystem(0.1, 0)
#visual_1.AddPointCloud(outcloud, b'cloudssdddd')
s = visual_1.AddSphere()
ECluster = outcloud.make_EuclideanClusterExtraction()
ECluster.set_ClusterTolerance(0.005)
ECluster.set_MinClusterSize(400)
ECluster.set_MaxClusterSize(5000)
#out = pcl.PointCloud()
cluster_indices = ECluster.Extract()
cloud_cluster = pcl.PointCloud()
for j, indices in enumerate(cluster_indices):
    points = np.zeros((len(indices), 3), dtype=np.float32)
    # for indice in range(len(indices)):
    for i, indice in enumerate(indices):
        points[i] = outcloud[indice]
    cloud_cluster.from_array(points)
visual_1.AddPointCloud(cloud_cluster, b'cloudssdddd')
#visual.Spin()
# visual.ShowColorCloud(outcloud, b'sddd')
#sual_1.AddCoordinateSystem(0.01,0)
#visual_1.AddPointCloud(outcloud,id=b'cloud')
#visual_1.Spin()
#visual_1.ShowMonochromeCloud(outcloud, b'cloud')



