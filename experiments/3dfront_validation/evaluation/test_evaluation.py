"""Analytic checks for distance, area sampling, ray depth and visibility semantics."""
import unittest
import numpy as np
import trimesh
from .evaluate_scene import make_query, distances, samples, raycast, classify_projection, Y_TO_Z

class EvaluationChecks(unittest.TestCase):
    def test_nearest_triangle_distance(self):
        mesh = trimesh.creation.box(extents=[2,2,2])
        actual = distances(make_query(mesh),np.array([[0,0,2.],[0,0,0.],[1,0,0.],[2,2,2.]]))
        np.testing.assert_allclose(actual,[1,1,0,np.sqrt(3)],atol=1e-6)

    def test_area_sampling(self):
        mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],[10,0,0],[12,0,0],[10,2,0]],
                             faces=[[0,1,2],[3,4,5]],process=False)
        points,faces=samples(mesh,100000,np.random.default_rng(42))
        self.assertAlmostEqual((faces==1).mean(),.8,delta=.005)
        self.assertTrue((points[:,2]==0).all())
        np.testing.assert_allclose(trimesh.transform_points(trimesh.transform_points(points,Y_TO_Z),np.linalg.inv(Y_TO_Z)),points)

    def test_camera_z_not_ray_length(self):
        mesh=trimesh.creation.box(extents=[20,20,.2]);mesh.apply_translation([0,0,-2.1])
        z=raycast(make_query(mesh),np.eye(4),np.pi/2,np.array([1.,2.,3.]),np.array([2.,2.,2.]),5,5)
        np.testing.assert_allclose(z,2,atol=1e-6)

    def test_visibility_and_free_space(self):
        camera={'transform_matrix':np.eye(4).tolist()}
        # Same ray: front of reference, on reference, occluded, outside frustum.
        points=np.array([[0,0,-1.],[0,0,-2.],[0,0,-3.],[100,0,-2.]])
        visible,hidden,unknown,front,background=classify_projection(points,[camera],[np.full((5,5),2.)],np.pi/2,2.)
        np.testing.assert_array_equal(visible,[False,True,False,False])
        np.testing.assert_array_equal(hidden,[False,False,True,False])
        np.testing.assert_array_equal(unknown,[True,False,False,True])
        np.testing.assert_array_equal(front,[True,False,False,False])
        self.assertFalse(background.any())

if __name__=='__main__':
    unittest.main()
