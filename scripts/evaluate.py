import torch
from model.threedepn import ThreeDEPNDecoder
import trimesh as tm
from skimage.measure import marching_cubes
import numpy as np
import pyvista as pv

def generate_samples(experiment, n):
    samples = []
    # Load model
    model = ThreeDEPNDecoder()
    model.load_state_dict(torch.load(f"runs/{experiment}/model_best.ckpt", map_location='cpu'))
    # Sample
    for i in range(n):
        # Dist = dist.Normal(torch.zeros(256), torch.ones(256))
        x_vad = torch.randn(256).unsqueeze(0)
        # Forward pass
        output_meshes_int = model(x_vad)
        # add to samples
        samples.append(output_meshes_int)
    samples = torch.stack(samples)
    return samples

def convert_df_to_point_cloud(sample, count=5000):
    """
    input: 
        sample: tensor (batch_size, 32, 32, 32)
        count: number of points to sample for the point cloud
    output: trimesh.caching.TrackedArray (subclass of numpy array) (count, 3)
    """
    verts, faces, _, values = marching_cubes(sample.detach().numpy(), level=1)
    mesh = tm.Trimesh(verts, faces, vertex_normals=values)
    point_cloud = tm.sample.sample_surface(mesh, count=count)
    return point_cloud[0]

def visualize_point_cloud(point_cloud):
    # points is a 3D numpy array (n_points, 3) coordinates of a sphere
    cloud = pv.PolyData(point_cloud)
    cloud.plot()

# https://stackoverflow.com/questions/47060685/chamfer-distance-between-two-point-clouds-in-tensorflow

def array2samples_distance(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1 
    """
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance_numpy(point_cloud1, point_cloud2):
    """
    input: 
        point_cloud1: numpy array (counts, 3)
        point_cloud2: numpy array (counts, 3)
    output: float
    """
    dist = 0
    av_dist1 = array2samples_distance(point_cloud1, point_cloud2)
    av_dist2 = array2samples_distance(point_cloud2, point_cloud1)
    dist = dist + av_dist1+av_dist2
    return dist

def mmd(set1, set2):
    """
    input: 
        set1: tensor (num_samples1, 1, 32, 32, 32)
        set2: tensor (num_samples2, 1, 32, 32, 32)
    output: numpy array (num_samples1)
    """
    distances = []
    for sample in set1:
        sample_distances = []
        point_cloud = convert_df_to_point_cloud(sample.squeeze(0))
        for sample2 in set2:
            point_cloud2 = convert_df_to_point_cloud(sample2.squeeze(0))
            cf_distance = chamfer_distance_numpy(point_cloud, point_cloud2)
            sample_distances.append(cf_distance)
        distances.append(sum(sample_distances) / len(sample_distances))
    return np.array(distances)