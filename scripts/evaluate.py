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
        sample: tensor (32, 32, 32)
        count: number of points to sample for the point cloud
    output: Tensor (count, 3)
    """
    verts, faces, _, values = marching_cubes(sample.detach().numpy(), level=1)
    mesh = tm.Trimesh(verts, faces, vertex_normals=values)
    point_cloud = tm.sample.sample_surface(mesh, count=count)
    return torch.from_numpy(point_cloud[0]).float()

def convert_set_to_point_cloud(samples, count=10000):
    """
    input: 
        samples: tensor (batch_size, 32, 32, 32)
        count: number of points to sample for the point cloud
    output: Tensor (batch_size, count, 3)
    """
    point_clouds = []
    for sample in samples:
        verts, faces, _, values = marching_cubes(sample.detach().numpy(), level=1)
        mesh = tm.Trimesh(verts, faces, vertex_normals=values)
        point_cloud = tm.sample.sample_surface(mesh, count=count)
        point_cloud = torch.from_numpy(point_cloud[0]).float()
        point_clouds.append(point_cloud)
    return torch.stack(point_clouds)

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
    expanded_array1 = torch.tile(array1, (num_point, 1))
    expanded_array2 = torch.reshape(
            torch.tile(torch.unsqueeze(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = torch.linalg.norm(expanded_array1-expanded_array2, axis=1)
    distances = torch.reshape(distances, (num_point, num_point))
    distances = torch.min(distances, axis=1)
    distances = torch.mean(distances.values)
    return distances

def chamfer_distance(point_cloud1, point_cloud2):
    """
    input: 
        point_cloud1: tensor (counts, 3)
        point_cloud2: tensor (counts, 3)
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
        set1: tensor (num_samples1, num_points, 3)
        set2: tensor (num_samples2, num_points, 3)
    output: numpy array (num_samples1)
    """
    set1 = set1.cuda()
    set2 = set2.cuda()
    distances = []
    for i, sample in enumerate(set1):
        sample_distances = []
        for sample2 in set2:
            cf_distance = chamfer_distance_numpy(sample, sample2)
            sample_distances.append(cf_distance)
        mmd = sum(sample_distances) / len(sample_distances)
        distances.append(mmd)
        print(f"{i + 1}: {mmd}")
    return torch.stack(distances)