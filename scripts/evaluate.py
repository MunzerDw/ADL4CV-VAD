import torch
from model.threedepn import ThreeDEPNDecoder
import trimesh as tm
from skimage.measure import marching_cubes
import numpy as np
from data.shapenet import ShapeNet

def generate_samples(experiment, n, device=None):
    samples = []
    # Load model
    model = ThreeDEPNDecoder()
    model.load_state_dict(torch.load(f"runs/{experiment}/model_best.ckpt", map_location=device))
    model.to(device)
    # Sample
    for _ in range(n):
        # Dist = dist.Normal(torch.zeros(256), torch.ones(256))
        x_vad = torch.randn(256, device=device).unsqueeze(0)
        # Forward pass
        output_meshes_int = model(x_vad)
        # add to samples
        samples.append(output_meshes_int)
    samples = torch.stack(samples).to(device)
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

def convert_set_to_point_cloud(samples, count=10000, device=None):
    """
    input: 
        samples: tensor (batch_size, 32, 32, 32)
        count: number of points to sample for the point cloud
    output: Tensor (batch_size, count, 3)
    """
    point_clouds = []
    for sample in samples:
        verts, faces, _, values = marching_cubes(sample.cpu().detach().numpy(), level=1)
        mesh = tm.Trimesh(verts, faces, vertex_normals=values)
        point_cloud = tm.sample.sample_surface(mesh, count=count)
        point_cloud = torch.from_numpy(point_cloud[0]).float().to(device)
        point_clouds.append(point_cloud)
    return torch.stack(point_clouds).to(device)

# def visualize_point_cloud(point_cloud):
#     # points is a 3D numpy array (n_points, 3) coordinates of a sphere
#     cloud = pv.PolyData(point_cloud)
#     cloud.plot()

# https://stackoverflow.com/questions/47060685/chamfer-distance-between-two-point-clouds-in-tensorflow

def _point_clouds_min_distance(point_cloud1, point_cloud2):
    """
    arguments: 
        point_cloud1: tensor (num_point, num_feature)
        point_cloud2: tensor (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1 
    """
    num_point, num_features = point_cloud1.shape
    expanded_array1 = torch.tile(point_cloud1, (num_point, 1)) # num_points * num_points, num_feature
    expanded_array2 = torch.reshape(
        torch.tile(
            torch.unsqueeze(point_cloud2, 1), 
            (1, num_point, 1)
        ),
        (-1, num_features)
    ) # num_points * num_points, num_feature
    distances = torch.linalg.norm(expanded_array1-expanded_array2, axis=1) # num_points * num_points
    distances = torch.reshape(distances, (num_point, num_point)) # num_points, num_points
    distances = torch.min(distances, axis=1) # num_points
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
    av_dist1 = _point_clouds_min_distance(point_cloud1, point_cloud2)
    av_dist2 = _point_clouds_min_distance(point_cloud2, point_cloud1)
    dist = dist + av_dist1 + av_dist2
    return dist

def _mmd(set1, set2):
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
            cf_distance = chamfer_distance(sample, sample2)
            sample_distances.append(cf_distance)
        mmd = sum(sample_distances) / len(sample_distances)
        distances.append(mmd)
        print(f"{i + 1}: {mmd}")
    return torch.stack(distances)

def MMD(experiment, split, filter_class, n_samples, device=None):
    # generate n new samples
    samples = generate_samples(experiment, n_samples, device)
    samples = samples.squeeze(1)
    val = []
    val_dataset = ShapeNet(split, filter_class=filter_class)
    for data_dict in val_dataset:
        target_df = torch.from_numpy(data_dict['target_df']).float().to(device)
        val.append(target_df)
    val = torch.stack(val).to(device)
    # convert sets to pointclouds
    samples_point_clouds = convert_set_to_point_cloud(samples, device=device)
    val_point_clouds = convert_set_to_point_cloud(val, device=device)
    mmd_value = _mmd(samples_point_clouds, val_point_clouds)
    return mmd_value.mean(), mmd_value

def IOU(experiment, split, filter_class, device):
    model = ThreeDEPNDecoder()
    model.load_state_dict(torch.load(f"runs/{experiment}/model_best.ckpt", map_location=device))
    latent_vectors = torch.load(f"runs/{experiment}/latent_best.pt", map_location=device)
    dataset = ShapeNet(split, filter_class=filter_class)
    model.to(device)
    # latent_vectors.to(device)
    sum1 = 0
    for index in range(len(dataset)):
        # idx = idx + 1
        sample = torch.tensor(dataset[index]['target_df']).to(device)
        sample[sample < 1] = 1
        sample[sample > 1] = 0
        x = latent_vectors[index].unsqueeze(0)
        output_mesh_iou = model(x)
        model_sample = output_mesh_iou[0]
        model_sample[model_sample < 1] = 1
        model_sample[model_sample > 1] = 0
        i = sample + model_sample
        u = sample + model_sample
        u[u > 1] = 1
        i = i - 1
        i[i < 0] = 0
        sum1 = sum1 + (torch.sum(i) / torch.sum(u)).item()

    return sum1 / len(dataset)
    
