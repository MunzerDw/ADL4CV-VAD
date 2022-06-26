import random
import torch
from model.threedepn import ThreeDEPNDecoder
import trimesh as tm
from skimage.measure import marching_cubes
import numpy as np
import pyvista as pv
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

# https://github.com/ChrisWu1997/Multimodal-Shape-Completion/blob/57c54b8c84e8413b6a699fe8d16e8bc88e1222fe/evaluation/mmd.py#L39

def scale_to_unit_sphere(points, center=None):
    """
    scale point clouds into a unit sphere
    input: 
        points: Tensor (num_point, 3)
    output: Tensor (num_point, 3)
    """
    if center is None:
        midpoints = (torch.max(points, dim=0).values + torch.min(points, dim=0).values) / 2
    else:
        midpoints = torch.asarray(center)
    points = points - midpoints
    scale = torch.max(torch.sqrt(torch.sum(points ** 2)))
    points = points / scale
    return points

def upsample_point_cloud(points, n_pts):
    """
    upsample points by random choice
    input: 
        points: Numpy (num_point, 3)
        n_pts: int
    output: Numpy (num_point, 3)
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts - points.shape[0])
    dup_points = points[p_idx]
    points = np.concatenate([points, dup_points], axis=0)
    return points

def convert_df_to_point_cloud(sample, count=2048):
    """
    input: 
        sample: tensor (32, 32, 32)
        count: number of points to sample for the point cloud
    output: tensor (count, 3)
    """
    verts, faces, _, values = marching_cubes(sample.cpu().detach().numpy(), level=1)
    mesh = tm.Trimesh(verts, faces, vertex_normals=values)
    point_cloud = tm.sample.sample_surface(mesh, count=count)
    point_cloud = point_cloud[0]
    # point_cloud = upsample_point_cloud(point_cloud, count)
    point_cloud = torch.from_numpy(point_cloud).float()
    point_cloud = scale_to_unit_sphere(point_cloud)
    return point_cloud

def convert_set_to_point_cloud(samples, count=2048, device=None):
    """
    input: 
        samples: tensor (batch_size, 32, 32, 32)
        count: number of points to sample for the point cloud
    output: tensor (batch_size, count, 3)
    """
    point_clouds = []
    for sample in samples:
        verts, faces, _, values = marching_cubes(sample.cpu().detach().numpy(), level=1)
        mesh = tm.Trimesh(verts, faces, vertex_normals=values)
        point_cloud = tm.sample.sample_surface(mesh, count=count)
        point_cloud = point_cloud[0]
        # point_cloud = upsample_point_cloud(point_cloud, count)
        point_cloud = torch.from_numpy(point_cloud).float().to(device)
        point_cloud = scale_to_unit_sphere(point_cloud)
        point_clouds.append(point_cloud)
    return torch.stack(point_clouds).to(device)

def visualize_point_cloud(point_cloud):
    # points is a 3D numpy array (n_points, 3) coordinates of a sphere
    cloud = pv.PolyData(point_cloud)
    cloud.plot()

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
    distances = torch.linalg.norm(expanded_array1-expanded_array2, dim=1) # num_points * num_points
    distances = torch.reshape(distances, (num_point, num_point)) # num_points, num_points
    distances = torch.min(distances, dim=1) # num_points
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
    distances = []
    for i, sample in enumerate(set1):
        sample_distances = []
        for sample2 in set2:
            cf_distance = chamfer_distance(sample, sample2)
            sample_distances.append(cf_distance)
        mmd = min(sample_distances)
        distances.append(mmd)
        # print(f"{i + 1}: {mmd}")
    return torch.stack(distances)

def min_sample(split, filter_class, sample, device=None):
    sample = convert_df_to_point_cloud(sample)
    sample = sample.to(device)
    # get test set
    val = []
    val_dataset = ShapeNet(split, filter_class=filter_class)
    for data_dict in val_dataset:
        target_df = torch.from_numpy(data_dict['target_df']).float().to(device)
        val.append(target_df)
    val = torch.stack(val).to(device)
    val_point_clouds = convert_set_to_point_cloud(val, count=2048, device=device)
    # get min cf distance
    min_cf_distance = 1000
    final_sample = None
    for i, sample_val in enumerate(val_point_clouds):
        cf_distance = chamfer_distance(sample, sample_val)
        if cf_distance < min_cf_distance:
            final_sample = val[i]
            min_cf_distance = cf_distance

    return final_sample, min_cf_distance

def MMD(experiment, split, filter_class, n_samples, device=None):
    # generate n new samples
    samples = generate_samples(experiment, n_samples, device)
    samples = samples.squeeze(1)
    # get test set
    val = []
    val_dataset = ShapeNet(split, filter_class=filter_class)
    for data_dict in val_dataset:
        target_df = torch.from_numpy(data_dict['target_df']).float().to(device)
        val.append(target_df)
    val = torch.stack(val).to(device)
    # convert sets to pointclouds
    samples_point_clouds = convert_set_to_point_cloud(samples, count=2048, device=device)
    val_point_clouds = convert_set_to_point_cloud(val, count=2048, device=device)
    mmd_values = _mmd(val_point_clouds, samples_point_clouds)
    return mmd_values.mean(), mmd_values

def TMD(experiment, n_samples, device=None):
    # generate n new samples
    samples = generate_samples(experiment, n_samples, device)
    samples = samples.squeeze(1)
    samples_point_clouds = convert_set_to_point_cloud(samples, count=2048, device=device)
    cd_distance_avgs = []
    for i, point_cloud1 in enumerate(samples_point_clouds):
        cd_distances = []
        for n, point_cloud2 in enumerate(samples_point_clouds):
            if i != n:
                cd_distance = chamfer_distance(point_cloud1, point_cloud2)
                cd_distances.append(cd_distance)
        cd_distance_avgs.append(sum(cd_distances) / len(cd_distances))
    return sum(cd_distance_avgs), samples
            

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
    
