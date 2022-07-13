import time
import numpy as np
from model.threedepn import ThreeDEPNDecoder
from util.visualization import visualize_mesh
from skimage.measure import marching_cubes
import torch.distributions as dist
import torch
from data.shapenet import ShapeNet
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output

def visualize_dataset_sample(filter_class, index):
    """
    input: 
        filter_class: string (class of the dataset (airplane, chair, car, etc.))
        index: int (index of sample to visualize)
    """
    dataset = ShapeNet('train', filter_class = filter_class)
    sample = dataset[index]
    input_mesh = marching_cubes(sample['target_df'], level=1)
    visualize_mesh(input_mesh[0], input_mesh[1], flip_axes=True)

def visualize_ad(experiment, index):
    """
    input: 
        experiment: string (name of the experiment)
        index: int (index of sample to visualize)
    """
    # Load model
    model = ThreeDEPNDecoder()
    model.load_state_dict(torch.load(f"runs/{experiment}/model_best.ckpt", map_location='cpu'))
    # Load latent codes
    latent_vectors = torch.load(f"runs/{experiment}/latent_best.pt", map_location = 'cpu')
    # Sample
    x = latent_vectors[index].unsqueeze(0)
    # Forward pass
    output_meshes_int = model(x)
    # Visualize
    output_mesh_int = marching_cubes(output_meshes_int[0].detach().numpy(), level=1)
    visualize_mesh(output_mesh_int[0], output_mesh_int[1], flip_axes=True)

def visualize_vad(experiment, index):
    """
    input: 
        experiment: string (name of the experiment)
        index: int (index of sample to visualize)
    """
    # Load model
    model = ThreeDEPNDecoder()
    model.load_state_dict(torch.load(f"runs/{experiment}/model_best.ckpt", map_location='cpu'))
    # Load latent codes
    latent_vectors = torch.load(f"runs/{experiment}/latent_best.pt", map_location = 'cpu')
    log_vars = torch.load(f"runs/{experiment}/latent_best.pt", map_location = 'cpu')
    # Sample
    x = latent_vectors[index]
    Dist = dist.Normal(x, torch.exp(log_vars[index]))
    x_vad = Dist.rsample().unsqueeze(0)
    # Forward pass
    output_meshes_int = model(x_vad)
    # Visualize
    output_mesh_int = marching_cubes(output_meshes_int[0].detach().numpy(), level=1)
    visualize_mesh(output_mesh_int[0], output_mesh_int[1], flip_axes=True)

def visualize_vad_norm(experiment):
    """
    input: 
        experiment: string (name of the experiment)
        index: int (index of sample to visualize)
    """
    # Load model
    model = ThreeDEPNDecoder()
    model.load_state_dict(torch.load(f"runs/{experiment}/model_best.ckpt", map_location='cpu'))
    # Sample
    # Dist = dist.Normal(torch.zeros(256), torch.ones(256))
    x_vad = torch.randn(256).unsqueeze(0)
    # Forward pass
    output_meshes_int = model(x_vad)
    # Visualize
    output_mesh_int = marching_cubes(output_meshes_int[0].detach().numpy(), level=1)
    visualize_mesh(output_mesh_int[0], output_mesh_int[1], flip_axes=True)

def visualize_interpolation_ad(experiment, index1, index2, a1=0.5, a2=0.5):
    """
    input: 
        experiment: string (name of the experiment)
        index1: int (index of sample 1)
        index2: int (index of sample 2)
        a1: float (weight of the first sample)
        a2: float (weight of the second sample)
    """
    # Load model
    model = ThreeDEPNDecoder()
    model.load_state_dict(torch.load(f"runs/{experiment}/model_best.ckpt", map_location='cpu'))
    # Load latent codes
    latent_vectors = torch.load(f"runs/{experiment}/latent_best.pt", map_location = 'cpu')
    # Sample
    x1 = latent_vectors[index1]
    x2 = latent_vectors[index2]
    x = (a1*x1 + a2*x2).unsqueeze(0)
    # Forward pass
    output_meshes_int = model(x)
    # Visualize
    output_mesh_int = marching_cubes(output_meshes_int[0].detach().numpy(), level=1)
    visualize_mesh(output_mesh_int[0], output_mesh_int[1], flip_axes=True)

def visualize_latent_space(experiment, indicies):
    """
    input: 
        experiment: string (name of the experiment)
        indicies: list[int] (indicies to visualize with a blue dot in the latent space)
    """
    latent_vectors = torch.load(f"runs/{experiment}/latent_best.pt", map_location = 'cpu').detach().numpy()
    pca = PCA(n_components=2)
    pca.fit(latent_vectors)
    pca.explained_variance_ratio_
    Z_pca = pca.transform(latent_vectors)
    Z_pca.shape
    fig, ax = plt.subplots()
    x = np.array([[1, 0, 0, 1]])
    C = np.repeat(x, latent_vectors.shape[0], axis=0)
    Z = np.repeat(1, latent_vectors.shape[0], axis=0)
    for i in indicies:
        C[i] = np.array([[0, 0, 1, 1]])
        Z[i] = 2
    ax.scatter(*Z_pca.T, zorder=1, c = C)
    ax.scatter(*Z_pca[indicies].T, zorder=2, c = C[indicies])
    ax.set_xlabel('$pc_1$')
    ax.set_ylabel('$pc_2$')
    plt.show()

def visualize_interpolation_ad_steps(experiment, index1, index2):
    """
    input: 
        experiment: string (name of the experiment)
        index1: int (index of sample 1)
        index2: int (index of sample 2)
    """
    a1 = 0.3
    a2 = 1 - a1
    while a1 < 0.8:
        clear_output(wait=True)
        print(a1, a2)
        visualize_interpolation_ad(experiment, index1, index2, a1, a2)
        a1 += 0.01
        a2 = 1 - a1
        time.sleep(0.5)