from model.threedepn import ThreeDEPNDecoder
from util.visualization import visualize_mesh
from skimage.measure import marching_cubes
import torch.distributions as dist
import torch
from data.shapenet import ShapeNet

def visualize_dataset_sample(filter_class, index):
    dataset = ShapeNet('train', filter_class = filter_class)
    sample = dataset[index]
    input_mesh = marching_cubes(sample['target_df'], level=1)
    visualize_mesh(input_mesh[0], input_mesh[1], flip_axes=True)

def visualize_ad(experiment, index):
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