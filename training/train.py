from pathlib import Path

import torch
import torch.distributions as dist
from model.threedepn import ThreeDEPNDecoder
from data.shapenet import ShapeNet
from pytorch_lightning.loggers import TensorBoardLogger


def train(model, train_dataloader, latent_vectors, latent_log_var, device, config):
    # Initialize logger
    logger = TensorBoardLogger("logs", name=config['experiment_name'])
    logger.log_graph(model)
    logger.log_hyperparams(config)
    
    # Define loss
    reconstruction_loss_criterion = torch.nn.SmoothL1Loss()
    reconstruction_loss_criterion.to(device)

    # Define params
    params = [
        {
            'params': latent_vectors,
            'lr': config['learning_rate_code']
        }
    ]
    if config['vad_free']:
        params.append({
            'params': latent_log_var,
            'lr': config['learning_rate_log_var']
        })
    
    if config['test'] is False:
        params.append({
            'params': model.parameters(),
            'lr': config['learning_rate_model']
        })

    print(f'Training params: {len(params)}')

    # Define Optimizers
    optimizer = torch.optim.Adam(params)

    # Here, we follow the original implementation to also use a learning rate scheduler -- it simply reduces the learning rate to half every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()

    # Keep track of running average of train loss for printing
    best_loss = float('inf')
    train_loss_running = 0.
    reconstruction_loss_running = 0.
    kl_loss_running = 0.

    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device, set optimizer gradients to zero, perform forward pass
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()

            # Create distribution from latent codes and laten variances (each sample has its own distribution)
            Dist = dist.Normal(latent_vectors[batch['index']], torch.exp(latent_log_var[batch['index']]))

            # Sample from each distribution
            x_vad = Dist.rsample()

            # Forward pass
            reconstruction = model(x_vad)
            
            # Targets
            target = batch['target_df']
            q_z = dist.Normal(0, 1)

            # Compute loss, Compute gradients, Update network parameters
            reconstruction_loss = reconstruction_loss_criterion(reconstruction, target)
            kl_loss = torch.mean(dist.kl_divergence(Dist, q_z))
            loss = reconstruction_loss + config['kl_weight'] * kl_loss

            # Compute gradients
            loss.backward()

            # Update network parameters
            optimizer.step()

            # Logging
            train_loss_running += loss.item()
            reconstruction_loss_running += reconstruction_loss.item()
            kl_loss_running += kl_loss.item()
            # iteration = epoch * len(train_dataloader) + batch_idx

            # if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
            if batch_idx == len(train_dataloader) - 1:
                samples_per_epoch = config["batch_size"] * len(train_dataloader)
                train_loss = train_loss_running / samples_per_epoch
                print(f"[{(epoch + 1):03d}/{config['max_epochs']:05d}] train_loss: {train_loss_running / samples_per_epoch:.6f}"
                      f' kl_loss: {kl_loss_running/ samples_per_epoch:.6f} normal_loss: {reconstruction_loss_running / samples_per_epoch:.6f}')
                logger.log_metrics({
                    'train_loss': train_loss_running / samples_per_epoch,
                    'kl_loss': kl_loss_running/ samples_per_epoch,
                    'reconstruction_loss': reconstruction_loss_running / samples_per_epoch,
                    'epoch': epoch + 1
                }, epoch + 1)
                train_loss_running = 0.
                reconstruction_loss_running = 0.
                kl_loss_running = 0.

                # Save model checkpoint
                if train_loss < best_loss:
                    torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_best.ckpt')
                    torch.save(latent_vectors, f'runs/{config["experiment_name"]}/latent_best.pt')
                    torch.save(latent_log_var, f'runs/{config["experiment_name"]}/log_var_best.pt')
                    best_loss = train_loss
                    
        # Update scheduler          
        scheduler.step()


def main(config):
    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = ShapeNet('train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        # worker_init_fn=train_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    # Instantiate model
    model = ThreeDEPNDecoder()

    # Initialize latent codes and latent variance
    latent_vectors = torch.normal(mean=0, std=1, size=(len(train_dataset), config['latent_code_length']), device=device)
    latent_vectors.requires_grad = True
    latent_log_var = torch.zeros(len(train_dataset), config['latent_code_length'], device=device)
    latent_log_var.requires_grad = config['vad_free']

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        print('Loading saved model, latent codes, and latent variances...')
        model.load_state_dict(torch.load(f"runs/{config['resume_ckpt']}/model_best.ckpt", map_location='cpu'))
        latent_vectors = torch.load(f"runs/{config['resume_ckpt']}/latent_best.pt", map_location = 'cpu')
        latent_log_var = torch.load(f"runs/{config['resume_ckpt']}/log_var_best.pt", map_location = 'cpu')

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, latent_vectors, latent_log_var, device, config)
