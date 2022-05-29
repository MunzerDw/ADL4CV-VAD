from pathlib import Path

import numpy as np
import torch
import torch.distributions as dist
from exercise_3.model.threedepn import ThreeDEPNDecoder
from exercise_3.data.shapenet import ShapeNet


def train(model, train_dataloader, val_dataloader, latent_vectors, latent_log_var, device, config):
    # TODO: Declare loss and move to device; we need both smoothl1 and pure l1 losses here
    best_loss = float('inf')
    loss_criterion = torch.nn.SmoothL1Loss()
    loss_criterion.to(device)
    loss_criterion_test = torch.nn.L1Loss()
    loss_criterion_test.to(device)
    # TODO: Declare optimizer with learning rate given in config
    if config['vad_free']:
        optimizer = torch.optim.Adam([
            {
                'params': model.parameters(),
                'lr': config['learning_rate_model']
            },
            {
                'params': latent_vectors,
                'lr': config['learning_rate_code']
            },
            {
                'params': latent_log_var,
                'lr': config['learning_rate_log_var']
            }
        ])
    else:
        optimizer = torch.optim.Adam([
            {
                'params': model.parameters(),
                'lr': config['learning_rate_model']
            },
            {
                'params': latent_vectors,
                'lr': config['learning_rate_code']
            }
        ])

    # Here, we follow the original implementation to also use a learning rate scheduler -- it simply reduces the learning rate to half every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # TODO: Set model to train
    model.train()
    best_loss_val = np.inf

    # Keep track of running average of train loss for printing
    train_loss_running = 0.
    init_loss_running = 0.
    kl_loss_running = 0.
    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            # TODO: Move batch to device, set optimizer gradients to zero, perform forward pass
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()
            Dist = dist.Normal(latent_vectors[batch['index']], torch.exp(latent_log_var[batch['index']]))
            x_vad = Dist.rsample()
            reconstruction = model(x_vad)
            # Mask out known regions -- only use loss on reconstructed, previously unknown regions
            #reconstruction[batch['input_sdf'][:, 1] == 1] = 0  # Mask out known
            target = batch['target_df']
            #target[batch['input_sdf'][:, 1] == 1] = 0

            # TODO: Compute loss, Compute gradients, Update network parameters
            init_loss = loss_criterion(reconstruction, target)
            q_z = dist.Normal(0, 1)
            kl_loss = torch.mean(dist.kl_divergence(Dist, q_z))
            loss = init_loss + config['kl_weight'] * kl_loss
            loss.backward()
            optimizer.step()
            # Logging
            train_loss_running += loss.item()
            init_loss_running += init_loss.item()
            kl_loss_running += kl_loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                train_loss = train_loss_running / config["print_every_n"]
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}'
                      f' kl_loss: {kl_loss_running/ config["print_every_n"]:.6f} normal_loss: {init_loss_running/config["print_every_n"]:.6f}')

                if train_loss < best_loss:
                    torch.save(model.state_dict(), f'exercise_3/runs/{config["experiment_name"]}/model_best.ckpt')
                    torch.save(latent_vectors, f'exercise_3/runs/{config["experiment_name"]}/latent_best.pt')
                    torch.save(latent_log_var, f'exercise_3/runs/{config["experiment_name"]}/log_var_best.pt')
                    best_loss = train_loss
                train_loss_running = 0.
                init_loss_running = 0.
                kl_loss_running = 0.
            # Validation evaluation and logging
        scheduler.step()


def main(config):
    """
    Function for training PointNet on ShapeNet
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

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
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=train_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    val_dataset = ShapeNet('val' if not config['is_overfit'] else 'overfit')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=val_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    # Instantiate model
    model = ThreeDEPNDecoder()

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)
    latent_vectors = torch.normal(mean=0, std=1, size=(len(train_dataset), config['latent_code_length']), device=device)
    latent_vectors.requires_grad = True
    # latent_vectors = torch.nn.Embedding(len(train_dataset), config['latent_code_length'], max_norm=1.0)
    # latent_log_var = torch.randn_like(latent_vectors, device=device)
    latent_log_var = torch.zeros(len(train_dataset), config['latent_code_length'], device=device)
    latent_log_var.requires_grad = config['vad_free']
    # Create folder for saving checkpoints
    Path(f'exercise_3/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, latent_vectors, latent_log_var, device, config)
