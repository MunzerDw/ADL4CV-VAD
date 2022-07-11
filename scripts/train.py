import json
import sys
import os
import argparse
from pathlib import Path
import torch
import torch.distributions as dist
from pytorch_lightning.loggers import TensorBoardLogger

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.join(os.getcwd()))

from model.threedepn import ThreeDEPNDecoder
from data.shapenet import ShapeNet
from scripts.evaluate import IOU

def train(model, train_dataloader, latent_vectors, latent_log_var, device, config):
    split = 'train' if not config['test'] else 'val'

    # Initialize logger
    logger = TensorBoardLogger("logs", name=config['experiment_name'])
    logger.log_graph(model)
    logger.log_hyperparams(config)
    # Define loss
    reconstruction_loss_criterion = torch.nn.L1Loss()
    reconstruction_loss_criterion.to(device)

    # Define params
    params = [
        {
            'params': latent_vectors,
            'lr': config['learning_rate_code']
        }
    ]
    if config['vad']:
        params.append({
            'params': latent_log_var,
            'lr': config['learning_rate_log_var']
        })
    
    if not config['test']:
        params.append({
            'params': model.parameters(),
            'lr': config['learning_rate_decoder']
        })
    if len(params) == 1:
        print(f'Training latent codes')
    if len(params) == 2 and config['vad']:
        print(f'Training latent codes and variances')
    if len(params) == 2 and not config['vad']:
        print(f'Training latent codes and decoder weights')
    if len(params) == 3:
        print(f'Training latent codes, variances, and decoder weights')

    # Define Optimizers
    optimizer = torch.optim.Adam(params)

    # Here, we follow the original implementation to also use a learning rate scheduler -- it simply reduces the learning rate to half every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=0.5)

    model.train()

    # Keep track of running average of train loss for printing
    best_loss = float('inf')
    train_loss_running = 0.
    reconstruction_loss_running = 0.
    kl_loss_running = 0.
    n = 0
    saved_model = False
    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device, set optimizer gradients to zero, perform forward pass
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()
            if config['vad']:

            # Create distribution from latent codes and laten variances (each sample has its own distribution)
                Dist = dist.Normal(latent_vectors[batch['index']], torch.exp(latent_log_var[batch['index']]))

                # Sample from each distribution
                x_vad = Dist.rsample()

                # Forward pass
                reconstruction = model(x_vad)
            else:
                reconstruction = model(latent_vectors[batch['index']])
            
            # Targets
            target = batch['target_df']
            q_z = dist.Normal(0, 1)

            # Compute loss, Compute gradients, Update network parameters
            reconstruction_loss = reconstruction_loss_criterion(reconstruction, target)
            if config['vad']:
                kl_loss = torch.mean(dist.kl_divergence(Dist, q_z))
                loss = reconstruction_loss + config['kl_weight'] * kl_loss
            else:
                loss = reconstruction_loss
            # Compute gradients
            loss.backward()

            # Update network parameters
            optimizer.step()

            # Logging
            train_loss_running += loss.item()
            reconstruction_loss_running += reconstruction_loss.item()
            if config['vad']:
                kl_loss_running += kl_loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                n = n + 1
                samples_per_epoch = config["print_every_n"]
                train_loss = train_loss_running / samples_per_epoch
                print(f"[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / samples_per_epoch:.6f}"
                      f' kl_loss: {kl_loss_running/ samples_per_epoch:.6f} normal_loss: {reconstruction_loss_running / samples_per_epoch:.6f}')
                logger.log_metrics({
                    'train_loss': train_loss_running / samples_per_epoch,
                    'kl_loss': kl_loss_running/ samples_per_epoch,
                    'reconstruction_loss': reconstruction_loss_running / samples_per_epoch,
                    'iter': n
                }, n)
                train_loss_running = 0.
                reconstruction_loss_running = 0.
                kl_loss_running = 0.

                # Save model checkpoint
                if train_loss < best_loss:
                    torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_best.ckpt')
                    torch.save(latent_vectors, f'runs/{config["experiment_name"]}/latent_best_{split}.pt')
                    torch.save(latent_log_var, f'runs/{config["experiment_name"]}/log_var_best_{split}.pt')
                    best_loss = train_loss
                    saved_model = True
        
        # Update KL-ratio
        if (epoch % config['kl_weight_increase_every_epochs'] == (config['kl_weight_increase_every_epochs'] - 1)) and (config['decoder_var']):
            config['kl_weight'] = config['kl_weight'] + config['kl_weight_increase_value']
            logger.log_metrics({
                'kl_weight': config['kl_weight'],
            }, epoch)
            print(f"[{epoch:03d}/{batch_idx:05d}] updated kl_weight: {config['kl_weight']}")

        # IOU
        if (epoch % config['iou_every_epoch'] == (config['iou_every_epoch'] - 1)) and (not config['decoder_var']) and saved_model:
            iou = IOU(config['experiment_name'], 'train', config['filter_class'], device)
            logger.log_metrics({
                'IOU': iou,
            }, epoch)
            print(f"[{epoch:03d}/{batch_idx:05d}] IOU {iou}")
            
        # Update scheduler          
        scheduler.step()

def parse_arguments():
    classes = ['airplane', 'car', 'chair', 'sofa', 'lamp', 'cabine', 'watercraft', 'table']

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('filter_class', choices=classes, type=str)
    parser.add_argument('--resume', help='resume the training of the experiment', action='store_true')
    parser.add_argument('--test', help='freeze decoder weights and train on validation set', action='store_true')
    parser.add_argument('--vad', help='train a variational auto-decoder', action='store_true')
    parser.add_argument('--kl_weight', type=int, default=0.03)
    parser.add_argument('--kl_weight_increase_every_epochs', help='increase kl-weight every how many epochs', type=int, default=100)
    parser.add_argument('--kl_weight_increase_value', help='increase kl-weight by how much (addition)', type=int, default=0.0)
    parser.add_argument('--latent_code_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate_decoder', type=int, default=0.01)
    parser.add_argument('--learning_rate_code', type=int, default=0.01)
    parser.add_argument('--learning_rate_log_var', type=int, default=0.01)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--print_every_n', type=int, default=100)
    parser.add_argument('--scheduler_step_size', type=int, default=100)
    parser.add_argument('--iou_every_epoch', help='calculate IOU every how many epochs (only for non-variational models)', type=int, default=50)
    parser.add_argument('--cpu', help='disable cuda', action='store_true')
    parser.add_argument('--num_workers', help='number of workers for the dataloader', type=int, default=4)

    args = parser.parse_args()
    return vars(args)

def main():
    # read arguments
    args = parse_arguments()
    split = 'train' if not args['test'] else 'val'

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and not args['cpu']:
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        print('Using CPU')

    # Create Dataloaders
    dataset = ShapeNet(split, filter_class=args['filter_class'])
    print(f"Data length ({args['filter_class']}): {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args['batch_size'], 
        shuffle=True,
        num_workers=args['num_workers'],
        pin_memory=True,
    )

    # Instantiate model
    model = ThreeDEPNDecoder()

    # Initialize latent codes and latent variance
    latent_vectors = torch.randn(size=(len(dataset), args['latent_code_length']), device=device)
    latent_vectors.requires_grad = True
    latent_log_var = torch.zeros(len(dataset), args['latent_code_length'], device=device)
    latent_log_var.requires_grad = args['vad']

    # Load model if testing
    if args['test'] and not args['resume']:
        print('Loading saved model for testing...')
        model.load_state_dict(torch.load(f"runs/{args['experiment_name']}/model_best.ckpt", map_location=device))

    # Load model if resuming from checkpoint
    if args['resume']:
        print('Loading saved model, latent codes, and latent variances...')
        model.load_state_dict(torch.load(f"runs/{args['experiment_name']}/model_best.ckpt", map_location=device))
        latent_vectors = torch.load(f"runs/{args['experiment_name']}/latent_best_{split}.pt", map_location = device)
        latent_log_var = torch.load(f"runs/{args['experiment_name']}/log_var_best_{split}.pt", map_location = device)

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'runs/{args["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Save config
    with open(f'runs/{args["experiment_name"]}/config.json', 'w') as f:
        json.dump(args, f)

    # Start training
    train(model, dataloader, latent_vectors, latent_log_var, device, args)

if __name__ == '__main__':
    try:
        main()
        sys.stdout.flush()
    except KeyboardInterrupt:
        print('Interrupted...')
        print('Best model, latent codes and variances are stored under runs/<experiment name>')
        print("You can resume an experiment by setting the '--resume' flag")
        try:
            sys.exit(0)
        except:
            os._exit(0)

# conda activate adl4cv
# python scripts/train.py car_exp car 