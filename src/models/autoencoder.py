import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional
import numpy as np
import os
import pickle
from config import DATA_DIR, MODELS_DIR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import random



class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        num_layers: int = 3,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        activation: str = 'relu'
    ):
        """
        Autoencoder with configurable architecture.
        
        Args:
            input_dim: Dimension of input features
            bottleneck_dim: Dimension of compressed representation
            num_layers: Number of layers in encoder (decoder mirrors this)
            use_layer_norm: Whether to use layer normalization
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout probability (0 = no dropout)
            activation: Activation function ('relu', 'leaky_relu', 'tanh')
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_layers = num_layers
        
        encoder_dims = self._generate_layer_dims(input_dim, bottleneck_dim, num_layers)
        decoder_dims = encoder_dims[::-1]  # Mirror for decoder
        
        self.activation = self._get_activation(activation)
        
        self.encoder = self._build_network(
            encoder_dims, 
            use_layer_norm, 
            use_batch_norm, 
            dropout_rate,
            is_encoder=True
        )
        
        # Build decoder
        self.decoder = self._build_network(
            decoder_dims, 
            use_layer_norm, 
            use_batch_norm, 
            dropout_rate,
            is_encoder=False
        )
        
    def _generate_layer_dims(self, start_dim: int, end_dim: int, num_layers: int) -> List[int]:
        """Generate gradually decreasing layer dimensions."""
        if num_layers == 1:
            return [start_dim, end_dim]
        
        dims = np.logspace(
            np.log10(start_dim), 
            np.log10(end_dim), 
            num=num_layers + 1
        )
        dims = [int(d) for d in dims]
        dims[0] = start_dim
        dims[-1] = end_dim
        
        return dims
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'tanh': nn.Tanh(),
            'elu': nn.ELU()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def _build_network(
        self, 
        dims: List[int], 
        use_layer_norm: bool,
        use_batch_norm: bool,
        dropout_rate: float,
        is_encoder: bool
    ) -> nn.Sequential:
        """Build encoder or decoder network."""
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or not is_encoder:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                
                layers.append(self.activation)

                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
    
    def train_model(
        self,
        batches: List[int],
        normalize: bool = True,
        model_name: str = 'autoencoder',
        log_interval: int = 1,
        save_interval: int = 10,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: Optional[str] = None,
    ):
        """
        Train the autoencoder with TensorBoard logging.
        
        Args:
            train_data: Training data (N x input_dim)
            val_data: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            verbose: Whether to print progress
            device: Device to train on
            log_dir: Directory for TensorBoard logs (default: 'runs/')
            experiment_name: Name for this experiment run
        
        Returns:
            Dictionary with training history
        """
        self.to(device)
        if log_dir is None:
            log_dir = 'runs'
        
        experiment_name = f'{model_name}'
        
        writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
        
        writer.add_text('Model/Architecture', str(self))
        writer.add_text('Model/Parameters', f'Total parameters: {sum(p.numel() for p in self.parameters())}')
        
        hparams = {
            'input_dim': self.input_dim,
            'bottleneck_dim': self.bottleneck_dim,
            'num_layers': self.num_layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'epochs': epochs
        }
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        global_step = 0
        for epoch in range(epochs):
            self.train()
            train_losses = []
            batch_losses = []
            print(f'Starting epoch {epoch+1}/{epochs}')

            train_data = _get_batch(batches)
            train_data = train_data.to(device)

            if normalize:
                train_data = (train_data - train_data.mean(dim=0)) / (train_data.std(dim=0) + 1e-8)
            
            indices = torch.randperm(len(train_data))
            for i in range(0, len(train_data), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = train_data[batch_indices]
                
                optimizer.zero_grad()
                recon = self.forward(batch)
                loss = criterion(recon, batch)
                
                loss.backward()
                
                total_grad_norm = 0
                for p in self.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                optimizer.step()
                
                batch_loss = loss.item()
                train_losses.append(batch_loss)
                batch_losses.append(batch_loss)
                
                writer.add_scalar('Loss/train_batch', batch_loss, global_step)
                writer.add_scalar('Gradients/total_norm', total_grad_norm, global_step)
                
                global_step += 1
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/train_std', np.std(train_losses), epoch)
            
            scheduler.step(avg_train_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, epoch)

            if epoch % save_interval == 0:
                torch.save(self.state_dict(), os.path.join(MODELS_DIR, f'{model_name}.pth'))
                print(f'Model saved to {os.path.join(MODELS_DIR, f'{model_name}.pth')}')
            
            if epoch % log_interval == 0:
                for name, param in self.named_parameters():
                    writer.add_histogram(f'Weights/{name}', param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                
                with torch.no_grad():
                    sample_size = min(1000, len(train_data))
                    sample_indices = torch.randperm(len(train_data))[:sample_size]
                    sample_data = train_data[sample_indices]
                    latent = self.encode(sample_data)
                    
                    writer.add_histogram('Latent/values', latent, epoch)
                    writer.add_scalar('Latent/mean', latent.mean().item(), epoch)
                    writer.add_scalar('Latent/std', latent.std().item(), epoch)
            
            if verbose and (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, LR: {current_lr:.2e}")
        
        final_metrics = {
            'final_train_loss': history['train_loss'][-1],
        }
        writer.add_hparams(hparams, final_metrics)        
        writer.close()
        
        if verbose:
            print(f"\nTensorBoard logs saved to: {os.path.join(log_dir, experiment_name)}")
            print(f"Run 'tensorboard --logdir={log_dir}' to view")
        
        return history


def load_batch(batch_number: int):
    batch_file = os.path.join(DATA_DIR, 'XY', 'autoencoder', f'batch_{batch_number}.pkl')
    with open(batch_file, 'rb') as f:
        data = pickle.load(f)
    return torch.FloatTensor(data)


def load_all_batches():
    batch_files = os.listdir(os.path.join(DATA_DIR, 'XY', 'autoencoder'))
    batches = []
    for file in batch_files:
        if not file.endswith('.pkl'):
            continue
        batches.append(int(file.split('_')[1].split('.')[0]))
    return batches


def _get_batch(batches: List[int]):
    while True:
        try:
            batch_idx = random.choice(batches)
            train_data = load_batch(batch_idx)
            train_data = torch.FloatTensor(train_data)
            print(f'Loaded batch {batch_idx} with {len(train_data)} samples')
            return train_data
        except Exception as e:
            print(f"Error loading batch {batch_idx}: {e}")
            continue


# Example usage
if __name__ == "__main__":
    input_dim = 830
    batches = load_all_batches()
    autoencoder = Autoencoder(
        input_dim=input_dim,
        bottleneck_dim=60,
        num_layers=4,
        use_layer_norm=True,
        use_batch_norm=True,
        dropout_rate=0.0
    )
    
    print("Autoencoder Architecture:")
    print(autoencoder)
    print(f"\nTotal parameters: {sum(p.numel() for p in autoencoder.parameters())}")
    
    # Train with TensorBoard logging
    history = autoencoder.train_model(
        batches=batches,
        epochs=500,
        batch_size=64,
        learning_rate=1e-2,
        weight_decay=1e-6,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_name='autoencoder_v2',
        log_interval=1,
        normalize=False
    )


