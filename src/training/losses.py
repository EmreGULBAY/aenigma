"""
Loss functions for TimeGAN training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeGANLoss:
    """
    Collection of loss functions for TimeGAN training.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize loss functions.
        
        Args:
            device: Device to compute losses on
        """
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def reconstruction_loss(
        self, 
        x_real: torch.Tensor, 
        x_reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruction loss for autoencoder (Embedder + Recovery).
        
        Args:
            x_real: Real sequences (batch, seq_len, input_dim)
            x_reconstructed: Reconstructed sequences (batch, seq_len, input_dim)
            
        Returns:
            MSE loss
        """
        return self.mse_loss(x_reconstructed, x_real)
    
    def supervised_loss(
        self, 
        h_real: torch.Tensor, 
        h_supervised: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised loss for temporal consistency.
        Predicts h[t+1] from h[t].
        
        Args:
            h_real: Real latent sequences (batch, seq_len, hidden_dim)
            h_supervised: Supervised predictions (batch, seq_len, hidden_dim)
            
        Returns:
            MSE loss between h[1:] and supervised(h[:-1])
        """
        return self.mse_loss(h_supervised[:, :-1, :], h_real[:, 1:, :])
    
    def generator_loss_unsupervised(
        self, 
        d_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Unsupervised generator loss (fool discriminator).
        
        Args:
            d_fake: Discriminator output for fake data (batch, 1)
            
        Returns:
            BCE loss (generator wants discriminator to output 1)
        """
        real_labels = torch.ones_like(d_fake)
        return self.bce_loss(d_fake, real_labels)
    
    def generator_loss_supervised(
        self, 
        h_fake: torch.Tensor, 
        h_supervised_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised generator loss (temporal consistency).
        
        Args:
            h_fake: Generated latent sequences (batch, seq_len, hidden_dim)
            h_supervised_fake: Supervised predictions on fake (batch, seq_len, hidden_dim)
            
        Returns:
            MSE loss for temporal consistency
        """
        return self.mse_loss(h_supervised_fake[:, :-1, :], h_fake[:, 1:, :])
    
    def generator_loss_moment(
        self, 
        x_real: torch.Tensor, 
        x_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Moment matching loss (match mean and variance).
        
        Args:
            x_real: Real sequences (batch, seq_len, input_dim)
            x_fake: Fake sequences (batch, seq_len, input_dim)
            
        Returns:
            Combined mean and variance loss
        """
        mean_real = torch.mean(x_real, dim=[0, 1])
        mean_fake = torch.mean(x_fake, dim=[0, 1])
        loss_mean = self.mse_loss(mean_fake, mean_real)
        
        var_real = torch.var(x_real, dim=[0, 1])
        var_fake = torch.var(x_fake, dim=[0, 1])
        loss_var = self.mse_loss(var_fake, var_real)
        
        return loss_mean + loss_var
    
    def discriminator_loss(
        self, 
        d_real: torch.Tensor, 
        d_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Discriminator loss (distinguish real from fake).
        
        Args:
            d_real: Discriminator output for real data (batch, 1)
            d_fake: Discriminator output for fake data (batch, 1)
            
        Returns:
            Combined BCE loss
        """
        real_labels = torch.ones_like(d_real)
        loss_real = self.bce_loss(d_real, real_labels)
        
        fake_labels = torch.zeros_like(d_fake)
        loss_fake = self.bce_loss(d_fake, fake_labels)
        
        return loss_real + loss_fake
    
    def embedding_loss(
        self, 
        h_real: torch.Tensor, 
        h_supervised: torch.Tensor
    ) -> torch.Tensor:
        """
        Embedding network loss during joint training.
        
        Args:
            h_real: Real latent sequences (batch, seq_len, hidden_dim)
            h_supervised: Supervised predictions (batch, seq_len, hidden_dim)
            
        Returns:
            MSE loss for temporal consistency
        """
        return self.mse_loss(h_supervised[:, :-1, :], h_real[:, 1:, :])


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    hotel_ids: torch.Tensor,
    lambda_gp: float = 10.0
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP (optional enhancement).
    
    Args:
        discriminator: Discriminator network
        real_data: Real sequences
        fake_data: Fake sequences
        hotel_ids: Hotel identifiers
        lambda_gp: Gradient penalty coefficient
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    device = real_data.device
    
    alpha = torch.rand(batch_size, 1, 1, device=device)
    
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    d_interpolates = discriminator(interpolates, hotel_ids)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    
    return gradient_penalty