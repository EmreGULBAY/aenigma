"""
Main TimeGAN model that orchestrates all networks.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging

from .networks import (
    EmbedderNetwork,
    RecoveryNetwork,
    GeneratorNetwork,
    DiscriminatorNetwork,
    SupervisorNetwork,
    HotelEmbedding
)

logger = logging.getLogger(__name__)


class TimeGAN(nn.Module):
    """
    Time-series Generative Adversarial Network with hotel conditioning.
    
    Architecture:
    - Embedder: Real sequences -> Latent space
    - Recovery: Latent space -> Real space
    - Generator: Noise + Hotel ID -> Latent sequences
    - Discriminator: Latent sequences + Hotel ID -> Real/Fake
    - Supervisor: Latent sequences -> Next-step prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_hotels: int = 3,
        hotel_embedding_dim: int = 16,
        dropout: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize TimeGAN model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of RNN layers
            num_hotels: Number of different hotels
            hotel_embedding_dim: Dimension of hotel embeddings
            dropout: Dropout rate
            device: Device to run on
        """
        super(TimeGAN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_hotels = num_hotels
        self.hotel_embedding_dim = hotel_embedding_dim
        self.device = device
        
        self.hotel_embedding = HotelEmbedding(num_hotels, hotel_embedding_dim)
        
        self.embedder = EmbedderNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.recovery = RecoveryNetwork(
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.generator = GeneratorNetwork(
            noise_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hotel_embedding_dim=hotel_embedding_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.discriminator = DiscriminatorNetwork(
            hidden_dim=hidden_dim,
            hotel_embedding_dim=hotel_embedding_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.supervisor = SupervisorNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers - 1,
            dropout=dropout
        )
        
        self.to(device)
        
        logger.info(f"TimeGAN initialized on {device}")
        logger.info(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}")
        logger.info(f"Total parameters: {self.count_parameters():,}")
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward_autoencoder(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through embedder and recovery (autoencoder).
        
        Args:
            x: Real sequences (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (latent_sequences, reconstructed_sequences)
        """
        h = self.embedder(x)
        x_tilde = self.recovery(h)
        return h, x_tilde
    
    def forward_supervisor(
        self, 
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through supervisor for next-step prediction.
        
        Args:
            h: Latent sequences (batch, seq_len, hidden_dim)
            
        Returns:
            Next-step predictions (batch, seq_len, hidden_dim)
        """
        return self.supervisor(h)
    
    def forward_generator(
        self, 
        z: torch.Tensor, 
        hotel_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through generator and recovery.
        
        Args:
            z: Random noise (batch, seq_len, hidden_dim)
            hotel_ids: Hotel identifiers (batch,)
            
        Returns:
            Tuple of (synthetic_latent, synthetic_sequences)
        """
        hotel_emb = self.hotel_embedding(hotel_ids)
        e = self.generator(z, hotel_emb)
        x_hat = self.recovery(e)
        return e, x_hat
    
    def forward_discriminator(
        self, 
        h: torch.Tensor, 
        hotel_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            h: Latent sequences (batch, seq_len, hidden_dim)
            hotel_ids: Hotel identifiers (batch,)
            
        Returns:
            Discrimination logits (batch, 1)
        """
        hotel_emb = self.hotel_embedding(hotel_ids)
        return self.discriminator(h, hotel_emb)
    
    def generate(
        self, 
        num_samples: int, 
        seq_len: int, 
        hotel_ids: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate synthetic sequences.
        
        Args:
            num_samples: Number of sequences to generate
            seq_len: Length of sequences
            hotel_ids: Hotel identifiers for conditioning (num_samples,)
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated sequences (num_samples, seq_len, input_dim)
        """
        self.eval()
        
        with torch.no_grad():
            z = torch.randn(
                num_samples, seq_len, self.hidden_dim, 
                device=self.device
            ) * temperature
            
            _, x_hat = self.forward_generator(z, hotel_ids)
        
        return x_hat
    
    def get_optimizer_groups(self) -> Dict[str, list]:
        """
        Get parameter groups for different training phases.
        
        Returns:
            Dictionary of parameter groups
        """
        return {
            'autoencoder': list(self.embedder.parameters()) + 
                          list(self.recovery.parameters()),
            'supervisor': list(self.supervisor.parameters()) + 
                         list(self.generator.parameters()),
            'generator': list(self.generator.parameters()) + 
                        list(self.supervisor.parameters()),
            'discriminator': list(self.discriminator.parameters()),
            'hotel_embedding': list(self.hotel_embedding.parameters())
        }
    
    def save_checkpoint(self, path: str, epoch: int, optimizers: Dict, losses: Dict):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizers: Dictionary of optimizers
            losses: Dictionary of losses
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'losses': losses,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_hotels': self.num_hotels,
                'hotel_embedding_dim': self.hotel_embedding_dim
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, optimizers: Optional[Dict] = None):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            optimizers: Optional dictionary of optimizers to load states
            
        Returns:
            Dictionary containing epoch and losses
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizers is not None and 'optimizers' in checkpoint:
            for k, v in optimizers.items():
                if k in checkpoint['optimizers']:
                    v.load_state_dict(checkpoint['optimizers'][k])
        
        logger.info(f"Checkpoint loaded from {path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'losses': checkpoint.get('losses', {})
        }