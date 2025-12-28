"""
Neural network architectures for TimeGAN.
Includes Embedder, Recovery, Generator, Discriminator, and Supervisor networks.
"""
import torch
import torch.nn as nn
from typing import Tuple


class EmbedderNetwork(nn.Module):
    """
    Embedder network: Maps real sequences to latent representations.
    Input: Real sequences (batch, seq_len, input_dim)
    Output: Latent sequences (batch, seq_len, hidden_dim)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super(EmbedderNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            
        Returns:
            Embedded sequences (batch, seq_len, hidden_dim)
        """
        h, _ = self.rnn(x)
        
        h = self.fc(h)
        h = self.activation(h)
        
        return h


class RecoveryNetwork(nn.Module):
    """
    Recovery network: Maps latent representations back to original space.
    Input: Latent sequences (batch, seq_len, hidden_dim)
    Output: Reconstructed sequences (batch, seq_len, output_dim)
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        output_dim: int, 
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super(RecoveryNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h: Latent sequences (batch, seq_len, hidden_dim)
            
        Returns:
            Reconstructed sequences (batch, seq_len, output_dim)
        """
        h_decoded, _ = self.rnn(h)
        
        x_tilde = self.fc(h_decoded)
        x_tilde = self.activation(x_tilde)
        
        return x_tilde


class GeneratorNetwork(nn.Module):
    """
    Generator network: Generates synthetic latent sequences from noise.
    Input: Random noise (batch, seq_len, noise_dim) + hotel_embedding
    Output: Synthetic latent sequences (batch, seq_len, hidden_dim)
    """
    
    def __init__(
        self, 
        noise_dim: int,
        hidden_dim: int,
        hotel_embedding_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super(GeneratorNetwork, self).__init__()
        
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.hotel_embedding_dim = hotel_embedding_dim
        self.num_layers = num_layers
        
        self.input_dim = noise_dim + hotel_embedding_dim
        
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        
    def forward(
        self, 
        z: torch.Tensor, 
        hotel_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Random noise (batch, seq_len, noise_dim)
            hotel_emb: Hotel embeddings (batch, hotel_embedding_dim)
            
        Returns:
            Generated latent sequences (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = z.shape
        
        hotel_emb_expanded = hotel_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        z_cond = torch.cat([z, hotel_emb_expanded], dim=-1)
        
        e, _ = self.rnn(z_cond)
        
        e = self.fc(e)
        e = self.activation(e)
        
        return e


class DiscriminatorNetwork(nn.Module):
    """
    Discriminator network: Distinguishes real from synthetic latent sequences.
    Input: Latent sequences (batch, seq_len, hidden_dim) + hotel_embedding
    Output: Probability of being real (batch, 1)
    """
    
    def __init__(
        self, 
        hidden_dim: int,
        hotel_embedding_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super(DiscriminatorNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.hotel_embedding_dim = hotel_embedding_dim
        self.num_layers = num_layers
        
        self.input_dim = hidden_dim + hotel_embedding_dim
        
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        h: torch.Tensor, 
        hotel_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h: Latent sequences (batch, seq_len, hidden_dim)
            hotel_emb: Hotel embeddings (batch, hotel_embedding_dim)
            
        Returns:
            Discrimination logits (batch, 1)
        """
        batch_size, seq_len, _ = h.shape
        
        hotel_emb_expanded = hotel_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        h_cond = torch.cat([h, hotel_emb_expanded], dim=-1)
        
        _, h_last = self.rnn(h_cond)
        
        h_last = h_last[-1]
        
        logits = self.fc(h_last)
        
        return logits


class SupervisorNetwork(nn.Module):
    """
    Supervisor network: Predicts next latent state for temporal consistency.
    Input: Latent sequences (batch, seq_len, hidden_dim)
    Output: Next-step predictions (batch, seq_len-1, hidden_dim)
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(SupervisorNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h: Latent sequences (batch, seq_len, hidden_dim)
            
        Returns:
            Next-step predictions (batch, seq_len, hidden_dim)
        """
        h_supervised, _ = self.rnn(h)
        
        h_supervised = self.fc(h_supervised)
        h_supervised = self.activation(h_supervised)
        
        return h_supervised


class HotelEmbedding(nn.Module):
    """
    Hotel embedding layer: Learns embeddings for different hotels.
    """
    
    def __init__(self, num_hotels: int, embedding_dim: int):
        super(HotelEmbedding, self).__init__()
        
        self.num_hotels = num_hotels
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(
            num_embeddings=num_hotels + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
    def forward(self, hotel_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hotel_ids: Hotel identifiers (batch,)
            
        Returns:
            Hotel embeddings (batch, embedding_dim)
        """
        return self.embedding(hotel_ids)