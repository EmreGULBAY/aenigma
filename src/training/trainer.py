"""
Training loop for TimeGAN with three-phase optimization.
"""
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

from ..models.timegan import TimeGAN
from .losses import TimeGANLoss

logger = logging.getLogger(__name__)


class TimeGANTrainer:
    """
    Trainer for TimeGAN with three-phase training:
    1. Autoencoder training (Embedder + Recovery)
    2. Supervised training (Supervisor + Generator)
    3. Joint adversarial training (All networks)
    """
    
    def __init__(
        self,
        model: TimeGAN,
        config: Dict,
        output_dir: str = 'outputs'
    ):
        """
        Initialize trainer.
        
        Args:
            model: TimeGAN model
            config: Training configuration
            output_dir: Directory to save outputs
        """
        self.model = model
        self.config = config
        self.device = model.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loss_fn = TimeGANLoss(device=self.device)
        
        self.history = {
            'autoencoder': [],
            'supervisor': [],
            'generator': [],
            'discriminator': [],
            'embedding': []
        }
        
        self._setup_optimizers()
        
        logger.info("TimeGAN Trainer initialized")
    
    def _setup_optimizers(self):
        """Setup optimizers for different network components."""
        lr = self.config['training']['learning_rate']
        beta1 = self.config['training']['beta1']
        beta2 = self.config['training']['beta2']
        
        param_groups = self.model.get_optimizer_groups()
        
        self.optimizer_autoencoder = optim.Adam(
            param_groups['autoencoder'] + param_groups['hotel_embedding'],
            lr=lr,
            betas=(beta1, beta2)
        )
        
        self.optimizer_supervisor = optim.Adam(
            param_groups['supervisor'],
            lr=lr,
            betas=(beta1, beta2)
        )
        
        self.optimizer_generator = optim.Adam(
            param_groups['generator'] + param_groups['hotel_embedding'],
            lr=lr,
            betas=(beta1, beta2)
        )
        
        self.optimizer_discriminator = optim.Adam(
            param_groups['discriminator'] + param_groups['hotel_embedding'],
            lr=lr,
            betas=(beta1, beta2)
        )
        
        self.optimizer_embedding = optim.Adam(
            param_groups['autoencoder'],
            lr=lr,
            betas=(beta1, beta2)
        )
    
    def _create_dataloader(
        self,
        sequences: np.ndarray,
        hotel_ids: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create DataLoader from numpy arrays.
        
        Args:
            sequences: Sequence data (num_samples, seq_len, num_features)
            hotel_ids: Hotel identifiers (num_samples,)
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader
        """
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        hotel_ids_tensor = torch.LongTensor(hotel_ids).to(self.device)
        
        dataset = TensorDataset(sequences_tensor, hotel_ids_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True
        )
        
        return dataloader
    
    def train_autoencoder(
        self,
        train_sequences: np.ndarray,
        train_hotel_ids: np.ndarray
    ) -> float:
        """
        Phase 1: Train autoencoder (Embedder + Recovery).
        
        Args:
            train_sequences: Training sequences
            train_hotel_ids: Training hotel IDs
            
        Returns:
            Average loss
        """
        logger.info("=" * 50)
        logger.info("Phase 1: Training Autoencoder")
        logger.info("=" * 50)
        
        epochs = self.config['training']['epochs_autoencoder']
        batch_size = self.config['data']['batch_size']
        
        dataloader = self._create_dataloader(
            train_sequences, train_hotel_ids, batch_size
        )
        
        best_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['patience']
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for x_real, _ in pbar:
                h, x_tilde = self.model.forward_autoencoder(x_real)
                
                loss = self.loss_fn.reconstruction_loss(x_real, x_tilde)
                
                self.optimizer_autoencoder.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.embedder.parameters(),
                    self.config['training']['gradient_clip']
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model.recovery.parameters(),
                    self.config['training']['gradient_clip']
                )
                
                self.optimizer_autoencoder.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = np.mean(epoch_losses)
            self.history['autoencoder'].append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss - self.config['training']['min_delta']:
                best_loss = avg_loss
                patience_counter = 0
                self.model.save_checkpoint(
                    str(self.output_dir / 'autoencoder_best.pt'),
                    epoch,
                    {'autoencoder': self.optimizer_autoencoder},
                    {'autoencoder': avg_loss}
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return best_loss
    
    def train_supervisor(
        self,
        train_sequences: np.ndarray,
        train_hotel_ids: np.ndarray
    ) -> float:
        """
        Phase 2: Train supervisor for temporal consistency.
        
        Args:
            train_sequences: Training sequences
            train_hotel_ids: Training hotel IDs
            
        Returns:
            Average loss
        """
        logger.info("=" * 50)
        logger.info("Phase 2: Training Supervisor")
        logger.info("=" * 50)
        
        epochs = self.config['training']['epochs_supervised']
        batch_size = self.config['data']['batch_size']
        
        dataloader = self._create_dataloader(
            train_sequences, train_hotel_ids, batch_size
        )
        
        best_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['patience']
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for x_real, hotel_ids in pbar:
                with torch.no_grad():
                    h, _ = self.model.forward_autoencoder(x_real)
                
                h_supervised = self.model.forward_supervisor(h)
                
                loss_supervised = self.loss_fn.supervised_loss(h, h_supervised)
                
                z = torch.randn_like(h)
                e, _ = self.model.forward_generator(z, hotel_ids)
                e_supervised = self.model.forward_supervisor(e)
                
                loss_generator = self.loss_fn.generator_loss_supervised(e, e_supervised)
                
                loss = loss_supervised + loss_generator
                
                self.optimizer_supervisor.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.supervisor.parameters(),
                    self.config['training']['gradient_clip']
                )
                
                self.optimizer_supervisor.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = np.mean(epoch_losses)
            self.history['supervisor'].append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss - self.config['training']['min_delta']:
                best_loss = avg_loss
                patience_counter = 0
                self.model.save_checkpoint(
                    str(self.output_dir / 'supervisor_best.pt'),
                    epoch,
                    {'supervisor': self.optimizer_supervisor},
                    {'supervisor': avg_loss}
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return best_loss
    
    def train_joint(
        self,
        train_sequences: np.ndarray,
        train_hotel_ids: np.ndarray
    ) -> Dict[str, float]:
        """
        Phase 3: Joint adversarial training.
        
        Args:
            train_sequences: Training sequences
            train_hotel_ids: Training hotel IDs
            
        Returns:
            Dictionary of average losses
        """
        logger.info("=" * 50)
        logger.info("Phase 3: Joint Adversarial Training")
        logger.info("=" * 50)
        
        epochs = self.config['training']['epochs_joint']
        batch_size = self.config['data']['batch_size']
        
        lambda_recon = self.config['training']['lambda_reconstruction']
        lambda_super = self.config['training']['lambda_supervised']
        lambda_adv = self.config['training']['lambda_adversarial']
        
        dataloader = self._create_dataloader(
            train_sequences, train_hotel_ids, batch_size
        )
        
        best_g_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {
                'generator': [],
                'discriminator': [],
                'embedding': []
            }
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for x_real, hotel_ids in pbar:
                batch_size_actual = x_real.size(0)
                seq_len = x_real.size(1)
                
                self.optimizer_discriminator.zero_grad()
                
                h_real, _ = self.model.forward_autoencoder(x_real)
                d_real = self.model.forward_discriminator(h_real.detach(), hotel_ids)
                
                z = torch.randn(
                    batch_size_actual, seq_len, self.model.hidden_dim,
                    device=self.device
                )
                e_fake, _ = self.model.forward_generator(z, hotel_ids)
                d_fake = self.model.forward_discriminator(e_fake.detach(), hotel_ids)
                
                loss_d = self.loss_fn.discriminator_loss(d_real, d_fake)
                
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.discriminator.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.optimizer_discriminator.step()
                
                epoch_losses['discriminator'].append(loss_d.item())
                
                self.optimizer_generator.zero_grad()
                self.optimizer_supervisor.zero_grad()
                
                z = torch.randn(
                    batch_size_actual, seq_len, self.model.hidden_dim,
                    device=self.device
                )
                e_fake, x_fake = self.model.forward_generator(z, hotel_ids)
                
                d_fake = self.model.forward_discriminator(e_fake, hotel_ids)
                loss_g_adv = self.loss_fn.generator_loss_unsupervised(d_fake)
                
                e_supervised = self.model.forward_supervisor(e_fake)
                loss_g_super = self.loss_fn.generator_loss_supervised(e_fake, e_supervised)
                
                loss_g_moment = self.loss_fn.generator_loss_moment(x_real, x_fake)
                
                loss_g = (
                    lambda_adv * loss_g_adv +
                    lambda_super * loss_g_super +
                    loss_g_moment
                )
                
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.generator.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.optimizer_generator.step()
                self.optimizer_supervisor.step()
                
                epoch_losses['generator'].append(loss_g.item())
                
                self.optimizer_embedding.zero_grad()
                
                h_real, x_tilde = self.model.forward_autoencoder(x_real)
                loss_e_recon = self.loss_fn.reconstruction_loss(x_real, x_tilde)
                
                h_supervised = self.model.forward_supervisor(h_real)
                loss_e_super = self.loss_fn.embedding_loss(h_real, h_supervised)
                
                loss_e = lambda_recon * loss_e_recon + lambda_super * loss_e_super
                
                loss_e.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.embedder.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.optimizer_embedding.step()
                
                epoch_losses['embedding'].append(loss_e.item())
                
                pbar.set_postfix({
                    'G': f"{loss_g.item():.3f}",
                    'D': f"{loss_d.item():.3f}",
                    'E': f"{loss_e.item():.3f}"
                })
            
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            
            for k, v in avg_losses.items():
                self.history[k].append(v)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"G: {avg_losses['generator']:.4f}, "
                f"D: {avg_losses['discriminator']:.4f}, "
                f"E: {avg_losses['embedding']:.4f}"
            )
            
            if avg_losses['generator'] < best_g_loss:
                best_g_loss = avg_losses['generator']
                self.model.save_checkpoint(
                    str(self.output_dir / 'timegan_best.pt'),
                    epoch,
                    {
                        'generator': self.optimizer_generator,
                        'discriminator': self.optimizer_discriminator,
                        'embedding': self.optimizer_embedding
                    },
                    avg_losses
                )
            
            if (epoch + 1) % 100 == 0:
                self.model.save_checkpoint(
                    str(self.output_dir / f'timegan_epoch_{epoch+1}.pt'),
                    epoch,
                    {
                        'generator': self.optimizer_generator,
                        'discriminator': self.optimizer_discriminator,
                        'embedding': self.optimizer_embedding
                    },
                    avg_losses
                )
        
        return avg_losses
    
    def train(
        self,
        train_sequences: np.ndarray,
        train_hotel_ids: np.ndarray
    ):
        """
        Full three-phase training pipeline.
        
        Args:
            train_sequences: Training sequences
            train_hotel_ids: Training hotel IDs
        """
        logger.info("Starting TimeGAN training pipeline")
        logger.info(f"Training samples: {len(train_sequences)}")
        logger.info(f"Sequence length: {train_sequences.shape[1]}")
        logger.info(f"Number of features: {train_sequences.shape[2]}")
        
        self.train_autoencoder(train_sequences, train_hotel_ids)
        
        self.train_supervisor(train_sequences, train_hotel_ids)
        
        self.train_joint(train_sequences, train_hotel_ids)
        
        logger.info("Training completed!")
        
        self.model.save_checkpoint(
            str(self.output_dir / 'timegan_final.pt'),
            -1,
            {
                'generator': self.optimizer_generator,
                'discriminator': self.optimizer_discriminator,
                'embedding': self.optimizer_embedding
            },
            {k: v[-1] if v else 0 for k, v in self.history.items()}
        )