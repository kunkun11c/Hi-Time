"""Fit the paper's residual K-Means targets on frozen training features."""
import torch
from torch.utils.data import DataLoader

from configs import Config
from models.encoder import Stage1Encoder
from models.discretizer import Stage2Discretizer


def train_stage2(config: Config, encoder: Stage1Encoder, train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 device: torch.device = None) -> Stage2Discretizer:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.eval()
    discretizer = Stage2Discretizer(
        d_model=config.model.d_model,
        codebook_size=config.model.codebook_size,
        num_rq_layers=config.model.num_quantize_layers,
        num_classes=config.data.num_classes,
        ema_decay=config.model.ema_decay,
    ).to(device)
    with torch.no_grad():
        features = torch.cat([encoder.encode(x.to(device))
                              for x, _ in train_loader], dim=0)
        # stage2_epochs is the Lloyd iteration limit (Sec. 4.2).
        discretizer.fit(features, max_iter=config.training.stage2_epochs,
                        seed=config.training.seed)
        output = discretizer(features)
        for i, (codes, residual) in enumerate(zip(output['codes'],
                                                  output['residuals'])):
            print(f'Stage2 layer {i + 1}: '
                  f'{codes.unique().numel()}/{config.model.codebook_size} codes, '
                  f'residual MSE={residual.square().mean().item():.6f}')
    return discretizer.eval()
