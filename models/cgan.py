"""
Conditional GAN components for tabular data.
Supports categorical and continuous conditioning for active-learning augmentation.
"""
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(dims: List[int], activation: nn.Module) -> nn.Sequential:
    """Construct a simple MLP with the provided dimensions."""
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i != len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def _encode_condition(
    labels: torch.Tensor, conditional_type: str, num_classes: Optional[int], cond_dim: int
) -> torch.Tensor:
    if conditional_type == "categorical":
        return F.one_hot(labels.long(), num_classes=num_classes).float()
    # continuous
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.shape[1] < cond_dim:
        labels = F.pad(labels, (0, cond_dim - labels.shape[1]))
    return labels[:, :cond_dim].float()


class ConditionalGenerator(nn.Module):
    """Generator conditioned on categorical or continuous labels."""

    def __init__(
        self,
        noise_dim: int,
        output_dim: int,
        conditional_type: str = "categorical",
        num_classes: Optional[int] = None,
        cond_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.conditional_type = conditional_type
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        activation_layer = nn.ReLU if activation == "relu" else nn.GELU
        cond_width = num_classes if conditional_type == "categorical" else cond_dim
        dims = [noise_dim + cond_width] + hidden_dims + [output_dim]
        self.model = _build_mlp(dims, activation_layer)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = _encode_condition(labels, self.conditional_type, self.num_classes, self.cond_dim)
        x = torch.cat([noise, cond], dim=1)
        return self.model(x)


class ConditionalDiscriminator(nn.Module):
    """Discriminator conditioned on categorical or continuous labels."""

    def __init__(
        self,
        input_dim: int,
        conditional_type: str = "categorical",
        num_classes: Optional[int] = None,
        cond_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.input_dim = input_dim
        self.conditional_type = conditional_type
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        if activation == "leaky_relu":
            activation_layer = lambda: nn.LeakyReLU(0.2)
        else:
            activation_layer = nn.ReLU

        cond_width = num_classes if conditional_type == "categorical" else cond_dim
        dims = [input_dim + cond_width] + hidden_dims + [1]
        self.model = _build_mlp(dims, activation_layer)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = _encode_condition(labels, self.conditional_type, self.num_classes, self.cond_dim)
        data = torch.cat([x, cond], dim=1)
        return self.model(data)


class CGAN:
    """
    Conditional GAN wrapper for tabular data.
    Categorical conditioning: uses one-hot labels.
    Continuous conditioning: concatenates target values.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: Optional[int] = None,
        conditional_type: str = "categorical",  # "categorical" or "continuous"
        cond_dim: int = 1,
        noise_dim: int = 32,
        g_hidden: Optional[List[int]] = None,
        d_hidden: Optional[List[int]] = None,
        device: str = "cpu",
        lr: float = 2e-4,
    ):
        self.device = device
        self.noise_dim = noise_dim
        self.conditional_type = conditional_type
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        self.generator = ConditionalGenerator(
            noise_dim=noise_dim,
            output_dim=input_dim,
            conditional_type=conditional_type,
            num_classes=num_classes,
            cond_dim=cond_dim,
            hidden_dims=g_hidden,
        ).to(device)

        self.discriminator = ConditionalDiscriminator(
            input_dim=input_dim,
            conditional_type=conditional_type,
            num_classes=num_classes,
            cond_dim=cond_dim,
            hidden_dims=d_hidden,
        ).to(device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )

    def _train_discriminator(self, real_x: torch.Tensor, real_y: torch.Tensor) -> float:
        batch_size = real_x.size(0)
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_x = self.generator(noise, real_y)

        self.d_optimizer.zero_grad()
        real_logits = self.discriminator(real_x, real_y)
        fake_logits = self.discriminator(fake_x.detach(), real_y)

        loss_real = self.criterion(real_logits, valid)
        loss_fake = self.criterion(fake_logits, fake)
        d_loss = 0.5 * (loss_real + loss_fake)
        d_loss.backward()
        self.d_optimizer.step()
        return float(d_loss.item())

    def _train_generator(self, labels: torch.Tensor) -> float:
        batch_size = labels.size(0)
        valid = torch.ones(batch_size, 1, device=self.device)
        self.g_optimizer.zero_grad()

        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_x = self.generator(noise, labels)
        logits = self.discriminator(fake_x, labels)
        g_loss = self.criterion(logits, valid)
        g_loss.backward()
        self.g_optimizer.step()
        return float(g_loss.item())

    def fit(
        self, dataloader: torch.utils.data.DataLoader, num_steps: int = 200
    ) -> Tuple[List[float], List[float]]:
        """
        Train the CGAN for a limited number of steps.

        Returns:
            discriminator_losses, generator_losses
        """
        d_losses: List[float] = []
        g_losses: List[float] = []

        step = 0
        while step < num_steps:
            for real_x, real_y in dataloader:
                real_x = real_x.to(self.device)
                real_y = real_y.to(self.device)

                d_loss = self._train_discriminator(real_x, real_y)
                g_loss = self._train_generator(labels=real_y)

                d_losses.append(d_loss)
                g_losses.append(g_loss)
                step += 1
                if step >= num_steps:
                    break

        return d_losses, g_losses

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        """Generate synthetic samples conditioned on provided labels (categorical or continuous)."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(labels.size(0), self.noise_dim, device=self.device)
            synth = self.generator(noise, labels.to(self.device))
        return synth.cpu()

    def sample_balanced(self, per_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a balanced set of synthetic samples for all classes."""
        if self.conditional_type != "categorical" or self.num_classes is None:
            raise ValueError("sample_balanced is only valid for categorical conditioning.")
        labels = torch.arange(self.num_classes).repeat_interleave(per_class)
        synth = self.sample(labels)
        return synth, labels

    def sample_from_continuous(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic samples conditioned on provided continuous labels.
        Labels should be float tensor; will be returned alongside samples.
        """
        if self.conditional_type != "continuous":
            raise ValueError("sample_from_continuous is only valid for continuous conditioning.")
        synth = self.sample(labels)
        return synth, labels.cpu()
