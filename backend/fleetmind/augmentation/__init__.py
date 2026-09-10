"""FleetMind augmentation subpackage (Phase 3: TimeGAN-lite for rare faults)."""

from fleetmind.augmentation.gan import ConditionalWaveGAN, Discriminator, Generator

__all__ = ["ConditionalWaveGAN", "Generator", "Discriminator"]
