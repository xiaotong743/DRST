"""The model interface `Steerer` requires.
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def prior(self, n, generator=None):
        """[n, ...] draw from the t=0."""

    @abstractmethod
    def endpoint(self, x, t):
        """-> (theta, v): theta = E[x_1|x_t]. One step denoiser """

    @abstractmethod
    def score(self, x, v, t):
        """grad_x log p_t(x)."""

    @abstractmethod
    def eps(self, t):
        """The diffusion coefficient at time t."""
