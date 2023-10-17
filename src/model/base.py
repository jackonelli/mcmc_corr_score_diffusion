from abc import ABC, abstractmethod


class EnergyModel(ABC):

    @abstractmethod
    def energy(self, *args, **kwargs):
        raise NotImplementedError
