"""Guidance"""
from abc import ABC, abstractmethod


class Guidance(ABC):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    @abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError
