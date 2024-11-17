import abc
from abc import ABC, abstractmethod
import numpy as np
import sympy as sym
from PIL import Image
from matplotlib.patches import FancyBboxPatch, FancyArrow
import os
from matplotlib import patheffects as pe

import matplotlib.pyplot as plt

class Dynamics(ABC):
    pass
    # @abstractmethod
    # def build_symbolic_dynamics(self):
    #     pass

    # @abstractmethod
    # def get_symbolic_states(self):
    #     pass

    # @abstractmethod
    # def get_symbolic_controls(self):
    #     pass

    # @abstractmethod
    # def get_symbolic_rates(self):
    #     pass

    # @abstractmethod
    # def __str__():
    #     pass

    # @abstractmethod
    # def __repr__():
    #     pass

