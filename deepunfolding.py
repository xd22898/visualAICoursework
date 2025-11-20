import copy
import os
import random
import math
import webbrowser

from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale
from tqdm import tqdm

