#! /usr/bin/env python3

# ==============================================================================
# Copyright© The University of Texas at Austin, 2025. All rights reserved.
# Author: Alex Diaz (j.a.diaz@utexas.edu)
# 
# About
# --------------
# Base class to inherit in custom models
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class BaseSSCModel(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()

        # Load .yaml configuration
        self.cfg = cfg
        self.num_class = cfg['model']['num_classes']
           
    @abstractmethod
    def forward(self, x):
        """
        Abstract forward pass method
        """   
        raise NotImplementedError
    
    @abstractmethod
    def compute_loss(self, prediction, target):
        """
        Abstract loss method
        """   
        raise NotImplementedError

    def predict(self, x):
        """
        Evaluation method
        """   
        self.eval()

        with torch.no_grad():
            return self.forward(x)
        
