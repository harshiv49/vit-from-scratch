
import torch
import torch.functional as F 
import torch.nn as nn 
from .module import Module

class Neuron:
    def __init__(self,number_of_inputs:int):
        self.weights  = torch.randn(number_of_inputs,requires_grad=True)
        self.bias = torch.randn((),requires_grad=True)
 
    def forward(self,x):
        # one input vector * one weight vector + bias
        return x @ self.weights + self.bias

    def parameters(self):
        return [self.weights,self.bias]

class NeuronTorchCompatible(Module):
    def __init__(self,number_of_inputs:int):
        super().__init__()
        self.weights  = torch.randn(number_of_inputs,requires_grad=True)
        self.bias = torch.randn((),requires_grad=True) 
 
    def forward(self,x) -> torch.Tensor:
        # one input vector * one weight vector + bias
        return x @ self.weights + self.bias

    def parameters(self) -> list[torch.Tensor]:
        return [self.weights,self.bias]


class TorchNeuron(nn.Module):
    def __init__(self, number_of_inputs: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(number_of_inputs))
        self.bias = nn.Parameter(torch.randn(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights + self.bias
