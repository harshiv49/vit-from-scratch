from abc import ABC, abstractmethod

import torch


class Module(ABC):
    def __init__(self):
        # some moudles have different behavior during training hence this is needed
        self.training = True
        self._parameters = []
        # if there are any non trainable parameters
        self._buffers = []
        self._modules = []

    @abstractmethod
    def forward(self,x) -> torch.Tensor:
        pass

    def __call__(self,x):
        return self.forward(x)

    def parameters(self):
        params = []
        for parameter in self._parameters:
            params.append(parameter)

        for submodule in self._modules:
            params.extend(submodule.parameters())

        return params 

    def add_parameter(self, parameter):
           self._parameters.append(parameter)

    def add_module(self, module):
       self._modules.append(module)

    def train(self):
       self.training = True

       for module in self._modules:
           module.train()

    def eval(self):
       self.training = False

       for module in self._modules:
           module.eval()
