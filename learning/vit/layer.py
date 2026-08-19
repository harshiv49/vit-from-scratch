import torch
import torch.functional as F 
from .module import Module
import torch.nn as nn 
from .neuron import Neuron, NeuronTorchCompatible, TorchNeuron


class Layer:
    def __init__(self,number_of_neurons:int,number_of_inputs:int):
        self.neurons = [ Neuron(number_of_inputs) for _ in range(number_of_neurons)  ]

    def forward(self,x:torch.Tensor) -> torch.Tensor:
          outputs = []
          for neuron in self.neurons:
              outputs.append(neuron.forward(x))
          return torch.stack(outputs,dim=-1) 

    def parameters(self) -> list[torch.Tensor]:
        params = []
        for neuron in self.neurons:
            # appends would add the lists with a new dimension extend will just add the element of the list 
            params.extend(neuron.parameters())
        return params


class LayerTorchCompatible(Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # we need to add neurons to the submodules array since neuron in itself is also a module
        # output_dim means how many neurons this layer has
        self.neurons = [NeuronTorchCompatible(input_dim) for _ in range(output_dim)]
        
        for neuron in self.neurons:
          self.add_module(neuron)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       outputs = []

       for neuron in self.neurons:
           # the reason this is possible is since we are defining a call interface 
           outputs.append(neuron(x))

       return torch.stack(outputs, dim=-1)
  
    def parameters(self):
        return super().parameters()
        


class TorchLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.neurons = nn.ModuleList(
            [TorchNeuron(input_dim) for _ in range(output_dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(x))
        return torch.stack(outputs, dim=-1)
