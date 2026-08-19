# ViT from Scratch

Building a Vision Transformer from small pieces so the PyTorch abstractions feel less magical.

The learning code lives here:

```text
learning/vit/
```

Current pieces:

- custom `Module` base class
- neuron, layer, MLP, and ReLU
- LayerNorm
- image patch embedding
- PyTorch `nn.Module` versions for comparison

Next piece:

- attention
