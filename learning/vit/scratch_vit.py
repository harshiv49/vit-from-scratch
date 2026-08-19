import torch
import torch.nn.functional as F 
import torch.nn as nn 

from .config import VitConfig
from .neuron import Neuron


class LayerNorm(nn.Module):
    """
    The patch embeddings output is B,N,D 
    B is the batch size , N is number of patches and D is the dimension of embeddings
    B, N, D = 2, 4, 3  
    LayerNorm normalizes each of these token vectors independently:

    x[0, 0, :]
    x[0, 1, :]
    x[0, 2, :]
    x[0, 3, :]
    x[1, 0, :]
    x[1, 1, :]
    x[1, 2, :]
    x[1, 3, :]

    Take one token:
    x = [1, 2, 3]
    D = 3

    ### Step 1: Mean
    μ = (1 + 2 + 3) / 3
    μ = 2

    ### Step 2: Variance

    σ² = ((1 - 2)² + (2 - 2)² + (3 - 2)²) / 3
    σ² = (1 + 0 + 1) / 3
    σ² = 2/3
    σ² ≈ 0.6667

    ### Step 3: Standard deviation

    σ = sqrt(2/3)
    σ ≈ 0.8165

    ### Step 4: Normalize each value

    x̂1 = (1 - 2) / 0.8165 ≈ -1.2247
    x̂2 = (2 - 2) / 0.8165 = 0
    x̂3 = (3 - 2) / 0.8165 ≈ 1.2247

    ## Steo 5: Scale and Shift 
    yi = γi * x̂i + βi
    """
    def __init__(self):
        super().__init__()
        
    def layer_norm_manual(self,x:torch.Tensor, gamma:torch.Tensor | None =None, beta:torch.Tensor | None =None, eps=1e-5):
        # We are calculating B * N means here and keeping the dimension 
        mean = x.mean(dim=-1,keepdim=True) # B * N * 1 
        # means get copied across the row D times 
        variance = (x - mean).mean(dim=-1,keepdim=True) #  ( B * N * D - B * N * 1)
        # sqrt of variance divided by all B * N * D elements 
        x_normalized = (x - mean) / torch.sqrt(variance + eps)
        if gamma is None:
           gamma = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)

        if beta is None:
           beta = torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)

        output = gamma * x_normalized + beta
        return output

    def layer_norm_torch(self,x:torch.Tensor, gamma:torch.Tensor | None =None, beta:torch.Tensor | None =None, eps=1e-5):
        embedding_dim = x.shape[-1]

        if gamma is None:
           gamma = torch.ones(embedding_dim, device=x.device, dtype=x.dtype)

        if beta is None:
           beta = torch.zeros(embedding_dim, device=x.device, dtype=x.dtype)

        output = F.layer_norm(
           x,
           normalized_shape=(embedding_dim,),
           weight=gamma,
           bias=beta,
           eps=eps,
           )

        return output


class Head(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.config = config
        self.key = torch.nn.Linear(self.config.embedding_dimension, self.config.head_size, bias=False)
        self.value = torch.nn.Linear(self.config.embedding_dimension, self.config.head_size, bias=False)
        self.query = torch.nn.Linear(self.config.embedding_dimension, self.config.head_size, bias=False)

    

class Block:
    def __init__(self):
        pass

    # this is an encoder based transformer 
    # layer norm 
    # self attention
    # layer norm 
    # multi layer perceptron 

class ImagePatchEmbedding:

    def __init__(self, config: VitConfig):
        self.config = config
        self.patch_projection = torch.nn.Linear(
            in_features=self.config.patch_dim,
            out_features=self.config.embedding_dimension 
        )
        self.unfolder = torch.nn.Unfold(
            kernel_size=(self.config.patch_height, self.config.patch_width),
            stride=(self.config.patch_height, self.config.patch_width),
        )

    # image into image embeddings 
    def convert_image_patches_to_embeddings_manual(self, images: torch.Tensor):
        all_patches = []
        batch_size,channels,height,width = images.shape
        
        for row in range(0,height,self.config.patch_height):
            for col in range(0,width,self.config.patch_width): 
                # patch (batch_size, channel , patch_height , patch_width)
                patch = images[:,:,row : row + self.config.patch_height,col: col + self.config.patch_width]
                # we flatten from the channel 
                patch = torch.flatten(patch,start_dim=1)
                # (8,48)
                all_patches.append(patch)

        # For a square patch size P, assuming the image divides evenly: N = (H / P) × (W / P) = H × W / P². 
        # With H = W = 32 and P = 4, there are 8 × 8 = 64 patches.
        # torch.stack combines multiple tensors by adding a new dimension.
        patches = torch.stack(all_patches,dim=1)    
        image_tokens = self.patch_projection(patches)
        return image_tokens


    def convert_image_patches_to_embeddings_torch(self):
        pass

    def convert_image_patches_to_embeddings_torch_nn(self, images: torch.Tensor):
        #  1. Extracts each patch.
        #  2. Flattens each patch.
        #  3. Places the flattened patches into one tensor.
        patches_unfold = self.unfolder(images)
        patches_unfold = patches_unfold.transpose(1, 2).contiguous()
        # torch.Size([8, 64, 48])
        image_tokens = self.patch_projection(patches_unfold)
        return image_tokens

config = VitConfig()
image_patch_embeddings = ImagePatchEmbedding(config)


