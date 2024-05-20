from .Diffusion360_nodes import VAEDecodeTiledBlended, Diffusion360Sampler
from .Diffusion360_nodes_diffusers import (
    Diffusion360LoaderText2Pano,
    Diffusion360SamplerText2Pano,
    Diffusion360LoaderImage2Pano,
    Diffusion360SamplerImage2Pano,
    InputText,
    InputImage,
)

NODE_CLASS_MAPPINGS = {
    "VAEDecodeTiledBlended": VAEDecodeTiledBlended,
    "Diffusion360Sampler": Diffusion360Sampler,
    "InputText": InputText,
    "Diffusion360LoaderText2Pano": Diffusion360LoaderText2Pano,
    "Diffusion360SamplerText2Pano": Diffusion360SamplerText2Pano,
    "Diffusion360LoaderImage2Pano": Diffusion360LoaderImage2Pano,
    "Diffusion360SamplerImage2Pano": Diffusion360SamplerImage2Pano,
    "InputImage": InputImage,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecodeTiledBlended": "VAE Decode (Tiled Blended)",
    "Diffusion360Sampler": "Diffusion360Sampler",
    "InputText": "InputText",
    "Diffusion360LoaderText2Pano": "LoadText2Pano",
    "Diffusion360SamplerText2Pano": "SamplerText2Pano",
    "Diffusion360LoaderImage2Pano": "LoadImage2Pano",
    "Diffusion360SamplerImage2Pano": "SamplerImage2Pano",
    "InputImage": "InputImage",

}
