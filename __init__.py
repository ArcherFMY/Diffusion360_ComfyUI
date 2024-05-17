from .Diffusion360_nodes import VAEDecodeTiledBlended, Diffusion360Sampler

NODE_CLASS_MAPPINGS = {
    "VAEDecodeTiledBlended": VAEDecodeTiledBlended,
    "Diffusion360Sampler": Diffusion360Sampler,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecodeTiledBlended": "VAE Decode (Tiled Blended)",
    "Diffusion360Sampler": "Diffusion360Sampler",
}
