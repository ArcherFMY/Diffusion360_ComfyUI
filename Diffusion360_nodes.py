import comfy
from comfy.ldm.models.autoencoder import AutoencoderKL
from comfy.samplers import KSAMPLER
from .utils import decode_tiled_blended, decode_tiled_blended_, common_ksampler


class VAEDecodeTiledBlended:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", ), "vae": ("VAE", ),
                             "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64})
                            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "Diffusion360"

    def decode(self, vae, samples, tile_size):
        vae.decode_tiled = decode_tiled_blended.__get__(vae, AutoencoderKL)
        vae.decode_tiled_blended_ = decode_tiled_blended_.__get__(vae, AutoencoderKL)
        return (vae.decode_tiled(samples["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8, ), )


class Diffusion360Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (['euler_blend'], ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Diffusion360"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)



NODE_CLASS_MAPPINGS = {
    "VAEDecodeTiledBlended": VAEDecodeTiledBlended,
    "Diffusion360Sampler": Diffusion360Sampler,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecodeTiledBlended": "VAE Decode (Tiled Blended)",
    "Diffusion360Sampler": "Diffusion360Sampler",
}
