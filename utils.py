import torch
import comfy
from comfy import model_management
from tqdm.auto import trange
import comfy.k_diffusion.utils as utils
import latent_preview
from comfy.samplers import KSAMPLER, ksampler, CFGGuider
from comfy.extra_samplers import uni_pc


def sampler_object(name):
    if name == "uni_pc":
        sampler = KSAMPLER(uni_pc.sample_unipc)
    elif name == "uni_pc_bh2":
        sampler = KSAMPLER(uni_pc.sample_unipc_bh2)
    elif name == "ddim":
        sampler = ksampler("euler", inpaint_options={"random": True})
    else:
        sampler = ksampler(name)
    return sampler


def sample_(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)


def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None,
           force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    if sigmas is None:
        sigmas = self.sigmas

    if last_step is not None and last_step < (len(sigmas) - 1):
        sigmas = sigmas[:last_step + 1]
        if force_full_denoise:
            sigmas[-1] = 0

    if start_step is not None:
        if start_step < (len(sigmas) - 1):
            sigmas = sigmas[start_step:]
        else:
            if latent_image is not None:
                return latent_image
            else:
                return torch.zeros_like(noise)

    sampler = sampler_object(self.sampler)
    sampler.sampler_function = sample_euler_blend

    return sample_(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options,
                  latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar,
                  seed=seed)


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    sampler = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
    sampler.sample = sample.__get__(sampler, KSAMPLER)
    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=None, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())

    out = latent.copy()
    out["samples"] = samples
    return (out, )


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)


@torch.no_grad()
def blend_h(a, b, blend_extent):
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, x] = a[:, :, :, -blend_extent
                                   + x] * (1 - x / blend_extent) + b[:, :, :, x] * (
                                x / blend_extent)
    return b


@torch.no_grad()
def sample_euler_blend(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    w = x.shape[-1]
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
        x = blend_h(x, x, 4)
    x = blend_h(x, x, 4)
    x = x[:, :, :, :w]
    return x


def decode_tiled_blended_(self, samples, tile_x=64, tile_y=64, overlap=16):
    steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y,
                                                                 overlap)
    steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2,
                                                                  tile_y * 2, overlap)
    steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2,
                                                                  tile_y // 2, overlap)
    pbar = comfy.utils.ProgressBar(steps)

    decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
    output = self.process_output(
        (tiled_scale_blended(samples, decode_fn, tile_x // 2, tile_y * 2, overlap,
                                         upscale_amount=self.upscale_ratio, output_device=self.output_device,
                                         pbar=pbar) +
         tiled_scale_blended(samples, decode_fn, tile_x * 2, tile_y // 2, overlap,
                                         upscale_amount=self.upscale_ratio, output_device=self.output_device,
                                         pbar=pbar) +
         tiled_scale_blended(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=self.upscale_ratio,
                                         output_device=self.output_device, pbar=pbar))
        / 3.0)
    return output


def decode_tiled_blended(self, samples, tile_x=64, tile_y=64, overlap = 16):
    model_management.load_model_gpu(self.patcher)
    output = self.decode_tiled_blended_(samples, tile_x, tile_y, overlap)
    return output.movedim(1, -1)


def blend_h(a, b, blend_extent):
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, x] = a[:, :, :, -blend_extent
                          + x] * (1 - x / blend_extent) + b[:, :, :, x] * (
                              x / blend_extent)
    return b


def tiled_scale_blended(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)), device=output_device)
    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        out = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device=output_device)
        out_div = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device=output_device)
        w = samples.shape[3]
        samples = torch.cat([samples, samples[:, :, :, :w // 4]], dim=-1)
        s = samples[b:b + 1]
        for y in range(0, s.shape[2], tile_y - overlap):
            # for x in range(0, s.shape[3], tile_x - overlap):
            #     x = max(0, min(s.shape[-1] - overlap, x))
            y = max(0, min(s.shape[-2] - overlap, y))
            s_in = s[:,:,y:y+tile_y,:]

            ps = function(s_in).to(output_device)
            ps = blend_h(
                ps[:,:,:,w*upscale_amount:],
                ps[:,:,:,:w*upscale_amount],
                w // 4 * upscale_amount
            )
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                    mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                    mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                    mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                    mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
            out[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),:] += ps * mask
            out_div[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),:] += mask
            if pbar is not None:
                pbar.update(1)

        output[b:b+1] = out/out_div
    return output