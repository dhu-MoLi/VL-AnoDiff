"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from utils import Dilation2d
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
    # New helper functions
    def calculate_distance_map(self, img, x0, mask, alpha=0.8):
        """Compute improved distance map using mixed L1+L2 distance."""
        # L2 distance component
        l2_dist = ((img - x0) ** 2).mean(dim=1)
        # L1 distance component - more robust to outliers
        l1_dist = torch.abs(img - x0).mean(dim=1)
        # Combined distance for better balance
        combined_dist = alpha * l2_dist + (1-alpha) * l1_dist
        # Apply mask
        masked_dist = mask.squeeze(1) * combined_dist
        return masked_dist
    def generate_weight_map(self, dis_map, mask, min_weight=0.8, max_weight=2.0):
        """Generate improved weight map."""
        # Handle zero values
        safe_dis_map = torch.where(dis_map == 0, 
                                  torch.tensor(1e-6, dtype=dis_map.dtype, device=dis_map.device), 
                                  dis_map)
        
        bs = dis_map.size(0)
        pixel_count = mask.view(bs, -1).sum(dim=1).view(bs, 1, 1)
        value_sum = safe_dis_map.view(bs, -1).sum(dim=1).view(bs, 1, 1)
        
        # Avoid division by zero
        safe_value_sum = torch.where(value_sum == 0,
                                     torch.ones_like(value_sum),
                                     value_sum)
        
        # Compute weights with smooth transition instead of hard threshold
        raw_weights = pixel_count * safe_dis_map / safe_value_sum
        
        # Apply sigmoid to smooth weight changes
        normalized_weights = 2.0 / (1.0 + torch.exp(-3.0 * (raw_weights - 0.5))) - 0.5
        scaled_weights = min_weight + (max_weight - min_weight) * normalized_weights
        
        # Apply weights only inside mask region
        final_weights = torch.where(mask.squeeze(1) > 0,
                                   scaled_weights,
                                   torch.ones_like(scaled_weights))
        
        return final_weights
        
    def adaptive_dilation(self, mask, timestep, img, max_dilation=5):
        """Adaptive dilation based on timestep and content."""
        # Base dilation rate derived from timestep
        # base_dilation = int((timestep + 100) * max_dilation // 1000)
        
        # # Create dilation operator
        # dilation = Dilation2d(m=2)
        # dilated_mask = dilation(mask)
        # if timestep < 200:
        #     dilated_mask = mask  # Skip dilation in final steps
        #     print('no dilation')
        #     # print('timestep',timestep)
        # else:
            # print('timestep',timestep)
        # base_dilation = int((timestep + 100) * max_dilation // 1000)
        base_dilation = 1
        # base_dilation = max(1, base_dilation)  # Ensure at least 1
        # print('base_dilation',base_dilation)
        dilation = Dilation2d(m=base_dilation)
        dilated_mask = dilation(mask)
        # dilated_mask = mask
        
        return dilated_mask

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    
                                                    )
        return samples, intermediates
 

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,adaptive_mask=False):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)  # Sample random noise when x_T is not provided
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        # Optional utility to save mask as image
        # def save_mask_as_image(mask_tensor, filename_prefix, step):
        #     """Convert mask tensor to image and save."""
        #     import torchvision.utils as vutils
        #     import os
            
        #     # Ensure output directory exists
        #     os.makedirs('mask_visualizations', exist_ok=True)
            
        #     # Clone mask and move to CPU
        #     mask_img = mask_tensor.detach().cpu()
            
        #     # Save original mask
        #     filename = f'mask_visualizations/{filename_prefix}_step{step}.png'
        #     vutils.save_image(mask_img, filename, normalize=True)
        #     print(f"Saved mask visualization to {filename}")
        # def save_mask_as_image(mask_tensor, filename_prefix, step, current_ts):
        #     """Convert mask tensor to image and save."""
        #     import torchvision.utils as vutils
        #     import os
            
        #     # Ensure output directory exists
        #     os.makedirs('mask_visualizations', exist_ok=True)
            
        #     # Clone mask and move to CPU
        #     mask_img = mask_tensor.detach().cpu()
            
        #     # Save mask without normalization, include timestep in filename
        #     filename = f'mask_visualizations/{filename_prefix}_ts{current_ts}_step{step}.png'
        #     vutils.save_image(mask_img, filename, normalize=False)
        #     print(f"Saved mask visualization to {filename}, ts={current_ts}")
        def save_mask_as_image(mask_tensor, filename_prefix, step, current_ts):
            """Convert mask tensor to image and save."""
            import torchvision.utils as vutils
            import os
            
            # Ensure output directory exists
            os.makedirs('mask_visualizations', exist_ok=True)
            
            # Clone mask and move to CPU
            mask_img = mask_tensor.detach().cpu()
            
            # Print mask value range
            print(f"Mask value range: min={mask_img.min().item():.4f}, max={mask_img.max().item():.4f}")
            
            # Save mask without normalization, include timestep in filename
            filename = f'mask_visualizations/{filename_prefix}_ts{current_ts}_step{step}.png'
            vutils.save_image(mask_img, filename, normalize=False)
            print(f"Saved mask visualization to {filename}, ts={current_ts}")

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            print(f'ts: {ts}')
            
            weight_map=None
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                # print('img_orig',img_orig.shape)#b*4*16*16
                # print('mask',mask.shape)#b*1*16*16
                # img = img_orig * mask + (1. - mask) * img
                '''Inpainting modification section'''
                
                # Save original mask
                # save_mask_as_image(mask, "original_mask", i)
                new_mask = mask
                if adaptive_mask:
                    # Use optimized adaptive dilation
                    new_mask = self.adaptive_dilation(mask, ts[0].item(), img)
                    print('ts',ts[0].item())
                    # print('timestep',timesteps)
                    # new_mask = self.adaptive_dilation(mask, timesteps, img)
                    
                    # Use improved distance calculation
                    # dis_map = self.calculate_distance_map(img, x0, new_mask)
                    
                    # Generate optimized weight map
                    # weight_map = self.generate_weight_map(dis_map, new_mask)
                    # print('weight_map',weight_map.shape)
                    # # Handle edge cases
                    # for tmpi in range(b):
                    #     if dis_map[tmpi].view(-1).sum() < 1e-6:
                    #         weight_map[tmpi,:,:] = 1
                else:
                    new_mask = mask
                    weight_map = None
                    # ==== Original code start ====
                    # if len(timesteps)==50:
                    #     # dialation = Dilation2d(m=int((ts[0] + 100) * 6// 1000))
                    #     # print(f'm: {int((ts[0] + 100) * 6// 1000)}')
                    #     # m_value = int((ts[0] + 100) * 1// 1000)
                    #     new_mask = self.adaptive_dilation(mask, ts[0].item(), img)

                    #     # # print(f'Before dilation - ts: {ts[0]}, m: {m_value}')
                    #     # # print(f'Original mask value range: min={mask.min().item():.4f}, max={mask.max().item():.4f}')
        
                    #     # dialation = Dilation2d(m=m_value)
                    #     # print(f'Current ts: {ts[0]}, m: {m_value}')
                    # else:
                    #     dialation = Dilation2d(m=int((ts[0] + 100) * 1 // 1000))
                    # ==== Original code end ====
                    
                    # # ==== Alternative 1 - reduce mask expansion range ====
                    # # To use option 1, comment out the original code above and uncomment below
                    # if len(timesteps)==50:
                    #     # Reduce coefficient from 16 to 4 and cap maximum at 2
                    #     m_value = min(2, int((ts[0] + 100) * 4 // 1000))
                    #     dialation = Dilation2d(m=m_value)
                    # else:
                    #     # Reduce coefficient from 8 to 2 and cap maximum at 1
                    #     m_value = min(1, int((ts[0] + 100) * 2 // 1000))
                    #     dialation = Dilation2d(m=m_value)
                    # # Add logging to observe dilation parameter changes
                    # print(f'adaptive--ok, dilation m={m_value}')
                    # # ==== Alternative 1 end ====
                    
                    # ==== Alternative 2 - use fixed small value ====
                    # To use option 2, comment out all code above and uncomment below
                    # # Use fixed small dilation values for minimal expansion
                    # if len(timesteps)==50:
                    #     dialation = Dilation2d(m=1)  # Fixed value 1
                    # else:
                    #     dialation = Dilation2d(m=1)  # Fixed value 1
                    # print('adaptive--ok, using fixed dilation m=1')
                    # ==== Alternative 2 end ====
                    # # # ==== Alternative 3 - reduce mask expansion range ====
                    # # # To use option 1, comment out the original code above and uncomment below
                    # if len(timesteps)==50:
                    #     # Reduce coefficient from 16 to 4 and cap maximum at 2
                    #     m_value = min(2, int((ts[0] + 100) * 6 // 1000))
                    #     dialation = Dilation2d(m=m_value)
                    # else:
                    #     # Reduce coefficient from 8 to 2 and cap maximum at 1
                    #     m_value = min(1, int((ts[0] + 100) * 4 // 1000))
                    #     dialation = Dilation2d(m=m_value)
                    # # Add logging to observe dilation parameter changes
                    # print(f'adaptive--ok, dilation m={m_value}')
                    # # # ==== Alternative 3 end ====
                    # # ==== Alternative 4 - reduce mask expansion range ====
                    # # To use option 1, comment out the original code above and uncomment below
                    # if len(timesteps)==50:
                    #     # Reduce coefficient from 16 to 4 and cap maximum at 2
                    #     m_value = min(2, int((ts[0] + 100) * 6 // 1000))
                    #     dialation = Dilation2d(m=m_value)
                    # else:
                    #     # Reduce coefficient from 8 to 2 and cap maximum at 1
                    #     m_value = min(1, int((ts[0] + 100) * 4 // 1000))
                    #     dialation = Dilation2d(m=m_value)
                    # # Add logging to observe dilation parameter changes
                    # print(f'adaptive--ok, dilation m={m_value}')
                    # # ==== Alternative 4 end ====
                    
                    # new_mask=dialation(mask)
                    # print('adaptive--ok')
                    # Save original mask
                    # save_mask_as_image(new_mask, "dilated_mask", i)
                    # Pass current ts value
                    # save_mask_as_image(new_mask, "dilated_mask", i, ts[0].item())
                # '''Inpainting modification section'''

                # dis_map = ((new_mask * (img - x0)) ** 2).mean(dim=1)
                # # dis_map = torch.where(dis_map == 0, -1, dis_map)   #b*32*32
                # dis_map=torch.where(dis_map==0,torch.tensor(-1.0,dtype=dis_map.dtype,device=dis_map.device),dis_map)
                # dis_map=1/dis_map
                # dis_map=new_mask.squeeze(1)*dis_map
                # bs=dis_map.size(0)
                # weight_map = new_mask.view(bs, -1).sum(dim=1).view(bs, 1, 1) * dis_map / (
                #     dis_map.view(bs, -1).sum(dim=1).view(bs, 1, 1))  # replace the softm
                # # weight_map = torch.where(weight_map == 0, 1, weight_map)##
                # weight_map = torch.where(weight_map == 0, torch.tensor(1.0, dtype=weight_map.dtype, device=weight_map.device),
                #                       weight_map)
                # # weight_map = torch.where(weight_map <1, 1, weight_map)##
                # weight_map = torch.where(weight_map < 1,
                #                          torch.tensor(1.0, dtype=weight_map.dtype, device=weight_map.device),
                #                          weight_map)

                # # weight_map = torch.where(weight_map > 1.5, 1.5, weight_map)##
                # weight_map = torch.where(weight_map > 1.5,
                #                          torch.tensor(1.5, dtype=weight_map.dtype, device=weight_map.device),
                #                          weight_map)
                # for tmpi in range(bs):
                #     if dis_map[tmpi].view(-1).sum()==0:
                #         weight_map[tmpi,:,:]=1
                    
                # # img = img * new_mask + (1 - new_mask) * img
                img = img_orig * new_mask + (1. - new_mask) * img
            '''Inpainting modification section'''

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec