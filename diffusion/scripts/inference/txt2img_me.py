import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat
from torchvision.utils import make_grid
from datetime import datetime  # Import datetime module

import sys
sys.path.append('/apdcephfs/private_laurelgui/projects/textual_inversion')
# sys.path.append('/media/limo/change_data/code/AnomalyGeneration/AnoGen-main/DIFFUSION/ldm')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torch.nn.modules.container import ParameterDict
torch.serialization.add_safe_globals([ParameterDict])

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256,256))
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = (image/127.5 - 1.0).astype(np.float32)
    return image

def preprocess_mask(mask_path, h, w):
    mask = Image.open(mask_path).convert('1')
    mask_resize = mask.resize((w, h))
    return 1-np.array(mask_resize).astype(np.float32)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)  # Load model checkpoint
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,  # originally 256
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,  # originally 256
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,  # originally 4
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/data/pretrained_models/ldm/text2img-large/model.ckpt", 
        help="Path to pretrained ldm text2img model")

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")
    
    parser.add_argument(
        "--image_prompt",
        type=str,
        help="image to prompt with, must specify a mask",
        default=None
    )

    parser.add_argument(
        "--mask_prompt",
        type=str,
        help="mask to prompt with, must specify image prompt",
        default=None
    )
    parser.add_argument(
        "--sample_name",
        type=str,
        help="sample_name",
        default=None
    )
    parser.add_argument(
        "--defect_name",
        type=str,
        help="defect_name",
        default=None
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        help="dataset_type",
        required=True,
        default=None
    )
    # parser.add_argument(
    #     "--adaptive_mask",
    #     type=bool,
    #     help="adaptive_mask",
    #     default=False
    # )


    opt = parser.parse_args()
    print("Embedding path:", opt.embedding_path)
    print("Checkpoint path:", opt.ckpt_path)


    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path
    model.embedding_manager.load(opt.embedding_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)  # originally DDIMSampler(model)


    image_prompt = opt.image_prompt
    mask_prompt = opt.mask_prompt

    # by default not do inpaint
    x0 = None
    mask = None

    # inpaint
    if image_prompt and mask_prompt:
        print("Using image as x0: " + image_prompt)
        print("Using mask image: " + mask_prompt)
        image_prompt_input = preprocess_image(image_prompt)
        image_prompt_input = rearrange(image_prompt_input, 'h w c -> c h w')
        image_prompt_input = torch.from_numpy(image_prompt_input)
        image_prompt_input = image_prompt_input.to(memory_format=torch.contiguous_format).float()
        image_prompt_input = repeat(image_prompt_input, 'c h w -> b c h w', b=opt.n_samples).to(device)
        encoder_posterior = model.encode_first_stage(image_prompt_input)  # Encode preprocessed input image
        x0 = model.get_first_stage_encoding(encoder_posterior).detach()  # Encoded result used as x0
        h = opt.H//opt.f
        w = opt.W//opt.f
        mask_prompt_input = preprocess_mask(mask_prompt, h, w)  # Preprocess mask
        mask = torch.tensor(mask_prompt_input)
        mask = repeat(mask, 'h w -> b h w', b=opt.n_samples).to(device)
        mask = mask[:, None, ...]
        print("mask shape:", mask.shape)


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt  # Input text prompt
    print('prompt',prompt)
    # Get current timestamp string
    # current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # Get current timestamp string
    # current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # sample_path = os.path.join(outpath, current_time+"_samples")  # Output directory for generated images
    sample_path = os.path.join(outpath, f"{opt.sample_name}")  # Output directory for generated images
    os.makedirs(sample_path, exist_ok=True)
    # Ensure directories exist before saving images
    images_dir = os.path.join(sample_path, 'images')
    os.makedirs(images_dir, exist_ok=True)
    # Ensure directories exist before saving images
    masks_dir = os.path.join(sample_path, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    base_count = len(os.listdir(images_dir))  # Image counter

    all_samples=list()  # Store generated images
    with torch.no_grad():  # Disable gradient computation
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:  # Unconditional guidance sampling
                
                uc = model.get_learned_conditioning(opt.n_samples * [""])  # Empty text conditioning
                print('uc',uc.shape)
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])  # Text conditioning
                shape = [4, opt.H//opt.f, opt.W//opt.f]
                print('c',c.shape)
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,  # Number of sampling steps
                                                 conditioning=c,  # Text conditioning
                                                 batch_size=opt.n_samples,  # Batch size
                                                 shape=shape,  # Image shape
                                                 verbose=False,  # Verbose logging
                                                 unconditional_guidance_scale=opt.scale,  # Guidance scale
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x0=x0,
                                                 mask=mask,
                                                 )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    
                    # Define image and mask save paths first
                    if opt.dataset_type == 'visa':
                        img_save_path = os.path.join(images_dir, f"{opt.sample_name}_{base_count:04}.jpg")
                        mask_save_path = os.path.join(masks_dir, f"{opt.sample_name}_{base_count:04}.png")
                    elif opt.dataset_type == 'mvtec':
                        img_save_path = os.path.join(images_dir, f"{opt.sample_name}_{opt.defect_name}_{base_count:03}.jpg")
                        mask_save_path = os.path.join(masks_dir, f"{opt.sample_name}_{opt.defect_name}_{base_count:03}.png")
                    else:
                        raise ValueError(f"Unknown dataset_type: {opt.dataset_type}")

                    Image.fromarray(x_sample.astype(np.uint8)).save(img_save_path)

                    # Save corresponding mask
                    if opt.mask_prompt is not None:
                        try:
                            mask_img = Image.open(opt.mask_prompt)
                            mask_img.save(mask_save_path)
                            print(f"Mask image saved to: {mask_save_path}")
                        except Exception as e:
                            print(f"Error saving mask image: {e}")
                    
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    # grid = torch.stack(all_samples, 0)
    # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    # grid = make_grid(grid, nrow=opt.n_samples)

    # # to image
    # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    # # Get current timestamp string
    # # current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # # # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.jpg'))
    # # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{current_time}.jpg'))
    # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'4_{opt.sample_name}_{opt.defect_name}_{base_count:04}.jpg'))
    

    print(f"Your samples are ready and waiting four you here: \n{sample_path} \nEnjoy.")
