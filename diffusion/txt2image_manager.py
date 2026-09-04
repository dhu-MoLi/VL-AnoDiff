import glob
import os
import argparse
import random
if __name__ == "__main__":
    mvtec_info = {
        'bottle': ['broken_large', 'broken_small', 'contamination'],
        'cable': ['bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 'cut_outer_insulation', 'missing_cable','missing_wire','poke_insulation'],
        'capsule': ['crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze'],
        'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
        'grid': ['bent', 'broken', 'glue', 'metal_contamination', 'thread'],
        'hazelnut': ['crack', 'cut', 'hole', 'print'],
        'leather': ['color', 'cut', 'fold', 'glue', 'poke'],
        'metal_nut': ['bent', 'color', 'scratch'],
        'pill': ['color', 'combined', 'contamination', 'crack', 'faulty_imprint', 'pill_type', 'scratch'],
        'tile': ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough'],
        'toothbrush': ['defective'],
        'transistor': ['bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
        'transistor': ['misplaced'],
        'wood': ['color', 'combined', 'hole', 'liquid', 'scratch'],
        'zipper': ['broken_teeth', 'combined', 'fabric_border', 'fabric_interior', 'split_teeth', 'rough', 'squeezed_teeth'],
        }
    visa_info = {
        # 'candle': ['None'],
        'capsules': ['None'],
        # 'cashew': ['None'],
        # 'chewinggum': ['None'],
        # 'fryum': ['None'],
        # 'macaroni1': ['None'],
        # 'macaroni2': ['None'],
        # 'pcb1': ['None'],
        # 'pcb2': ['None'],
        # 'pcb3': ['None'],
        # 'pcb4': ['None'],
        # 'pipe_fryum': ['None']
    }
# export PYTHONPATH=$PYTHONPATH:/path/to/AnoGen-main-0504
# python /media/limo/change_data/code/AnomalyGeneration/AnoGen-main/DIFFUSION/scripts/txt2img_me.py

# Generate_path = 'output_0401__logs_adaptive/'
# # Generate_path = 'output222/'
# # mvtec_info = {
# #     'bottle': ['broken_large']
# # }
# for sample_name in mvtec_info.keys():
#         for defect_name in mvtec_info[sample_name]:
#             if True:
#                 print("making {}-{}: ".format(sample_name, defect_name))
#                 root_path = os.path.join(Generate_path, sample_name, defect_name)#
#                 print(root_path)
#                 try:
#                     # os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 1 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'logs/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/001.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/001.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
#                     # os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 1 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'logs/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/002.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/001.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
#                     os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 1 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'logs/{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/001.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/001.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}" )
#                     os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 1 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'logs/{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/002.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/001.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
#                     print("done_{sample_name}_{defect_name}")
#                 except Exception as e:
#                 #     with open('failed_commands.txt', 'a') as f:
#                 #         f.write(f"Failed command: python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 2 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/001.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/001.png' --outdir 'outputs_001_001'/mvtec --sample_name {sample_name} --defect_name {defect_name}\n")
#                 #     print(f"Failed to execute command for {sample_name} - {defect_name}: {e}")
#                     continue             
# print("done_all")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, default="mvtec")
    parser.add_argument("--embedding_path", type=str, required=True, default="log_new/transistor_transistor_transistor/embeddings.pt")
    parser.add_argument("--ckpt_path", type=str, default="models/ldm/text2img-large/model.ckpt")
    parser.add_argument("--prompt", type=str, default="*")
    parser.add_argument("--mask_prompt", type=str, required=True, default="examples/demo/mask.png")
    parser.add_argument("--image_prompt", type=str, required=True, default="datasets/mvtec/transistor/train/good/000.png")
    parser.add_argument("--outname", type=str, required=True, default="temp-generated-output")
# Generate_path = 'mvtec_dataset_0423_adaptive_noadaptive_36_newmask_single_prompt'
    args = parser.parse_args()
    # Generate_path = 'mvtec-transistor-simple-generation'
# mvtec_info = {
#     'hazelnut': ['crack']
# }
    if args.dataset_type == "mvtec":
        for sample_name in mvtec_info.keys():
                for defect_name in mvtec_info[sample_name]:
                    if True:
                        print("making {}-{}: ".format(sample_name, defect_name))
                        root_path = os.path.join(Generate_path, sample_name, defect_name)#
                        print(root_path)
                        try:
                            os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'examples/demo/mask_000.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/000.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'examples/demo/mask_001.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/001.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'examples/demo/mask_002.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/002.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'examples/demo/mask_003.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/003.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'examples/demo/mask_004.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/004.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'examples/demo/mask_005.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/005.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            # os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/002.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/001.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            # os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/003.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/002.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            # os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/004.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/003.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            # os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/005.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/004.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            # os.system(f"python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/006.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/005.png' --outdir {Generate_path}/mvtec --sample_name {sample_name} --defect_name {defect_name}")
                            print("done_{sample_name}_{defect_name}")
                        except Exception as e:
                        #     with open('failed_commands.txt', 'a') as f:
                        #         f.write(f"Failed command: python scripts/inference/txt2img_me.py --ddim_eta 0.0 --n_samples 2 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'log_new/{defect_name}_{sample_name}_{defect_name}/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt '*' --mask_prompt 'mvtec_masks/{sample_name}/{defect_name}/001.png' --image_prompt 'datasets/mvtec/{sample_name}/train/good/001.png' --outdir 'outputs_001_001'/mvtec --sample_name {sample_name} --defect_name {defect_name}\n")
                        #     print(f"Failed to execute command for {sample_name} - {defect_name}: {e}")
                            continue           
    elif args.dataset_type == "visa":
        for sample_name in visa_info.keys():
             if True:
                print("making {}: ".format(sample_name))                
                try:
                    embedding_path = os.path.join(args.embedding_path, sample_name, 'checkpoints/embeddings.pt')
                    print(embedding_path)
                    masks_dir = glob.glob(os.path.join(args.mask_prompt, sample_name, '*.png'))  # mask_prompt e.g. visa_balanced_masks_me_0/visa or visa_masks
                    # print(masks_dir)
                    images_dir = glob.glob(os.path.join(args.image_prompt, sample_name, 'train/ok', '*.JPG'))  # image_prompt e.g. datasets/visa
                    # print(images_dir)
                    # for i in range(len(masks_dir)):
                    for i in range(100):
                        # Get the i-th mask path
                        mask_path = sorted(masks_dir)[i]
                        # Get mask filename
                        mask_filename = os.path.basename(mask_path)
                        # print(mask_filename)
                        # Get corresponding image path
                        for j in range(5):
                            # candle_promt = [
                            #     'A small, irregularly shaped white spot with a slightly rough texture and high contrast against the dark background',
                            #     'A larger, more elongated white streak with a smooth texture and sharp edges, standing out prominently against the dark background',
                            #     'Multiple small, scattered white spots with varying sizes and textures, creating a speckled pattern against the dark background',
                            #     'A small, irregularly shaped area with a lighter color and rough texture, contrasting sharply with the surrounding smooth surface',
                            #     'A cluster of tiny, irregular spots with a slightly darker hue and a granular texture, standing out against the uniform background',
                            #     'A linear scratch with a jagged edge, displaying a lighter shade and a matte finish that disrupts the otherwise smooth and glossy surface',
                            #     'A small, sharp, white protrusion with a high contrast against the dark background',
                            #     'A rough, irregularly shaped indentation with a matte texture and low contrast edges',
                            #     'A smooth, circular depression with a glossy finish and a distinct, sharp edge'
                            # ]
                            prompt_dir = os.path.join('prompts/visa/raw', sample_name, 'prompts')
                            print(prompt_dir)
                            prompt_files = glob.glob(os.path.join(prompt_dir, '*.txt'))
                            # print(promt_files)
                            
                            image_idx = (j + 1) * 160
                            if image_idx < len(images_dir):
                                image_path = sorted(images_dir)[image_idx]
                                # Build output directory
                                outdir = os.path.join('datasets_visa', args.outname)
                                # import ipdb;ipdb.set_trace()
                                prompts = []
                                for prompt_file in prompt_files:
                                    # print(prompt_file)
                                    with open(prompt_file, 'r') as f:
                                        prompts.extend([line.split(':')[1].strip() for line in f.readlines()[2:5]])
                                prompt = random.choice(prompts)
                                print(prompt)
                                # import ipdb;ipdb.set_trace()
                                os.makedirs(outdir, exist_ok=True)                                
                                # Build command
                                cmd = f"python scripts/inference/txt2img_me.py --dataset_type visa --ddim_eta 0.0 --n_samples 2 --n_iter 1 --scale 10.0 --ddim_steps 50 --embedding_path '{embedding_path}' --ckpt_path '{args.ckpt_path}' --prompt '{prompt}' --mask_prompt '{mask_path}' --image_prompt '{image_path}' --outdir {outdir} --sample_name {sample_name}"                                
                                # Execute command
                                os.system(cmd)
                                # import ipdb;ipdb.set_trace()
                                print(f"Processed mask {i} with image {image_idx}: {mask_filename}")                        
                except Exception as e:
                    print(f"Failed to execute command for {sample_name}: {e}")
                    continue
    print("done_all")

#                    python scripts/inference/txt2img_me.py --dataset_type visa --ddim_eta 0.0 --n_samples 3 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path 'visa_test/candle2025-04-27T19-05-44_llm_enhanced_visa_candle_test/checkpoints/embeddings.pt' --ckpt_path 'models/ldm/text2img-large/model.ckpt' --prompt 'A small, circular area with a darker color and a smoother texture, blending in but still noticeable against the uniform surface of the candle *' --mask_prompt 'datasets/visa/candle/ground_truth/ko/006.png' --image_prompt 'datasets/visa/candle/train/ok/0001.JPG' --outdir visa_dataset_test --sample_name candle
#python txt2image_manager.py --dataset_type visa --embedding_path 'visa_10_baseline' --mask_prompt 'visa_masks' --image_prompt 'datasets/visa' --outname 'visa_10_baseline'
#python txt2image_manager.py --dataset_type visa --embedding_path 'weights_visa/me_0429' --mask_prompt 'visa_balanced_masks_me_0' --image_prompt 'datasets/visa' --outname 'visa_me_temp_delete'
#python txt2image_manager.py --dataset_type visa --embedding_path 'weights_visa/adaptive_weight' --mask_prompt 'visa_balanced_masks_me_0' --image_prompt 'datasets/visa' --outname 'visa_me_adaptive_weight'