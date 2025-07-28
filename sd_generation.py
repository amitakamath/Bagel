
import os
import pdb
import json
import torch
import random
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline, \
        StableDiffusion3Pipeline


def read_metadata(filename):
    # something like evaluation_metadata.jsonl
    metadata_list = [json.loads(line) for line in open(filename).readlines()]
    return metadata_list


def load_model(model_name):
    if model_name == '2.1':
        model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, \
                subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, \
                scheduler=scheduler, torch_dtype=torch.float16)

    elif model_name == 'xl':
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", \
                torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

    elif model_name == '3':
        pipe = StableDiffusion3Pipeline.from_pretrained(\
                "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)

    elif model_name == '3.5-med':
        pipe = StableDiffusion3Pipeline.from_pretrained(\
                "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)

    elif model_name == '3.5-large':
        pipe = StableDiffusion3Pipeline.from_pretrained(\
                "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
    else:
        raise NotImplementedError
    pipe.to('cuda')
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Generate images with SD")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--metadata_file", type=str, required=True, help="JSON file containing lines of metadata for each prompt.")
    parser.add_argument("--model", type=str, required=True, choices=["2.1", "xl", \
            "3", "3.5-med", "3.5-large"], help="Model name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_path = os.path.join(args.output_dir, "images")
    os.makedirs(images_path, exist_ok=True)

    pipe = load_model(args.model)
    metadata_list = read_metadata(args.metadata_file)

    for idx, metadata in enumerate(tqdm(metadata_list)):
        outpath = os.path.join(images_path, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        
        if args.model in ['2.1', 'xl']:
            image = pipe(metadata['prompt']).images[0]
        elif args.model in ['3']:
            image = pipe(metadata['prompt'], negative_prompt="", \
                    num_inference_steps=28, guidance_scale=7.0).images[0]
        elif args.model in ['3.5-med']:
            image = pipe(metadata['prompt'], num_inference_steps=40, \
                    guidance_scale=4.5).images[0]
        elif args.model in ['3.5-large']:
            image = pipe(metadata['prompt'], num_inference_steps=28, \
                    guidance_scale=3.5).images[0]

        samples_path = os.path.join(outpath, "samples")
        os.makedirs(samples_path, exist_ok=True)
        image.save(os.path.join(samples_path, "00000.png"))
        json.dump(metadata, open(os.path.join(outpath, "metadata.jsonl"), "w"))

    print("Done!")


if __name__ == "__main__":
    main()

