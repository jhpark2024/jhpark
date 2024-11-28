import argparse
import torch
import json
from diffusers import DiffusionPipeline

def get_arg_parser():
    parser = argparse.ArgumentParser('Image Generation', add_help=False)
    parser.add_argument('--result_path', type=str, default='example_result/2407349.json',
                        help="Path of the Extracted Visual Semantics")
    parser.add_argument('--model_name', type=str, default='longlian/lmd_plus')
    parser.add_argument('--custom_pipeline', type=str, default='llm_grounded_diffusion')
    parser.add_argument('--gligen_scheduled_sampling_beta', type=str, default=0.4)
    parser.add_argument('--num_inference_steps', type=str, default=50)
    parser.add_argument('--num_images_per_prompt', type=str, default=1)
    parser.add_argument('--save_path', type=str, default='generated_image/2407349.jpg')
    return parser

def generate(args, Visual_Semantics, pipe):
    Filtered_SceneGraph = Visual_Semantics['Filtered_SceneGraph']
    prompt = ", ".join(Visual_Semantics['Filtered_SceneGraph'])
    phrases = []
    layout = []

    for sub_graph in Filtered_SceneGraph:
        parts = sub_graph.split(' ')
        if len(parts) > 2:
            phrases.append(parts[0])
            layout.append(Visual_Semantics['Objects_Layouts'][str(''.join([i for i in parts[0] if not i.isdigit()]))]
                          [int(''.join([i for i in parts[0] if i.isdigit()]))])
            phrases.append(parts[-1])
            layout.append(Visual_Semantics['Objects_Layouts'][str(''.join([i for i in parts[-1] if not i.isdigit()]))]
                          [int(''.join([i for i in parts[-1] if i.isdigit()]))])
        else:
            phrases.append(parts[0])
            layout.append(Visual_Semantics['Objects_Layouts'][str(''.join([i for i in parts[0] if not i.isdigit()]))]
                          [int(''.join([i for i in parts[0] if i.isdigit()]))])

    generated_images = pipe(
        prompt=prompt,
        phrases=phrases,
        boxes=layout,
        gligen_scheduled_sampling_beta=args.gligen_scheduled_sampling_beta,
        output_type="pil",
        num_inference_steps=args.num_inference_steps,
        lmd_guidance_kwargs={},
        num_images_per_prompt=args.num_images_per_prompt
        ).images

    return generated_images

def main(args):
    pipe_l = DiffusionPipeline.from_pretrained(
        args.model_name,
        custom_pipeline=args.custom_pipeline,
        custom_revision='main',
        variant="fp16",
        torch_dtype=torch.float16
    )
    pipe_l = pipe_l.to("cuda")

    with open(f'{args.result_path}', 'r') as file:
        Visual_Semantics = json.load(file)

    generated_images = generate(args, Visual_Semantics, pipe_l)
    generated_images[0].save(f"{args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Generation', parents=[get_arg_parser()])
    args = parser.parse_args()
    main(args)