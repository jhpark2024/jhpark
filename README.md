# Transmit What You Need: A Task-Adaptive Semantic Communications for Visual Information
[[ArXiv Preprint]](https://arxiv.org/abs/2412.13646)
## Installation
```
pip install -r requirements.txt
```

## Usage
### Visaul Semantics Extraction
Download the [model](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) which is pretrained on the Visual Genome dataset. Then, put it under `pretrained` directory.

With the comand
```
python inference.py --img_path $IMAGE_PATH --result_path $RESULT_PATH --resume $MODEL_PATH
```
extracted visual semantics(objects, layouts, SG, and filtered SG) are saved in `result_path` as a json file.

without arguments, the comand
```
python inference.py
```
utilizes image `example/2407349.jpg` and the results are saved as `example_result/2407349.json`.

### Visualization
To visulaize extracted semantics:
```
python visualize.py --img_path $IMAGE_PATH --result_path $RESULT_PATH
```

We attached visualized semantics on `example/2407349.jpg`.
For clarity, `Filtered Scene Graph` only depicts informative relations. 

<p align="center">
  <img src="example_result/visualization_2407349.png">
</p>

### Generation
We utilize `diffusers` library to generate image through latent diffusion model.

Running the below comand automatically loads the pretrained model described in [here](https://github.com/TonyLianLong/LLM-groundedDiffusion.git).
```
python generate.py --result_path $RESULT_PATH --save_path $SAVE_PATH
```
The number of denoising time steps and generated images can be adjusted with `--num_inference_steps` and `--num_images_per_prompt` respectively.
the generated image is saved in `--save_path` directory.
