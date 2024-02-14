# Gradio + Stable Diffusion application

## Assignment
Implement a user-friendly GUI or WebUI interface to run image generation for the StableDiffusion neural network.
The form should contain the following elements: 
- text field for entering prompt & negative_prompt
- display of generation result
- button to start generation

## Requirements
- Docker

## Run application
0. Clone the repository and navigate to the application folder:
   ```git clone https://github.com/MRossa157/gradio_sd_test```
2. Build a docker image:
   ```docker build -t gradio_stable_diffusion_2.1 -f docker/app/dockerfile .```
3. Run the assembled docker container:
   ```docker run --gpus all gradio_stable_diffusion_2.1```
5. Navigate to the ___ and enjoy to use!
