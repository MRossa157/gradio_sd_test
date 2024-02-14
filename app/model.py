from abc import ABC

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline

class AbstractModel(ABC):
    def __init__(self) -> None:
        pass       
    
    def get_picture(self):
        raise NotImplementedError()


class SD_2_1(AbstractModel):
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise AssertionError('Please install CUDA')  
        
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1"
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, 
            torch_dtype=torch.float16
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.pipe = self.pipe.to("cuda")
        
    def get_picture(self, prompt, negative_prompt=None):
        image = self.pipe(prompt, negative_prompt = negative_prompt).images[0]
        return image
    
    
class SD_XL(AbstractModel):
    def __init__(self) -> None:        
        if not torch.cuda.is_available():
            raise AssertionError('Please install CUDA')
        
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        
        self.pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        
        # FIXME If you start it not on OS Windows, then uncomment it
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe = self.pipe.to("cuda")
            
        
    def get_picture(self, prompt, negative_prompt=None, num_inference_steps = 40, denoising_end = 0.8):
        image = self.pipe(
                            prompt = prompt, 
                            negative_prompt = negative_prompt, 
                            num_inference_steps = num_inference_steps,
                            denoising_end = denoising_end,
                                                        
                        ).images[0]
        return image