from abc import ABC

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class AbstractModel(ABC):
    def __init__(self, pretrained_model_name_or_path = None) -> None:
        pass       
    
    def get_picture(self):
        raise NotImplementedError()


class Model(AbstractModel):
    def __init__(self, pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1") -> None:
       
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, 
            torch_dtype=torch.float16
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        else:
            raise AssertionError('Please install CUDA')
        
    def get_picture(self, prompt, negative_prompt=""):
        image = self.pipe(prompt, negative_prompt = negative_prompt).images[0]
        return image