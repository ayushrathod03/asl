import os
import argparse
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

import torch
print('torch version',torch.__version__)               # PyTorch version
print('torch version cuda',torch.version.cuda) 
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from accelerate import PartialState
from accelerate.utils import set_seed
import time
import torch.nn.functional as F
from PIL import Image
import random
import wandb
import numpy as np
import random
from torch.utils.data import random_split, DataLoader
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance
from data_helpers import VideoData
from diffusers import SanaPipeline, SanaVideoPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from fvd import compute_fvd


from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=2)
parser.add_argument("--name",type=str,default="jlbaker361/model",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--save_dir",type=str,default="weights")
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--load_hf",action="store_true")
parser.add_argument("--rank",type=int,default=4)
parser.add_argument("--num_inference_steps",type=int,default=50)



def main(args):
    accelerator=Accelerator(log_with="wandb",
                            mixed_precision=args.mixed_precision,
                            gradient_accumulation_steps=args.gradient_accumulation_steps,
                            )
    set_seed(123)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    state = PartialState()
    print(f"Rank {state.process_index} initialized successfully")
    if accelerator.is_main_process or state.num_processes==1:
        accelerator.print(f"main process = {state.process_index}")
    if accelerator.is_main_process or state.num_processes==1:
        try:
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)
        except HfHubHTTPError:
            print("hf hub error!")
            time.sleep(random.randint(5,120))
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)

    


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    
    #with accelerator.autocast():
    
    pipeline=SanaVideoPipeline.from_pretrained("Efficient-Large-Model/SANA-Video_2B_480p_diffusers",device=device)
    transformer=pipeline.transformer
    vae=pipeline.vae
    text_encoder=pipeline.text_encoder
    scheduler=pipeline.scheduler
    tokenizer=pipeline.tokenizer

    dataset=VideoData("0.5","jlbaker361/wlasl",tokenizer)
    test_size=int(len(dataset)//10)
    train_size=int(len(dataset)-2*test_size)

    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(42)

    # Split the dataset
    train_dataset, test_dataset,val_dataset = random_split(dataset, [train_size, test_size,test_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size, shuffle=True)

    save_subdir=os.path.join(args.save_dir,args.name)
    os.makedirs(save_subdir,exist_ok=True)

    WEIGHTS_NAME="diffusion_pytorch_model.safetensors"
    CONFIG_NAME="config.json"
    
    save_path=os.path.join(save_subdir,WEIGHTS_NAME)
    config_path=os.path.join(save_subdir,CONFIG_NAME)
    
    
    
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    trans_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    transformer.add_adapter(trans_lora_config)
    
    params= filter(lambda p: p.requires_grad, transformer.parameters())
    
    optimizer=torch.optim.AdamW(params)
    
    train_dataset,optimizer,transformer,vae,text_encoder,scheduler=accelerator.prepare(train_dataset,optimizer,transformer,vae,text_encoder,scheduler)

    start_epoch=1
    try:
        if args.load_hf:
            pretrained_config_path=api.hf_hub_download(args.name,CONFIG_NAME,force_download=True)
            transformer.load_lora_adapter(args.name,weight_name=WEIGHTS_NAME
                                        )
            with open(pretrained_config_path,"r") as f:
                data=json.load(f)
            start_epoch=data["start_epoch"]+1
    except Exception as e:
        accelerator.print(e)
        
    def inference(label:str,dataloader:DataLoader):
        fake=[]
        real=[]
        for b,batch in enumerate(dataloader):
            real.append(batch["video"])
            text=batch["text"]
            
            video = pipeline(
                prompt=text,
                negative_prompt="",
                height=448,
                width=896,
                frames=75,
                guidance_scale=6,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
                output_type="pt"
            )[0]
            if b==0:
                accelerator.print("video size ",video.size())
            fake.append(video)
        fake= torch.cat(fake)
        real=torch.cat(real)
        
        fvd_score = compute_fvd(real, fake, 4, device, batch_size=4)
        
        accelerator.log({f"{label}_fvd":fvd_score})
        
    def save(e:int,state_dict:dict):
        #state_dict=???
        print("state dict len",len(state_dict))
        torch.save(state_dict,save_path)
        with open(config_path,"w+") as config_file:
            data={"start_epoch":e}
            json.dump(data,config_file, indent=4)
            pad = " " * 2048  # ~1KB of padding
            config_file.write(pad)
        print(f"saved {save_path}")
        try:
            transformer.save_lora_adapter(save_subdir,weight_name=WEIGHTS_NAME)
            api.upload_file(path_or_fileobj=save_path,path_in_repo=WEIGHTS_NAME,repo_id=args.name)
            api.upload_file(path_or_fileobj=config_path,path_in_repo=CONFIG_NAME,
                                    repo_id=args.name)
            print(f"uploaded {args.name} to hub")
        except Exception as e:
            accelerator.print("failed to upload")
            accelerator.print(e)

    for e in range(start_epoch,args.epochs+1):
        start=time.time()
        loss_buffer=[]
        for b,batch in enumerate(train_loader):
            if b==args.limit:
                break

            
            with accelerator.accumulate(params):
                video=batch["video"].permute(0, 2, 1, 3, 4).contiguous().to(device)
                text=batch["text"]
                tokenized_text=batch["token"]['input_ids'].to(device)
                encoder_attention_mask=batch["token"]["attention_mask"].to(device).float()
                
                if e==start_epoch and b==0:
                    accelerator.print("video ",video.size(),video.dtype,video.device)
                
                latents=vae.encode(video).latent_dist.sample()
                noise = torch.randn_like(latents)
                
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(tokenized_text, return_dict=False)[0]
                
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                with accelerator.autocast():
                    model_pred = transformer(
                        noisy_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timesteps,
                        return_dict=False,
                    )[0]
                    
                    loss=F.mse_loss(model_pred.float(),noise.float())
                
                loss_buffer.append(loss.cpu().detach().numpy())
                
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        end=time.time()
        accelerator.log({
            f"avg_loss":np.mean(loss_buffer),
            "train_loss":train_loss
        })
        accelerator.print(f"epoch {e} elapsed {end-start}")


if __name__=='__main__':
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")