import os
import torch
import torch.nn as nn
import wandb
from time import time
from tqdm import tqdm
from accelerate import Accelerator
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from omegaconf import OmegaConf



class TrainingLogger():
    # a simple logger
    # only do action on main process
    def __init__(self, trainer):
        self.trainer = trainer
        log_dir = trainer.log_dir + "/" + trainer.launch_name
        wandb.init(
            project = "mutok",
            name = trainer.launch_name,
            config = OmegaConf.to_container(self.trainer.config),
        )
        self.log_freq = trainer.log_frequency
        self._prev_t = time()
        print("======= start training =======")
    
    def update_loss(self, loss, steps):
        loss = loss.detach().item()
        print(f"\r[epoch]:{self.trainer.current_epoch}/{self.trainer.epoches}  ", end="")
        print(f"[step]:{steps}/{len(self.trainer.train_dataloader)}  ", end="")
        print(f"[loss]:{loss:.3e}  ", end="")
        print(f"[time]:{time()-self._prev_t:.3f}s  ", end="")
        print(f"[lr]:{self.trainer.optimizer.param_groups[0]['lr']:.3e}  ", end="")
        if steps % self.log_freq == 0:
            print(f"")  # Start a new row
            #self.tensorboard_writer.add_scalar("Loss", loss, steps)
            wandb.log({"train_loss": loss}, step=self.trainer.global_step)
        self._prev_t = time()
    
    def update_gen_sample(self, log_dict):
        print("======= new sample uploaded =======")
        for key in log_dict:
            print(f"[description]:{key}")
        log_dict = {k: wandb.Audio(v,sample_rate=self.trainer.config.sample_rate) for k,v in log_dict.items()}
        wandb.log(log_dict , step=self.trainer.global_step)
        print("---------------------------")



class Trainer():
    def __init__(self, model, codec, train_dataloader, valid_dataloader, config) -> None:
        self.model = model
        self.codec = codec
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.config = config
        # Training config
        self.launch_name = config.launch_name
        self.epoches = config.epoches
        self.log_dir = config.log_dir
        self.model_cache = config.model_cache
        self.log_frequency = config.log_frequency
        self.checkpoint_frequency = config.checkpoint_frequency
        self.warmup_step = getattr(config, "warmup_step", 200)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.current_epoch = 0
        self.global_step = 0
        # use accelerater
        self.accelerator = Accelerator(
            device_placement = False, 
            mixed_precision = "no",
            gradient_accumulation_steps = self.gradient_accumulation_steps,
        )
        self.device = self.accelerator.device
        self.model = self.model.to(self.device)
        self.codec = self.codec.to(self.device)
        if train_dataloader:
            self._train_preparation()
        #print(f"{self.device=}, {is_main_process()=}")
        print(f"Model num param: {sum(x.nelement() for x in model.parameters()) // (2 ** 20)} M")
        self.upload_ground_truth_sample()

    
    def _train_preparation(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.config.learning_rate, weight_decay=1e-4)
        self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(self.optimizer, self.warmup_step, len(self.train_dataloader) * (self.epoches - self.current_epoch) // self.gradient_accumulation_steps, min_lr_rate=0.1)

        self.logger = TrainingLogger(self)
        if self.config.resume_from_checkpoint:
            self.load_training_state(self.config.resume_from_checkpoint)

        # prepare with accelerate
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        self.valid_dataloader = self.accelerator.prepare(self.valid_dataloader)
        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)
        


    def train(self):
        while self.current_epoch <= self.epoches:
            self.model.train()
            for step, (drum, main) in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.global_step += 1

                    drum, main = drum.to(self.device), main.to(self.device)
                    drum, main = self.codec.prepare_input(drum, main)
                    loss = self.model(drum, main)    
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)         
                    # training routine
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.logger.update_loss(loss, step)
                    self.optimizer.zero_grad()
               

            # checkpoint
            self.save_training_state(self.model_cache + "/" + self.launch_name + "/latest.pt")
            if self.current_epoch % self.checkpoint_frequency == 0:
                self.save_training_state(self.model_cache + "/" + self.launch_name + f"/{self.current_epoch}step.pt")
            self.update_gen_sample()
            self.current_epoch += 1
    

    def save_training_state(self, path):
        state_dict = {
            "config":self.config,
            "epoch":self.current_epoch,
            "steps":self.global_step,
            "model": self.model.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        torch.save(state_dict, path)

    def load_training_state(self, path):
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict["model"])
        self.current_epoch = state_dict.get("epoch", 0)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        #self.scheduler.load_state_dict(state_dict["scheduler"])
        self.global_step = state_dict.get("steps", 0)
        print(f"load from {path} ({self.current_epoch} epoch, {self.global_step} steps)")

    def upload_ground_truth_sample(self):
        for drum_vis, main_vis in self.valid_dataloader:
            break
        self.drum_vis, self.main_vis = drum_vis, main_vis
        log_dict = {
            "ground_truth drum": self.drum_vis[0, 0],
            "ground_truth main": self.main_vis[0, 0],
            "ground_truth together": (self.main_vis[0, 0] + self.drum_vis[0, 0]) * 0.701
        }
        self.logger.update_gen_sample(log_dict)
    
    def update_gen_sample(self):
        self.model.eval()
        with torch.no_grad():
            main = self.main_vis.to(self.device)
            main = self.codec.encode_continuous(main)
            gen_drum_latent = self.model.inference(main)   
            gen_drum = self.codec.decode_continuous(gen_drum_latent)
            gen_wav = gen_drum.cpu()

        log_dict = {
            f"Generated drum at epoch {self.current_epoch}": gen_wav[0, 0],
            f"Generated full at epoch {self.current_epoch}": (self.main_vis[0, 0] + gen_wav[0, 0]) * 0.701
        }
        self.logger.update_gen_sample(log_dict)
