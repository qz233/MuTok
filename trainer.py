import os
import torch
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
        if not self.trainer.config.resume_from_checkpoint and self.trainer.config.type == "train":
            os.system(f"rm -r {log_dir}/*")
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
            wandb.log({"step": steps, "train_loss": loss})
        self._prev_t = time()
    
    def update_eval(self, eval_dict:dict):
        print("======= eval result =======")
        for metric, value in eval_dict.items():
            print(f"[{metric}]:{value}")
            #self.tensorboard_writer.add_scalar(f"eval/{metric}", value, self.trainer.global_step)
            wandb.log({"step": self.trainer.global_step, f"eval/{metric}": value})
        print("---------------------------")



class Trainer():
    def __init__(self, model, train_dataloader, valid_dataloader, config) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.config = config
        # Training config
        self.launch_name = config.launch_name
        self.epoches = config.epoches
        self.log_dir = config.log_dir
        self.model_cache = config.model_cache
        self.log_frequency = config.log_frequency
        self.predict_frequency = config.predict_frequency
        self.checkpoint_frequency = config.checkpoint_frequency
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
        if train_dataloader:
            self._train_preparation()
        #print(f"{self.device=}, {is_main_process()=}")

    
    def _train_preparation(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.config.learning_rate, weight_decay=1e-4)
        self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(self.optimizer, self.config.warmup_steps, len(self.train_dataloader) * self.epoches, min_lr_rate=0.1)

        self.logger = TrainingLogger(self)
        if self.config.resume_from_checkpoint:
            self.load_training_state(self.config.resume_from_checkpoint)

        # prepare with accelerate
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        self.valid_dataloader = self.accelerator.prepare(self.valid_dataloader)
        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)
        
        # Set major eval metric
        #self.main_metric = getattr(self.config, "main_metric", "mse")


    def train(self):
        while self.current_epoch <= self.epoches:
            for steps, inputs in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.global_step += 1
                    loss = self.model.training_step(inputs)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.logger.update_loss(loss, self.global_step)
                    self.optimizer.zero_grad()
                
                if self.global_step % self.predict_frequency == 0:
                    results = self.evaluate()
                    self.logger.update_eval(results)

                    # checkpoint
                    self.save_training_state(self.model_cache + "/" + self.launch_name + "/latest.pt")
                    if self.current_epoch % self.checkpoint_frequency == 0:
                        self.save_training_state(self.model_cache + "/" + self.launch_name + f"/{self.current_epoch}step.pt")
            self.current_epoch += 1
        
        try:
            os.rmdir(self.model_cache + "/" + self.launch_name + "/latest.pt")
        except:
            pass
    def evaluate(self):
        metrics = get_metrics(self.config)
        for inputs in tqdm(self.valid_dataloader):
            from time import time
            outputs = self.model.test(inputs)
            metrics.update(outputs["target"], outputs["pred"])
        # aggregate all results
        return metrics.accumulate_result()


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
        state_dict = torch.load(path, map_location="cpu")
        self.current_epoch = state_dict.get("epoch", 0)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.global_step = state_dict.get("steps", 0)
        if is_main_process():
            print(f"load from {path} ({self.current_epoch} epoch, {self.logger._global_step} steps)")
