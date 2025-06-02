# %%
import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tempfile
import json
from tqdm import tqdm
import wave
import soundfile as sf


def read_wav_segment(file_path, start_frame, duration):
    with sf.SoundFile(file_path) as f:
        f.seek(start_frame)
        segment = f.read(duration, dtype='float32')
    segment = torch.from_numpy(segment.T).float()
    segment = segment.mean(dim=0, keepdim=True).float()
    return segment

def get_wav_duration(file_path, sr=16000):
    with wave.open(file_path, 'rb') as wav_file:
        duration = wav_file.getnframes() / sr  
    return duration

def fix_shape(wav, length):
    wav = wav[:, :length]
    if wav.shape[-1] < length:
        wav = torch.cat([wav, torch.zeros(wav.shape[0], length - wav.shape[-1])], dim=-1)
    return wav

class DrumDataset(Dataset):
    def __init__(self, config, type):
        self.config = config
        self.dataset_path = getattr(config, f"{type}_dataset_path", "./cache/free_music_archive")
        self.sr = getattr(config, "sample_rate", 16000)
        self.wav_window_len = config.seq_len // config.codec_sample_rate
        files = os.listdir(self.dataset_path)
        self.mp3s = []
        self.wav_len = []

        # First load metadata
        with open(os.path.join(self.dataset_path, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]
        
        for entry in self.metadata:
            entry["valid_t"] = torch.FloatTensor(entry["valid_t"])
            # correct wrong interval
            drum_f = os.path.join(self.dataset_path, "drum", entry["file"])
            main_f = os.path.join(self.dataset_path, "main", entry["file"])
            max_time = min(get_wav_duration(drum_f), get_wav_duration(main_f))
            entry["valid_t"][-1, 1] = max_time
            self.wav_len.append((entry["valid_t"][:, 1]-entry["valid_t"][:, 0]).sum().item())
        self.wav_len = torch.FloatTensor(self.wav_len)
        self.samples_per_wav = (self.wav_len // self.wav_window_len).to(torch.int32)
        self.samples_per_wav = self.samples_per_wav.cumsum(0)

        print("========= Done loading dataset =========")
        print(f"{len(self)} samples, apprixmately {self.wav_len.sum() / 3600} h")

    def __len__(self):
        return self.samples_per_wav[-1]
    
    def __getitem__(self, idx):
        # first map to a music file (longer music are sampled more often)
        wav_idx = torch.bucketize(idx, self.samples_per_wav, right=True)
        meta = self.metadata[wav_idx]

        # reject sample a valid window
        invalid = True
        while invalid:
            start = np.random.rand() * meta["valid_t"][-1, 1]
            for drum_start, drum_end in meta["valid_t"]:
                if drum_start < start and start + self.wav_window_len < drum_end:
                    invalid = False

        drum = read_wav_segment(os.path.join(self.dataset_path, "drum", meta["file"]), 
                                start_frame=start * self.sr,
                                duration=self.wav_window_len * self.sr)
        main = read_wav_segment(os.path.join(self.dataset_path, "main", meta["file"]), 
                        start_frame=start * self.sr,
                        duration=self.wav_window_len * self.sr)
        drum = fix_shape(drum, self.wav_window_len * self.sr)
        main = fix_shape(main, self.wav_window_len * self.sr)
        return drum, main

def get_drum_dataloader(config, type):
    dataset = DrumDataset(config, type)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=config.num_workers)
    return dataloader

class Config:
    seq_len = 1000
    codec_sample_rate = 50
    train_dataset_path = "/data/haoyun/drum_ds"
    batch_size = 16
    num_workers = 0
# %%
if __name__ == "__main__":
    config = Config()
    dataloader = get_drum_dataloader(config, "train")
    for x in tqdm(dataloader):
       pass
