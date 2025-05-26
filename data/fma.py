import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ffmpeg
import tempfile
from tqdm import tqdm


def mp3_to_wav(file, bytes=True, start=0, end=None):
    # Read a mp3 file (bytes=False) or mp3 bytes (bytes=True)
    if bytes:    
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file = tmp_file.name
    # cut mp3 with ffmpeg for efficiency
    (
        ffmpeg
        .input('pipe:0')
        .output(tmp_file, ss=start, to=end, c='copy')
        .overwrite_output()
        .run(input=file, quiet=True)
    )
    audio, sr = torchaudio.load(tmp_file)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    audio = audio.mean(dim=0, keepdim=True).float()
    if bytes:
        os.unlink(tmp_file)
    return audio
    
def get_duration(file, bytes=True):
    if bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(file)
            file = tmp.name
    probe = ffmpeg.probe(file)
    duration = float(probe['format']['duration'])
    if bytes:
        os.unlink(file)
    return duration 

class FMADataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_path = getattr(config, "dataset_path", "./cache/free_music_archive")
        self.sr = getattr(config, "sr", 16000)
        self.wav_window_len = config.seq_len // config.codec_sample_rate * self.sr
        files = os.listdir(self.dataset_path)
        self.mp3s = []
        self.wav_len = []

        # First store duration of each mp3
        for file in files:
            data = pd.read_parquet(os.path.join(self.dataset_path, file))
            for i in tqdm(range(len(data))):
                audio = data.iloc[i]["audio"]["bytes"]
                self.mp3s.append(audio)
                self.wav_len.append(get_duration(audio))
        
        self.wav_len = torch.tensor(self.wav_len)
        self.samples_per_wav = (self.wav_len // 30 + 1).to(torch.int32)
        self.samples_per_wav = self.samples_per_wav.cumsum(0)

        print("========= Done loading dataset =========")
        print(f"{len(self)} samples, apprixmately {self.wav_len.sum() / 3600} h")

    def __len__(self):
        return self.samples_per_wav[-1]
    def __getitem__(self, idx):
        # first map to a music file (longer music are sampled more often)
        wav_idx = torch.bucketize(idx, self.samples_per_wav, right=True)
        mp3 = self.mp3s[wav_idx]
        mp3_len = self.wav_len[wav_idx]

        # pad and random crop
        start = torch.randint(0, max(mp3_len - self.wav_window_len, 1), (1,)).item() / self.sr
        end = start + self.wav_window_len / self.sr
        wav = mp3_to_wav(mp3, start=start, end=end)
        wav = wav[:, :self.wav_window_len]
        if self.wav_len[wav_idx] < self.wav_window_len:
            wav = torch.cat([wav, torch.zeros(wav.shape[0], self.wav_window_len - wav.shape[-1])], dim=-1)

        return wav

def get_dataloader(config):
    dataset = FMADataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=config.num_workers)
    return dataloader

class Config:
    seq_len = 1500
    codec_sample_rate = 50
    dataset_path = "../cache/free_music_archive"
    batch_size = 16
    num_workers = 8

if __name__ == "__main__":

    config = Config()
    dataloader = get_dataloader(config)
    for x in tqdm(dataloader):
       pass
