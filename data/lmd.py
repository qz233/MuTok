# %%
import os
import mido
from tqdm import tqdm

LMD_PATH = "../../lmd_matched/"
LMD_META_PATH = "../../lmd_matched_h5/"
# %%

all_files = []
all_files_meta = []
for dirpath, dirnames, filenames in tqdm(os.walk(LMD_PATH)):
    for filename in filenames:
        all_files.append(os.path.join(dirpath, filename))
        
for dirpath, dirnames, filenames in tqdm(os.walk(LMD_META_PATH)):
    for filename in filenames:
        all_files_meta.append(os.path.join(dirpath, filename))
# %%

# %%
import mido

mid = mido.MidiFile(all_files[300])



# %%
import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage

def extract_drums(input_file, output_file):
    mid = MidiFile(input_file)
    out_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    drum_hits = []
    tempo = 500000
    for i, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        out_mid.tracks.append(new_track)
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.is_meta:
                new_track.append(msg)
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
            
            elif msg.type in ['note_on', 'note_off', 'control_change', 'pitchwheel'] and msg.channel == 9:
                new_track.append(msg)
                if msg.type == 'note_on' and msg.velocity > 20:
                    drum_hits.append(abs_time)

    out_mid.save(output_file)
    drum_hits = np.array(sorted(list(set(drum_hits))))

    drum_hits = mido.tick2second(drum_hits, mid.ticks_per_beat, tempo)
    interval = drum_hits[1:] - drum_hits[:-1]
    
    invalid_region = np.argwhere(interval > 2)
    if len(invalid_region) > 1:
        invalid_region = invalid_region[0].tolist()
        valid_region = ([(drum_hits[0], drum_hits[invalid_region[0]])] +
                        [(drum_hits[invalid_region[i]+1], drum_hits[invalid_region[i+1]]) for i in range(len(invalid_region)-1)] +
                        [(drum_hits[invalid_region[-1]+1], drum_hits[-1])])
    else:
        valid_region = [(drum_hits[0], drum_hits[-1])]
    print(valid_region)
    return drum_hits


drum_hits = extract_drums(all_files[300], "demo.mid")
# %%
mid.save("example.mid")
# %%
import h5py

file = h5py.File(all_files_meta[300], 'r')
# %%
file['musicbrainz']['artist_mbtags'][:]

# %%

import fluidsynth

wav_file = "temp.wav"
midi_file = all_files[300]
fs = fluidsynth.Synth()
fs.start(driver="file", filename=wav_file)

sfid = fs.sfload("musescore.sf2")
fs.program_select(0, sfid, 0, 0)
fs.midi_file_play(midi_file)
fs.delete()
# %%
import tinysoundfont
import time

synth = tinysoundfont.Synth()
synth.start()
sfid = synth.sfload("florestan-piano.sf2")
synth.program_select(0, sfid, 0, 0)