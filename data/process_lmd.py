# %%
import os
import mido
import tempfile
import numpy as np
import json

from mido import MidiFile, MidiTrack, MetaMessage
from tqdm import tqdm
from midi2audio import FluidSynth

BASE_DIR = "/data/haoyun/mutok/drum_test"
LMD_PATH = "/data/haoyun/mutok/lmd_matched"
#LMD_META_PATH = "../../lmd_matched_h5/"
SOUNDFONT_PATH = "/data/haoyun/mutok/GeneralUser-GS-v1.471.sf2"

def get_midi_path(LMD_PATH):
    all_files = []
    for dirpath, dirnames, filenames in tqdm(os.walk(LMD_PATH)):
        #for filename in filenames:
        if filenames:
            all_files.append(os.path.join(dirpath, filenames[0]))
    return all_files


def seperate_drums(input_file):
    try:
        mid = MidiFile(input_file)
    except:
        # Unreadable-file
        return None
    drum_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    main_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    drum_hits = []
    total_time = []
    tempo = 500000
    existing_end = False

    for track in mid.tracks:
        drum_track = MidiTrack()
        main_track = MidiTrack()
        drum_mid.tracks.append(drum_track)
        main_mid.tracks.append(main_track)
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.is_meta:
                drum_track.append(msg.copy())
                main_track.append(msg.copy())
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                if msg.type == 'end_of_track':
                    existing_end = True
                
            
            elif msg.type in ['note_on', 'note_off', 'control_change', 'pitchwheel'] and msg.channel == 9:
                drum_track.append(msg.copy())
                if msg.type in ['note_on', 'note_off']:
                    main_track.append(msg.copy(velocity=0))
                else:
                    main_track.append(msg.copy())
                if msg.type == 'note_on' and msg.velocity > 20:
                    drum_hits.append(abs_time)
            else:
                if msg.type in ['note_on', 'note_off']:
                    drum_track.append(msg.copy(velocity=0))
                else:
                    drum_track.append(msg.copy())
                main_track.append(msg.copy())
        total_time.append(abs_time)
    total_time = max(total_time)

    def track_length(track):
        return sum(msg.time for msg in track if not msg.is_meta or msg.type != 'end_of_track')
    def calc_length(midi):
        return max(track_length(track) for track in midi.tracks)
    
    
    if not existing_end:
        drum_length = calc_length(drum_mid)
        main_length = calc_length(main_mid)
        max_length = min(drum_length, main_length)

        drum_mid.tracks[0].append(MetaMessage('end_of_track', time=max_length-track_length(drum_mid.tracks[0])))
        main_mid.tracks[0].append(MetaMessage('end_of_track', time=max_length-track_length(main_mid.tracks[0])))

    if len(drum_hits) < 50:
        return None

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
    
    return drum_mid, main_mid, valid_region

def produce_dataset(base_dir, start_num=0, filename_offset=0, max_num=10000):
    midi_files = get_midi_path(LMD_PATH)
    midi_files = midi_files[start_num:]
    if start_num == 0 and os.path.exists(os.path.join(base_dir, "metadata.json")):
        os.remove(os.path.join(base_dir, "metadata.json"))


    index = 0
    for i, file in enumerate(tqdm(midi_files)):
        out = seperate_drums(file)
        if out is None:
            continue

        drum_mid, main_mid, valid_region = out
        file_name = f"{index + filename_offset:05d}.wav"
        # output tmp midi file
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_drum_file = tmp.name
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_main_file = tmp.name

        drum_mid.save(tmp_drum_file)
        main_mid.save(tmp_main_file)

        drum_wav_path = os.path.join(base_dir, "drum", file_name)
        main_wav_path = os.path.join(base_dir, "main", file_name)
        FluidSynth(SOUNDFONT_PATH, 16000).midi_to_audio(tmp_drum_file, drum_wav_path)
        FluidSynth(SOUNDFONT_PATH, 16000).midi_to_audio(tmp_main_file, main_wav_path)
        os.unlink(tmp_drum_file)
        os.unlink(tmp_main_file)

        # update meta
        meta = {"file":file_name, "id":i + start_num, "valid_t":valid_region}
        with open(os.path.join(base_dir, "metadata.json"), mode="a+") as f:
            f.write(json.dumps(meta) + "\n")

        index += 1
        if index == max_num:
            break


if __name__ == "__main__":
   produce_dataset(BASE_DIR, start_num=4605, filename_offset=4000, max_num=200)
