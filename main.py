from cochlea import generate_spectrograms_from_ds
from transformations.room import get_random_room, get_room_impulse, convolve_with_room
import json
from types import SimpleNamespace
import random

print("reading impulses")

def composed_aug(x, sr):
    x = apply_noise(x)
    x = apply_background_sound(x, background_noises)
    h = random.choice(impulses)
    y = np.convolve(x, h)
    y = np.resize(y, len(x))
    return y

def room_aug(x):
    room = get_random_room()
    ir = get_room_impulse(room)

    return convolve_with_room(x, ir)

print("beginning preprocess")
with open("params.json") as file:
    hparams = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    generate_spectrograms_from_ds(
    ds_path=hparams.ds_path, 
    mapping_filename="maestro-v3.0.0.csv", 
    save_path=hparams.save_path, 
    augment_fn=room_aug,
    hparams=hparams)
