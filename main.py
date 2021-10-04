from cochlea import generate_spectrograms_from_ds
from augmentations.naive import random_augmentations
from transformations.stft import stft_stacked
import json
from types import SimpleNamespace
with open("params.json") as file:
    hparams = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    generate_spectrograms_from_ds(
    ds_path="/home/jerms/data/maestro-v3.0.0", 
    mapping_filename="maestro-v3.0.0.csv", 
    save_path="/home/jerms/disk1/stft_original", 
    transform_fn=stft_stacked,
    augment_fn=None,
    hparams=hparams)
