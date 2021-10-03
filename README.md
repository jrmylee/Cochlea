# PFC - Data Pipeline for Audio Transformations

Intended as a one stop shop for audio transformations and output to local disk.

PFC currently chunks audio into 2 second clips and transforms them using your transformation of choice, and saves them to a directory of your choosing.

The output script that is intended to be modified with your paths is `main.py`.

# Structure
The code is structured as follows:

`main.py` - Your entry point for using the software.  Change the directories and input functions as needed.
`augmentations` - Audio augmentations prior to transformation.
`tranformations` - Audio transformation to save to disk.  Usually FFT/Mel Spectrogram/STFT/Etc...
`prefrontal.py` - Code responsible for iterating through the dataset, augmenting, transformating, and ultimately saving to disk.

# Parameters
If you want to change any parameters for any of the transformations, you can do so in `params.json`.

# Dataset
This code is intended to work with the MAESTRO dataset, but can be easily extended to other datasets.  Contact me if you want support for another dataset.