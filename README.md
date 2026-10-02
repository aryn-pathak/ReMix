# ReMix (1.0)

> An audio spatializer that separates a song into stems and places each one at a distinct 3D position around the listener, creating a wider, more immersive headphone experience.

## Requirements

- Python 3.12+ (tested, might work on older versions)
- [SADIE II dataset](https://zenodo.org/records/12092466/files/D1_HRIR_SOFA.zip?download=1), subject D1 (Neumann KU100), 48 kHz. Not included in this repo, see Setup.
- Headphones, since the output is binaural and doesn't work properly on speakers

## Setup

Create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**Optional (Linux):** the default `torch` from PyPI is the CUDA build and pulls in several GB of NVIDIA packages (vs ~200 MB for CPU only). If you don't have an NVIDIA GPU, install the smaller CPU-only build first:

```sh
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then install the dependencies:

```sh
pip install -r requirements.txt
```

`matplotlib` is only needed for the reference scripts (`visualisation.py`, `analysis.py`).

Finally, download the SADIE II zip linked above, extract the 48 kHz KU100 HRIR file, and save it in the project root as `SADIEII_KU100.sofa`.

## Run

With the virtual environment activated, run:

```sh
python stems.py
```

The first run downloads the Demucs `htdemucs_6s` model weights. The output is written next to the input song as `<song>_spatial.wav`.

## Files

- `stems.py`: The main file to run. Separates the song into stems, processes them with `audio.py`, and saves the result.
- `audio.py`: The functions that apply the cues to the audio and process it.
- `tui.py`: The text UI for advanced positioning in `stems.py`.
- `compute_absorption.py`: A one-time-run script that calculates a constant set of dB/m attenuation values for different frequencies, used for air absorption. Uses the ISO 9613-1 standard.
- `visualisation.py`: A set of functions I used to try understanding the HRIRs.
- `analysis.py`: An experimental (incomplete) attempt at compressing HRIRs. Not used by the main pipeline, so it's safe to ignore.

## How it works

Each of the six stems is rendered binaurally on its own and placed at a different position around the listener. `audio.py` applies 6 psychoacoustic cues to create a sense of direction, space, and distance.

Cues used:

- *HRIR, for direction* (a collection of interaural time difference, interaural level difference, and pinna filtering): convolves the signal with an impulse response recorded by placing a speaker at a particular azimuth and elevation around a mannequin.
- *Level falloff, for distance*: the loss of energy (loudness) of sound over distance (inverse square law).
- *Air absorption, for distance*: frequency-dependent attenuation, meant to replicate how air absorbs sound. Uses an equalizer to apply the attenuation calculated by `compute_absorption.py`, multiplied by distance.
- *Direct-to-reverberant ratio (DRR), for distance*: the balance of direct sound to reverb indicates distance relative to the room.
- *Early reflections, for space*: the 5-35 ms reflections that imply room size and spaciousness (relying on the precedence/Haas effect so they fuse with the direct sound). These also have HRIRs applied to replicate the direction each reflection comes from, which the brain uses without you noticing.
- *Late reverb, for space*: the diffuse tail that conveys the room's size and absorption.

For each stem, the direct sound (with falloff, absorption and HRIR), early reflections, and late reverb are created separately and added together, left and right channels separately. The result is scaled to prevent clipping.

A more detailed explanation is in the [devlog](https://stardance.hackclub.com/projects/62390/devlogs/63226).

## License

<!-- Add your license here, e.g. MIT, and make sure a LICENSE file is in the repo -->

## Acknowledgements

- **SADIE II**: SADIE Database Copyright 2018 The University of York.
  This product includes data developed at The Audio Lab, University of York, UK
  (Cal Armstrong, Lewis Thresh, Gavin Kearney) as part of the SADIE Project.
  [Database link](https://www.york.ac.uk/sadie-project/database.html)

- **HT Demucs**: Stem separation uses Hybrid Transformer Demucs by Meta AI
  ([repository](https://github.com/facebookresearch/demucs)).