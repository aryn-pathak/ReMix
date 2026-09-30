import math
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # run any op mps lacks on the cpu

import librosa
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from audio import SR, ROOM_DIMENSIONS, process_audio

DEFAULT_DISTANCE = 2.0  # meters

# azimuths go anticlockwise from straight ahead (+y is to the left), so 45 is front-left
# and 315 front-right. Left to right across the front, 18 deg apart, none in 45..315
STEM_ORDER = ["guitar", "piano", "other", "bass", "drums", "vocals"]
AZIMUTHS = {stem: (45 - 18 * i) % 360 for i, stem in enumerate(STEM_ORDER)}


def ask_path():
    while True:
        # terminals quote or backslash-escape dragged-in paths
        path = input("Song file path: ").strip().strip("'\"").replace("\\ ", " ")
        path = os.path.expanduser(path)
        if os.path.isfile(path):
            return path
        print(f"No file at {path}")


def ask_room():
    try:
        x, y, z = (float(v) for v in input("Room dimensions x y z (m): ").split())
        if min(x, y, z) > 0:
            return {"length": x, "width": y, "height": z}
    except ValueError:
        pass
    room = ROOM_DIMENSIONS
    print(f"Invalid, using {room['length']} {room['width']} {room['height']}")
    return room


def ask_distance(room):
    try:
        d = float(input("Source distance (m): "))
        if d <= 0:
            raise ValueError
    except ValueError:
        d = DEFAULT_DISTANCE
        print(f"Invalid, using {d}")

    half = min(room["length"], room["width"]) / 2
    if d > half:
        d = half - 1
        if d <= 0:  # room under 2 m across, half - 1 would put the source behind the listener
            d = half / 2
        print(f"Too far for this room, using {d:g}")
    return d


def separate(path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = get_model("htdemucs_6s")
    model.eval()

    wave, _ = librosa.load(path, sr=model.samplerate, mono=False)
    if wave.ndim == 1:
        wave = np.stack([wave, wave])
    mix = torch.from_numpy(wave)

    # demucs expects a standardised input, as its own cli does
    ref = mix.mean(0)
    mix = (mix - ref.mean()) / ref.std()
    print(f"Separating on {device}...")
    with torch.no_grad():
        stems = apply_model(model, mix[None], device=device, progress=True)[0]
    stems = stems * ref.std() + ref.mean()

    # {name: (samples, 2)}
    return {name: stems[i].cpu().numpy().T for i, name in enumerate(model.sources)}, model.samplerate


def main():
    path = ask_path()
    room = ask_room()
    dist = ask_distance(room)

    listener = (room["length"] / 2, room["width"] / 2, room["height"] / 2)
    stems, stem_sr = separate(path)

    rendered = []
    for name in STEM_ORDER:
        az = math.radians(AZIMUTHS[name])
        source = (listener[0] + dist * math.cos(az), listener[1] + dist * math.sin(az), listener[2])
        print(f"Spatialising {name} at {AZIMUTHS[name]} deg...")
        rendered.append(process_audio(stems[name], stem_sr, room, listener, source, normalize=False))

    n = max(len(r) for r in rendered)
    mix = np.sum([np.pad(r, ((0, n - len(r)), (0, 0))) for r in rendered], axis=0)
    peak = np.max(np.abs(mix))
    if peak > 1:
        mix /= peak

    out_path = os.path.splitext(path)[0] + "_spatial.wav"
    sf.write(out_path, mix, SR)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
