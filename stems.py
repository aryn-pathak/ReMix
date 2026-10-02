import curses
import math
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # run any op mps lacks on the cpu

import librosa
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

import tui
from audio import SR, ROOM_DIMENSIONS, process_audio

DEFAULT_DISTANCE = 2.0  # meters

# azimuths go anticlockwise from straight ahead (+y is to the left), so 45 is front-left
# and 315 front-right. Spread evenly the long way round, 45 to 315 through the back
STEM_ORDER = ["guitar", "drums", "piano", "other", "bass", "vocals"]
AZIMUTHS = {stem: 45 + 270 * i / (len(STEM_ORDER) - 1) for i, stem in enumerate(STEM_ORDER)}


def ask_path():
    while True:
        # terminals quote or backslash-escape dragged-in paths
        path = input("Song file path (type it or drag the file in): ").strip().strip("'\"").replace("\\ ", " ")
        path = os.path.expanduser(path)
        if os.path.isfile(path):
            return path
        print(f"No file at {path}")


def ask_room():
    try:
        x, y, z = (float(v) for v in input("Room length width height in m, e.g. 9 7 3.2: ").split())
        # sources sit at least 1 m out, so the room has to be 2 m across
        if min(x, y) >= 2 and z >= 1:
            return {"length": x, "width": y, "height": z}
    except ValueError:
        pass
    room = ROOM_DIMENSIONS
    print(f"Invalid (needs at least 2 x 2 x 1), using {room['length']} {room['width']} {room['height']}")
    return room


def ask_distance(room):
    try:
        d = float(input("Distance from you to every instrument in m, e.g. 2: "))
        if d <= 0:
            raise ValueError
    except ValueError:
        d = DEFAULT_DISTANCE
        print(f"Invalid, using {d}")

    half = min(room["length"], room["width"]) / 2
    if d > half:
        d = max(half - 1, 1)
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
    try:
        mode = curses.wrapper(tui.choose, "Positioning:", ["simple", "advanced"])
    except curses.error:  # not a real terminal, e.g. an IDE's run console
        print("No interactive terminal, using simple positioning")
        mode = "simple"
    if mode == "advanced":
        positions = curses.wrapper(tui.place, room, STEM_ORDER, AZIMUTHS)
        curses.wrapper(tui.elevate, room, positions)
    else:
        dist = ask_distance(room)
        positions = {name: (AZIMUTHS[name], dist, 0.0) for name in STEM_ORDER}

    listener = (room["length"] / 2, room["width"] / 2, room["height"] / 2)
    stems, stem_sr = separate(path)

    rendered = []
    for name in STEM_ORDER:
        az, dist, height = positions[name]
        source = (listener[0] + dist * math.cos(math.radians(az)),
                  listener[1] + dist * math.sin(math.radians(az)),
                  listener[2] + height)
        print(f"Spatialising {name} at {az:g} deg, {dist:g} m, {height:+g} m...")
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
