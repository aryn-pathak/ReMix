import multiprocessing
import os
import sys

# let ops mps doesn't support fall back to the cpu; must be set before torch is imported
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import demucs.api
import numpy as np
import questionary
import soundfile as sf
import torch
from questionary import Choice

from audio import SR, process_audio

ROOMS = {
    "Studio": {
        "ceiling": 0.70,   # acoustic clouds / absorbent panels
        "floor":   0.10,   # hardwood, often left fairly reflective
        "walls":   0.50,   # mix of foam/fibreglass panels, bass traps, diffusers
    },
    "Church": {
        "ceiling": 0.05,   # plaster or vaulted stone
        "floor":   0.02,   # stone/marble tiles (pews and people would raise this a bit)
        "walls":   0.03,   # bare stone or brick
    },
    "Outdoors": {
        "ceiling": 1.00,   # open sky, nothing reflects back
        "floor":   0.20,   # grass/soil (use ~0.02 for asphalt or concrete)
        "walls":   1.00,   # no walls
    },
}

ROOM_SIZES = {
    "Small":  (4, 3, 2.5),
    "Medium": (8, 6, 3),
    "Large":  (25, 15, 10),
}

# htdemucs_6s stems, spread evenly around the listener
INSTRUMENT_ORDER = ["drums", "vocals", "bass", "other", "guitar", "piano"]

DEFAULT_DIST = 3.0


def ask(question):
    # questionary returns None on ctrl-c, bail instead of carrying None into the maths
    answer = question.ask()
    if answer is None:
        sys.exit(1)
    return answer


def ask_path():
    while True:
        # terminals quote or backslash-escape paths dragged in
        path = input("Path to song (type it or drag the file in): ").strip().strip("'\"").replace("\\ ", " ")
        path = os.path.expanduser(path)
        if os.path.isfile(path):
            return path
        print(f"  no file at {path}")


def ask_float(prompt, default=None, lo=None, hi=None):
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        try:
            v = float(s)
        except ValueError:
            print("  not a number, try again")
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            print(f"  must be between {lo} and {hi}")
            continue
        return v


def ask_absorption():
    A = ask(questionary.select(
        "Choose a room type:",
        choices=[Choice(name, value=coeffs) for name, coeffs in ROOMS.items()]
                + [Choice("Custom", value="custom")],
    ))
    if A == "custom":
        # lower bound > 0 so the total absorption (Sabine denominator) can't be zero
        A = {surface: ask_float(f"Absorption coefficient for {surface} (0.01–1): ", lo=0.01, hi=1)
             for surface in ("ceiling", "floor", "walls")}
    return A


def ask_room():
    size = ask(questionary.select(
        "Choose a room size:",
        choices=[Choice(name, value=dims) for name, dims in ROOM_SIZES.items()]
                + [Choice("Custom", value="custom")],
    ))
    while size == "custom":
        s = input("Enter room size (length, width, height) in meters: ")
        try:
            dims = tuple(float(x) for x in s.replace(",", " ").split())
        except ValueError:
            dims = ()
        if len(dims) == 3 and all(d > 0 for d in dims):
            size = dims
        else:
            print("  need three positive numbers, e.g. 8 6 3")
    length, width, height = size
    return {"length": length, "width": width, "height": height}


def ask_distance(room):
    # sources sit on a circle around the listener (room centre), so the circle has to fit
    # inside the room or the image sources land on/inside the listener
    max_dist = round(min(room["length"], room["width"]) / 2 * 0.95, 2)
    default = min(DEFAULT_DIST, max_dist)
    return ask_float(f"Instrument distance from listener in m (0.1–{max_dist:g}, default {default:g}): ",
                     default=default, lo=0.1, hi=max_dist)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def separate_and_spatialise(separator, songfile, A, room, dist):
    _, separated = separator.separate_audio_file(songfile)
    listener = (room["length"] / 2, room["width"] / 2, room["height"] / 2)

    out = None
    for i, instrument in enumerate(INSTRUMENT_ORDER):
        print(f"spatialising {instrument}...")
        az = 360 * i / len(INSTRUMENT_ORDER)
        stem = separated[instrument].cpu().numpy().T
        stereo = process_audio(A, room, stem, az, dist, listener,
                               sampling_rate=separator.samplerate, normalize=False)
        if out is None:
            out = stereo
        else:
            n = max(len(out), len(stereo))
            out = np.pad(out, ((0, n - len(out)), (0, 0))) + np.pad(stereo, ((0, n - len(stereo)), (0, 0)))

    peak = np.abs(out).max()
    if peak > 1:
        out = out / peak
    return out


def main():
    if len(sys.argv) > 1:
        songfile = sys.argv[1]
        if not os.path.isfile(songfile):
            sys.exit(f"no such file: {songfile}")
    else:
        songfile = ask_path()

    room = ask_room()
    A = ask_absorption()
    dist = ask_distance(room)

    separator = demucs.api.Separator(
        model="htdemucs_6s",
        device=pick_device(),
        shifts=1,
        overlap=0.25,
        split=True,
        progress=True,
    )

    out = separate_and_spatialise(separator, songfile, A, room, dist)
    outfile = os.path.splitext(songfile)[0] + "_spatialised.wav"
    sf.write(outfile, out, SR)
    print(f"wrote {outfile}")

    # double-clicking the app on windows opens a console that closes as soon as we exit
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        input("Press Enter to exit")


if __name__ == "__main__":
    # in the packaged app, helper processes torch starts re-run this executable; this hands them off
    multiprocessing.freeze_support()
    main()
