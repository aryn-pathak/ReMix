import demucs.api
from numba.core.types import none

from audio import process_audio
import questionary
import questionary
from questionary import Choice
import numpy as np

separator = demucs.api.Separator(
    model="htdemucs_6s",
    device="cpu",
    shifts=1,
    overlap=0.25,
    split=True,
    progress=True,
)

dist = int(input("instrument distance from listener (m), (defaults to 3): ")) or 3

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

A = questionary.select(
    "Choose a room type:",
    choices=[Choice(name, value=coeffs) for name, coeffs in ROOMS.items()]
            + [Choice("Custom", value="custom")],
).ask()

if A == "custom":
    A = {
        surface: float(input(f"Absorption coefficient for {surface} (0–1):"))
        for surface in ("ceiling", "floor", "walls")
    }

roomSize = questionary.select(
    "Choose a room size:",
    choices=[
        Choice("Small",  value=(4, 3, 2.5)),
        Choice("Medium", value=(8, 6, 3)),
        Choice("Large",  value=(25, 15, 10)),
        Choice("Custom", value="custom"),
    ]
).ask()

if roomSize == "custom":
    s = input("Enter room size (x, y, z) in meters:")
    roomSize = tuple(float(x) for x in s.replace(",", " ").split())

instrument_order = ["drums", "vocals", "bass", "other" "guitar", "piano"]

def separate_and_spatialise(songfile):
    separated = separator.separate_audio_file(songfile)[1]
    listener_coords = ((roomSize[0] / 2), (roomSize[1] / 2), (roomSize[2] / 2))
    leftOut = None
    rightOut = None

    for i, instrument in enumerate(instrument_order, start=1):
        az = 360*i/len(instrument_order)
        left, right = process_audio(A, roomSize, separated[instrument], az, dist, listener_coords)
        if leftOut is None:
            leftOut = np.array(left, dtype=np.float64)
            rightOut = np.array(right, dtype=np.float64)
        else:
            leftOut += left
            rightOut += right

    max_sample = max(np.abs(leftOut).max(), np.abs(rightOut).max())
    if max_sample > 1:
        leftOut = leftOut / max_sample
        rightOut = rightOut / max_sample

    return leftOut, rightOut