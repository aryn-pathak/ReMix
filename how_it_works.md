# The cues, and what I explored

## Falloff: `apply_falloff()`

How sound loses energy with distance (inverse square law).

## Air absorption: `apply_eq()`, `apply_air_absorption()`

Air absorbs higher frequencies more than lower ones. I expected this to be minor, but it needed a whole EQ system, whose maths was interesting and took ages to wrap my head around. It also taught me about filters, which I used in later functions.

Changes to one frequency interfered heavily with others, which I fixed with a peaking biquad filter (mostly AI-written). The attenuation constants come from a one-time-run file using the ISO standard, and are multiplied by distance and fed into the EQ.

## Early reflections: `early_reflections()`

Initial reflections from nearby surfaces, ~5-35ms after the direct sound. By the Haas effect the brain doesn’t hear them separately, but they imply spaciousness. I first used IACC and ASW (below), then dropped them for directional cues (HRIR), so each of the 6 reflections comes from its surface’s direction. It accounts for room size and pressure reflection (from an absorption coefficient).

## Decorrelation: `decorrelate()`

Artificially decorrelates a signal (FIR comb filters) into L and R, lowering IACC (interaural cross-correlation) and increasing ASW (apparent source width). An IACC of 1 sounds like a point source; lower sounds wider. It worked against my goal (localisation, not a larger source), so I dropped it from the direct audio, then from early reflections. Scrapped, but an interesting concept.

## Reverb: `room_reverb()`

Probably the most important cue for space. Late reverb is thousands of colliding reflections, an unintelligible wash. I built it from feedback comb filters, combined and diffused through an allpass filter, and cut off at RT60 (time to decay 60 dB, from speed of sound, room size and absorption).

**Problem:** the tail was distinct and ruined localisation (which relies on level differences between ears), like in a hall. Most of the energy making it legible was in its first ~50ms, masking the early reflections. Cutting the first [early reflection length - 20ms] helped a bit. Switching to an FDN made it denser and decorrelated L/R by design, so I could drop `decorrelate()`.

## DRR (part of reverb)

Direct-to-reverberant ratio simulates source distance relative to the room: closer to the listener than the walls means louder direct sound, closer to the walls means louder reverb. I set the direct amplitude to 1 and calculated a gain for the reverb.

## Apply IR

The only cue for direction. It convolves left and right impulse responses for an azimuth and elevation onto the direct audio and early reflections. They come from a HRIR dataset (.SOFA, SADIE II) recorded on a binaural mannequin. I tried optimising this earlier and failed (older devlogs), and want to revisit it.

## Process audio

1. Converts to mono, resamples to 48kHz if needed.
2. Gets azimuth, elevation and distance from coordinates.
3. Applies air absorption, falloff, then IR on the direct sound.
4. Generates early reflections and late reverb (L and R).
5. Sums everything, scales to prevent clipping, returns L and R arrays :D
