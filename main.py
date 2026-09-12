import numpy as np
import sofar as sf
import matplotlib.pyplot as plt

N = 384 # no of samples in a measurement
T = 8 # length of a measurement in ms (48kHz sampling rate)

sofa = sf.read_sofa("KU100051023_4_processed.sofa")
sources = sofa.SourcePosition[:, :2]

def getir(az, el):
    difference = sources - [az, el]
    bestM = np.argmin(np.sum(difference ** 2, axis=1))
    right_IR = sofa.Data_IR[bestM, 0, :]
    left_IR = sofa.Data_IR[bestM, 1, :]
    return right_IR, left_IR

def plot_ir(az, el):
    timeAxis = [i*(T/N) for i in range(384)]
    plt.plot(timeAxis, getir(az, el)[1], color="red", label="Left IR")
    plt.plot(timeAxis, getir(az, el)[0], color="blue", label="Right IR")
    plt.xlabel("time (ms)")
    plt.ylabel("values")
    plt.legend()
    plt.show()

def firstsample(wave, threshold=0.15):
    peak = np.max(np.abs(wave))
    threshold = threshold * peak
    return np.argmax(np.abs(wave) >= threshold)

def itd(right, left):
    rightTime = (T/N)*firstsample(right)
    leftTime = (T/N)*firstsample(left)
    return leftTime - rightTime # +ve means left

def plot_itd(el):
    elevation = sources[:, 1]
    idx = np.where(np.isclose(elevation, el, atol=1e-5))[0]
    azAxis = np.sort(sources[idx, 0])

    itdarr = []
    for az in azAxis:
        itdarr.append(itd(*getir(az, 0)))

    plt.xlabel("azimuth (degrees)")
    plt.ylabel("ITD (ms)")
    plt.plot(azAxis, itdarr)
    plt.show()

def get_freq(az, el):
    right_IR, left_IR = getir(az, el)
    d = itd(right_IR, left_IR)


    right_freq = 20 * np.log10(np.abs(np.fft.rfft(right_IR)) + 1e-9)
    left_freq = 20 * np.log10(np.abs(np.fft.rfft(left_IR)) + 1e-9)
    return right_freq, left_freq

def smooth_fractional_octave(freq, mag_db, fraction=3):
    smoothed = np.zeros_like(mag_db)
    for i, f in enumerate(freq):
        if f == 0:
            smoothed[i] = mag_db[i]
            continue
        f_lo = f / (2 ** (1 / (2 * fraction)))
        f_hi = f * (2 ** (1 / (2 * fraction)))
        mask = (freq >= f_lo) & (freq <= f_hi)
        smoothed[i] = np.mean(mag_db[mask])
    return smoothed

def plot_freqDiff(az, el):
    right_freq, left_freq = get_freq(az, el)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("magnitude")
    freqAxis = np.fft.rfftfreq(n=384, d=1 / 48000)

    mask = freqAxis <= 20000
    left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)  # 1/3-octave
    right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)
    freqDiff_smooth = left_smooth - right_smooth

    plt.plot(freqAxis[mask], freqDiff_smooth[mask], color="red", label="Frequency Difference")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.show()

def plot_freq(az, el, max_freq=20000):
    right_freq, left_freq = get_freq(az, el)
    freqAxis = np.fft.rfftfreq(n=384, d=1 / 48000)
    mask = freqAxis <= max_freq

    left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)
    right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)

    plt.figure(figsize=(10, 5))
    plt.plot(freqAxis[mask], left_smooth[mask], color="red", label="Left Ear", linewidth=1.5)
    plt.plot(freqAxis[mask], right_smooth[mask], color="blue", label="Right Ear", linewidth=1.5)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"HRTF Magnitude Spectrum (Azimuth: {az}°, Elevation: {el}°)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.show()

azimuth = sources[:, 0]

freqAxis = np.fft.rfftfreq(n=384, d=1 / 48000)
mask = freqAxis <= 20000

fig, ax = plt.subplots()
Lcmap = plt.get_cmap("viridis")
Rcmap = plt.get_cmap("magma")

# for i, az in enumerate(azimuth):
#     right_freq, left_freq = get_freq(az, 0)
#     left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)
#     right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)
#     freqDiff_smooth = left_smooth[mask] - right_smooth[mask]
#
#     if sum(freqDiff_smooth) < 0:
#         plt.plot(freqAxis[mask], np.abs(freqDiff_smooth), color=Rcmap(i / len(azimuth)))
#     # elif sum(freqDiff_smooth) > 0:
#         # plt.plot(freqAxis[mask], np.abs(freqDiff_smooth), color=Lcmap(i / len(azimuth)))
#     else:
#         continue
# plt.show()

freqs = freqAxis[mask]
az_rad = np.deg2rad(azimuth)
Theta, R = np.meshgrid(az_rad, freqs, indexing='ij')
Z = np.zeros((len(azimuth), len(freqs)))

for i, az in enumerate(azimuth):
    right_freq, left_freq = get_freq(az, 0)
    left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)
    right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)

    freqDiff_smooth = left_smooth[mask] - right_smooth[mask]
    Z[i, :] = np.abs(freqDiff_smooth)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
c = ax.pcolormesh(Theta, R, Z, cmap='magma', shading='nearest')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

plt.colorbar(c, ax=ax, label='Magnitude (|Left - Right|)', pad=0.1)
plt.show()