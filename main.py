import numpy as np
import sofar as sf
import matplotlib.pyplot as plt

N = 256  # no of samples in a measurement
T = 256 / 48000 * 1000  # length of a measurement in ms (48kHz sampling rate)

sofa = sf.read_sofa("SADIEII_KU100.sofa")
sources = sofa.SourcePosition[:, :2]

def getir(az, el):
    # Fixed: Handle 360-degree azimuth wrap-around correctly
    az_diff = (sources[:, 0] - az + 180) % 360 - 180
    el_diff = sources[:, 1] - el
    bestM = np.argmin(az_diff ** 2 + el_diff ** 2)

    # Fixed: Correct standard SOFA receiver mapping (0 = Left, 1 = Right)
    left_IR = sofa.Data_IR[bestM, 0, :]
    right_IR = sofa.Data_IR[bestM, 1, :]

    return right_IR, left_IR

def plot_ir(az, el):
    timeAxis = [i * (T / N) for i in range(N)]
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
    rightTime = (T / N) * firstsample(right)
    leftTime = (T / N) * firstsample(left)
    return leftTime - rightTime  # +ve means left

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
    plt.ylabel("absolute magnitude difference")
    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)

    mask = freqAxis <= 20000
    left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)  # 1/3-octave
    right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)

    freqDiff_smooth = left_smooth - right_smooth
    freqs = freqAxis[mask]
    diff = freqDiff_smooth[mask]

    pos_diff = np.where(diff >= 0, diff, np.nan)
    neg_diff = np.where(diff < 0, np.abs(diff), np.nan)

    plt.plot(freqs, pos_diff, color="red", label="Left")
    plt.plot(freqs, neg_diff, color="blue", label="Right")

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.show()

def plot_freq(az, el, max_freq=20000):
    right_freq, left_freq = get_freq(az, el)
    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)
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

def plot_heatmapDiff():
    azimuth = sources[:, 0]
    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)
    mask = freqAxis <= 20000

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

def plot_heatmap():
    azimuth = sources[:, 0]
    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)
    mask = freqAxis <= 20000

    freqs = freqAxis[mask]
    az_rad = np.deg2rad(azimuth)
    Theta, R = np.meshgrid(az_rad, freqs, indexing='ij')

    # Create two separate matrices for left and right magnitudes
    Z_left = np.zeros((len(azimuth), len(freqs)))
    Z_right = np.zeros((len(azimuth), len(freqs)))

    for i, az in enumerate(azimuth):
        right_freq, left_freq = get_freq(az, 0)
        left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)
        right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)

        # Store the magnitudes
        Z_left[i, :] = np.abs(left_smooth[mask])
        Z_right[i, :] = np.abs(right_smooth[mask])

    # Calculate global min and max so both plots share the same color scale
    vmin = min(Z_left.min(), Z_right.min())
    vmax = max(Z_left.max(), Z_right.max())

    # Create a figure with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(16, 8))

    # Plot Left Magnitude
    c1 = axs[0].pcolormesh(Theta, R, Z_left, cmap='magma', shading='nearest', vmin=vmin, vmax=vmax)
    axs[0].set_theta_zero_location("N")
    axs[0].set_theta_direction(-1)
    axs[0].set_title("Left Magnitude", pad=20)

    # Plot Right Magnitude
    c2 = axs[1].pcolormesh(Theta, R, Z_right, cmap='magma', shading='nearest', vmin=vmin, vmax=vmax)
    axs[1].set_theta_zero_location("N")
    axs[1].set_theta_direction(-1)
    axs[1].set_title("Right Magnitude", pad=20)

    # Add a shared colorbar for both subplots
    fig.colorbar(c2, ax=axs, label='Magnitude', pad=0.1)

    plt.show()

plot_freq(90,0)