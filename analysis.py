import numpy as np
import sofar as sf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 256  # no of samples in a measurement
T = 256 / 48000 * 1000  # length of a measurement in ms (48kHz sampling rate)

sofa = sf.read_sofa("SADIEII_KU100.sofa")
sources = sofa.SourcePosition[:, :2]

eq_idx = np.where(np.isclose(sources[:, 1], 0, atol=1e-5))[0]
az_list = np.sort(sources[eq_idx, 0])

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

# heatmap of ILD
def plot_heatmapDiff():
    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)
    mask = freqAxis <= 20000
    freqs = freqAxis[mask]

    az_rad = np.deg2rad(az_list)
    Theta, R = np.meshgrid(az_rad, freqs, indexing='ij')
    Z = np.zeros((len(az_list), len(freqs)))

    for i, az in enumerate(az_list):
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

# plots heatmap of left and right IRs separately
def plot_heatmap():
    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)
    mask = freqAxis <= 20000
    freqs = freqAxis[mask]

    az_rad = np.deg2rad(az_list)
    Theta, R = np.meshgrid(az_rad, freqs, indexing='ij')

    Z_left = np.zeros((len(az_list), len(freqs)))
    Z_right = np.zeros((len(az_list), len(freqs)))

    for i, az in enumerate(az_list):
        right_freq, left_freq = get_freq(az, 0)
        left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)
        right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)

        Z_left[i, :] = left_smooth[mask]
        Z_right[i, :] = right_smooth[mask]

    abs_max = max(np.abs(Z_left).max(), np.abs(Z_right).max())
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(16, 8))

    c1 = axs[0].pcolormesh(Theta, R, Z_left, cmap='RdBu_r', shading='nearest', vmin=-abs_max, vmax=abs_max)
    axs[0].set_theta_zero_location("N")
    axs[0].set_theta_direction(-1)
    axs[0].set_title("Left Amplitude", pad=20)

    c2 = axs[1].pcolormesh(Theta, R, Z_right, cmap='RdBu_r', shading='nearest', vmin=-abs_max, vmax=abs_max)
    axs[1].set_theta_zero_location("N")
    axs[1].set_title("Right Amplitude", pad=20)
    axs[1].set_theta_direction(-1)

    fig.colorbar(c2, ax=axs, label='Amplitude', pad=0.1)
    plt.show()

def plot_heatmapAvg():
    elevation = sources[:, 1]
    idx = np.where(np.isclose(elevation, 0, atol=1e-5))[0]
    azimuth = np.sort(sources[idx, 0])
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

        Z[i, :] = (left_smooth[mask] + right_smooth[mask]) / 2

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    c = ax.pcolormesh(Theta, R, Z, cmap='magma', shading='nearest')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    plt.colorbar(c, ax=ax, label='Arithmetic average of left and right', pad=0.1)
    plt.show()

def animate_freqDiff(az_list, el, interval=200, save_path=None):

    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)
    mask = freqAxis <= 20000
    freqs = freqAxis[mask]

    fig, ax = plt.subplots()
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("absolute magnitude difference")
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # placeholder lines, updated each frame
    line_pos, = ax.plot([], [], color="red", label="Left")
    line_neg, = ax.plot([], [], color="blue", label="Right")
    ax.legend()

    # set static x-limits up front; y-limits we'll adjust based on data
    ax.set_xlim(freqs.min(), freqs.max())

    title = ax.set_title("")

    def compute_diff(az, el):
        right_freq, left_freq = get_freq(az, el)
        left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)
        right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)
        freqDiff_smooth = left_smooth - right_smooth
        diff = freqDiff_smooth[mask]
        pos_diff = np.where(diff >= 0, diff, np.nan)
        neg_diff = np.where(diff < 0, np.abs(diff), np.nan)
        return pos_diff, neg_diff

    # precompute global y-limit so the axis doesn't jump around between frames
    all_vals = []
    for az in az_list:
        p, n = compute_diff(az, el)
        all_vals.append(p)
        all_vals.append(n)
    all_vals = np.concatenate(all_vals)
    ymax = np.nanmax(all_vals)
    ax.set_ylim(0, ymax * 1.05)

    def update(frame_idx):
        az = az_list[frame_idx]
        pos_diff, neg_diff = compute_diff(az, el)
        line_pos.set_data(freqs, pos_diff)
        line_neg.set_data(freqs, neg_diff)
        title.set_text(f"az = {az}°, el = {el}°")
        return line_pos, line_neg, title

    anim = FuncAnimation(fig, update, frames=len(az_list), interval=interval, blit=False)

    if save_path:
        anim.save(save_path)  # needs ffmpeg for .mp4, pillow for .gif
    else:
        plt.show()

    return anim  # keep a reference so it doesn't get garbage-collected

def animate_freq(az_list, el, max_freq=20000, interval=200, save_path=None):
    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)
    mask = freqAxis <= max_freq
    freqs = freqAxis[mask]

    frames_data = []
    global_min = float('inf')
    global_max = float('-inf')

    print("Precomputing Left/Right frames...")
    for az in az_list:
        right_freq, left_freq = get_freq(az, el)
        left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)
        right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)

        l_data = left_smooth[mask]
        r_data = right_smooth[mask]

        # Track limits for a stable Y-axis
        current_min = min(l_data.min(), r_data.min())
        current_max = max(l_data.max(), r_data.max())
        if current_min < global_min: global_min = current_min
        if current_max > global_max: global_max = current_max

        frames_data.append((l_data, r_data, az))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Add 5% padding to the top and bottom of the Y-axis
    y_range = global_max - global_min
    ax.set_ylim(global_min - 0.05 * y_range, global_max + 0.05 * y_range)
    ax.set_xlim(freqs.min(), freqs.max())

    bark_edges = [
        0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
        1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
        9500, 12000, 15500, 20000
    ]

    for i in range(len(bark_edges) - 1):
        low = bark_edges[i]
        high = bark_edges[i + 1]

        if low > max_freq:
            break
        high = min(high, max_freq)

        alpha_val = 0.15 if i % 2 == 0 else 0.05
        ax.axvspan(low, high, color='gray', alpha=alpha_val)

        ax.axvline(high, color='black', linestyle=':', alpha=0.3)

    line_right, = ax.plot([], [], color="blue", label="Right Ear", linewidth=1.5)
    ax.legend()

    title = ax.set_title("")

    def update(frame_idx):
        l_data, r_data, az = frames_data[frame_idx]

        # line_left.set_data(freqs, l_data)
        line_right.set_data(freqs, r_data)
        title.set_text(f"HRTF Magnitude Spectrum (Azimuth: {az}°, Elevation: {el}°)")

        return line_right, title

    anim = FuncAnimation(fig, update, frames=len(az_list), interval=interval, blit=False)

    if save_path:
        anim.save(save_path)
    else:
        plt.show()

    return anim

def animate_freqAvg(az_list, el, max_freq=20000, interval=200, save_path=None):
    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)
    mask = freqAxis <= max_freq
    freqs = freqAxis[mask]

    frames_data = []
    global_min = float('inf')
    global_max = float('-inf')

    print("Precomputing Average frames...")
    for az in az_list:
        right_freq, left_freq = get_freq(az, el)
        left_smooth = smooth_fractional_octave(freqAxis, left_freq, fraction=3)
        right_smooth = smooth_fractional_octave(freqAxis, right_freq, fraction=3)

        avg_data = (left_smooth[mask] + right_smooth[mask]) / 2

        # Track limits for a stable Y-axis
        if avg_data.min() < global_min: global_min = avg_data.min()
        if avg_data.max() > global_max: global_max = avg_data.max()

        frames_data.append((avg_data, az))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Add 5% padding to the top and bottom of the Y-axis
    y_range = global_max - global_min
    ax.set_ylim(global_min - 0.05 * y_range, global_max + 0.05 * y_range)
    ax.set_xlim(freqs.min(), freqs.max())

    line_avg, = ax.plot([], [], color="purple", label="Average (Left & Right)", linewidth=1.5)
    ax.legend()

    title = ax.set_title("")

    def update(frame_idx):
        avg_data, az = frames_data[frame_idx]

        line_avg.set_data(freqs, avg_data)
        title.set_text(f"Average Magnitude Spectrum (Azimuth: {az}°, Elevation: {el}°)")

        return line_avg, title

    anim = FuncAnimation(fig, update, frames=len(az_list), interval=interval, blit=False)

    if save_path:
        anim.save(save_path)
    else:
        plt.show()

    return anim

# az_list = np.arange(0, 360, 1)
# anim = animate_freq(az_list, el=0)

def plot_error():
    freqAxis = np.fft.rfftfreq(n=N, d=1 / 48000)
    mask = freqAxis <= 20000

    all_curves = []
    for az in az_list:
        curve = smooth_fractional_octave(freqAxis, get_freq(az, 0), fraction=3)
        all_curves.append(curve[mask])  # Mask applied once

    all_curves = np.array(all_curves)
    errorArr = []
    window = 5

    for i in range(len(az_list) - window + 1):
        curves_subset = all_curves[i: i + window]
        avgCurve = np.mean(curves_subset, axis=0)  # No double-masking
        error = np.mean(np.linalg.norm(curves_subset - avgCurve, axis=1))

        errorArr.append(error)

    plt.figure(figsize=(10, 5))
    valid_azimuths = az_list[:len(errorArr)]

    plt.plot(valid_azimuths, errorArr, marker='.', color='purple')
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Spectral Difference (Smoothed)")
    plt.title("Smoothed Spectral Difference Between Adjacent Azimuths (Elevation: 0°)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()

plot_error()