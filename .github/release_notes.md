Separates a song into six stems and places each one around you in a virtual room. Use headphones.

## Download

| System | File |
| --- | --- |
| macOS (Apple Silicon, M1 or newer) | `ReMix-macos-arm64.zip` |
| Windows 10/11 (64-bit) | `ReMix-windows-x64.zip` |

Unzip it and keep the `ReMix` folder together; the app needs the `_internal` folder next to it.

## First launch

The app isn't code-signed, so your system will warn you the first time.

**macOS:** open Terminal and run this once (drag the `ReMix` folder onto the Terminal window to fill in the path):

```sh
xattr -dr com.apple.quarantine /path/to/ReMix
```

Then double-click `ReMix` inside the folder. If you'd rather not use Terminal: double-click `ReMix`, close the warning, go to **System Settings → Privacy & Security**, click **Open Anyway** next to ReMix, and repeat for any other blocked file it mentions.

**Windows:** double-click `ReMix.exe`. If SmartScreen shows "Windows protected your PC", click **More info → Run anyway**.

## Using it

1. Drag your song into the window (or type its path) and press Enter. MP3, WAV, FLAC, OGG and AIFF work (not M4A/AAC).
2. Pick a room size, a room type and how far away the instruments are.
3. The first run downloads the Demucs separation model (about 50 MB), so you need internet once.
4. The result is saved next to your song as `<song>_spatialised.wav`.

On Apple Silicon, separation runs on the GPU. On Windows it runs on the CPU, which takes a few minutes per song.
