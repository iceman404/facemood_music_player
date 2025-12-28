# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your music files:
   - Place music files in `music/happy/`, `music/sad/`, `music/surprised/`, and `music/angry/` folders
   - Supported formats: MP3, WAV, OGG, FLAC

## Running the Application

```bash
python facemood_player.py
```

## First Run

On first run, a `config.json` file will be automatically created with default settings. You can customize it later.

## Basic Usage

1. Position yourself in front of the camera
2. Make facial expressions (happy, sad, surprised, angry)
3. The application will detect your emotion and play matching music
4. Use keyboard controls:
   - `q`: Quit
   - `s`: Show statistics
   - `p`: Pause/Resume music
   - `+`/`=`: Increase volume
   - `-`: Decrease volume

## Troubleshooting

- **Camera not working**: Check camera permissions and try different device IDs in `config.json`
- **No music playing**: Ensure music files exist in emotion folders and are in supported formats
- **Poor detection**: Ensure good lighting and face is clearly visible

For more details, see the main [README.md](README.md).

