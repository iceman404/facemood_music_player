# Face Mood Music Player

A professional emotion-based music player that uses computer vision to detect facial expressions and automatically plays music matching the user's mood. The application analyzes facial landmarks in real-time using MediaPipe and OpenCV, then selects appropriate music tracks based on detected emotions.

## Features

- **Real-Time Emotion Detection**: Analyzes facial expressions to detect emotions including happiness, sadness, surprise, and anger
- **Automated Music Playback**: Automatically plays mood-aligned music from predefined playlists
- **Modular Architecture**: Professional code structure with separate modules for configuration, face detection, emotion analysis, and music playback
- **Statistics Tracking**: Tracks emotion detection statistics and music playback history
- **Volume Control**: Adjustable volume with keyboard shortcuts
- **Playlist Management**: Supports multiple audio formats (MP3, WAV, OGG, FLAC)
- **Configuration System**: JSON-based configuration for easy customization
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Cross-Platform**: Works on Linux, Windows, and macOS

## How It Works

1. **Face Detection**: Uses MediaPipe Face Detection to locate faces in the camera feed
2. **Landmark Extraction**: Extracts facial landmarks using MediaPipe Face Mesh
3. **Emotion Analysis**: Calculates facial feature distances (mouth, eyebrows, eyes) to classify emotions
4. **Music Selection**: When an emotion is detected consistently, plays a random track from the corresponding playlist
5. **Continuous Monitoring**: Continuously monitors facial expressions and adapts music selection accordingly

## Technologies Used

- **Python 3.7+**: Core programming language
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Facial landmark detection and tracking
- **Pygame**: Audio playback and music management
- **NumPy**: Numerical computations for facial feature analysis

## Installation & Setup

### Prerequisites

- Python 3.7 or higher
- Webcam or camera device
- Audio output device

### Clone the Repository

```bash
git clone https://github.com/iceman404/facemood_music_player.git
cd facemood_music_player
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Up Music Playlists

Organize your music files in folders according to emotions:

```
music/
├── happy/
│   ├── song1.mp3
│   ├── song2.mp3
│   └── ...
├── sad/
│   ├── song1.mp3
│   └── ...
├── surprised/
│   ├── song1.mp3
│   └── ...
└── angry/
    ├── song1.mp3
    └── ...
```

Supported audio formats: MP3, WAV, OGG, FLAC

### Configuration

The application uses a `config.json` file for configuration. On first run, a default configuration file will be created. You can customize:

- Camera settings (device ID, resolution)
- Face detection confidence thresholds
- Emotion detection thresholds
- Music playback settings (volume, fade duration)
- Display options (landmarks, colors, fonts)
- Logging preferences

### Run the Application

```bash
python facemood_player.py
```

Or with custom configuration:

```bash
python facemood_player.py --config custom_config.json
```

Or with custom music path:

```bash
python facemood_player.py --music-path /path/to/music
```

## Keyboard Controls

- **q**: Quit the application
- **s**: Show statistics summary
- **p**: Pause/Resume music playback
- **+** or **=**: Increase volume
- **-**: Decrease volume

## Project Structure

```
facemood_music_player/
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── face_detector.py    # Face detection using MediaPipe
│   ├── emotion_detector.py # Emotion detection logic
│   ├── music_player.py     # Music playback management
│   ├── statistics.py       # Statistics tracking
│   ├── utils.py            # Utility functions
│   └── main.py             # Main application logic
│
├── music/                  # Music playlists organized by emotion
│   ├── happy/
│   ├── sad/
│   ├── surprised/
│   └── angry/
│
├── facemood_player.py      # Entry point script
├── config.json             # Configuration file (auto-generated)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── LICENSE                 # License file
```

## Customization

### Adjusting Emotion Detection Sensitivity

Edit `config.json` to modify emotion detection thresholds:

```json
{
  "emotion": {
    "threshold_count": 40,
    "sad": {
      "mouth_threshold": 0.1,
      "brow_inner_threshold": 0.05,
      "eye_openness_threshold": 0.15
    },
    "happy": {
      "mouth_distance_threshold": 0.45
    }
  }
}
```

### Changing Music Settings

Modify music-related settings in `config.json`:

```json
{
  "music": {
    "volume": 0.7,
    "fade_duration": 1.0,
    "base_path": "music"
  }
}
```

### Camera Configuration

Adjust camera settings:

```json
{
  "camera": {
    "device_id": 0,
    "width": 640,
    "height": 480
  }
}
```

## Troubleshooting

### Camera Not Working

- Ensure your camera is connected and not being used by another application
- Try changing the `device_id` in `config.json` (0, 1, 2, etc.)
- Check camera permissions on your system

### No Music Playing

- Verify that music files exist in the emotion folders
- Check that audio files are in supported formats (MP3, WAV, OGG, FLAC)
- Ensure audio output device is working
- Check the log file (`facemood.log`) for error messages

### Poor Emotion Detection

- Ensure good lighting conditions
- Position face clearly in front of camera
- Adjust emotion thresholds in `config.json` based on your observations
- Check that face is fully visible in the frame

## Development

### Running from Source

To run the application directly from the source code:

```bash
python -m src.main
```

### Adding New Emotions

1. Add emotion detection logic in `src/emotion_detector.py`
2. Create corresponding music folder (e.g., `music/neutral/`)
3. Update emotion counter initialization in `src/main.py`
4. Add configuration thresholds in `config.json`

### Extending Functionality

The modular architecture makes it easy to extend:

- Add new face detection backends
- Implement machine learning-based emotion detection
- Add support for streaming music services
- Integrate with smart home systems
- Add web interface or API

## Future Enhancements

- Machine learning-based emotion recognition for improved accuracy
- Integration with music streaming services (Spotify, Apple Music)
- Voice control for manual music selection
- Multi-face detection and tracking
- Emotion history visualization
- Web-based dashboard for statistics
- Mobile app companion
- Support for video file input

## Contributing

Contributions are welcome! If you have ideas or improvements, please feel free to submit a pull request. When contributing:

1. Follow the existing code style
2. Add appropriate logging
3. Update documentation as needed
4. Test your changes thoroughly

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe for facial landmark detection
- OpenCV community for computer vision tools
- Pygame for audio playback capabilities
