# 🎵 Face Mood Music Player 🎵

This project is a smart music player powered by a Raspberry Pi 4B 8GB that detects a user's facial expression and plays music that matches their mood. The goal was to create an interactive and personalized music experience using computer vision, machine learning, and Python.

## 🚀 Features

- **Real-Time Mood Detection**: Analyzes facial expressions to determine 3 emotions: happiness, sadness, and surprise.
- **Automated Music Recommendation**: Plays mood-aligned music from a predefined playlist.
- **Raspberry Pi Powered**: Designed to run smoothly on a Raspberry Pi, making it portable and efficient.
- **Customizable Playlists**: Modify playlists and mood-matching logic easily.

## 🎯 How It Works

1. **Face Detection & Emotion Analysis**: Uses OpenCV and MediaPipe for accurate facial landmarks.
2. **Mood Classification**: Maps emotions to 3 mood categories (Happy, Sad, Surprised), with neutral as default.
3. **Music Playback**: Plays a suitable playlist using a Python-based audio player.

## 🛠️ Technologies Used

- **Python**: Core scripting language.
- **OpenCV**: For face detection and image processing.
- **TensorFlow/Keras**: Pre-trained models for emotion recognition.
- **MediaPipe**: Facial landmarks for efficient emotion detection.
- **Raspberry Pi 4B 8GB**: Hardware platform for portability.

## 📷 Project Demo

![Project Demo](https://via.placeholder.com/800x400.png?text=Demo+GIF+Here)

*Add a GIF or image here.*

## 💻 Installation & Setup

### Clone the repository:

```bash
git clone https://github.com/yourusername/facemood_music_player.git
cd facemood_music_player
```

### Install the dependencies:

```bash
pip install -r requirements.txt
```

### Set up your music playlists:

Organize your music files in folders according to moods (e.g., `music/happy/`, `music/sad/`, `music/surprised/`) and update the paths in the code.

### Run the application:

```bash
python facemood_player.py
```

Enjoy the mood-based music experience!

## 📂 Project Structure

```plaintext
facemood_music_player/
│
├── music/
│   ├── happy/
│   ├── sad/
│   ├── surprised/
├── facemood_player.py
├── requirements.txt
└── README.md
```

## 🎯 Customization

You can tweak the emotion-to-mood mapping and adjust the music selection logic by editing the `facemood_player.py` file. Feel free to extend the functionality by integrating more APIs or adding additional emotion categories.

## 📖 Future Enhancements

- **Spotify Integration**: Connect to a Spotify API (with possible delays).
- **Voice Control**: Implement voice commands.
- **Improved Accuracy**: Fine-tune based on landmark distances.

## 🤝 Contributing

Contributions are welcome! If you have ideas or improvements, feel free to submit a pull request.

[![Contribute](https://img.shields.io/badge/Contribute-PRs_Welcome-blue?style=for-the-badge)](https://github.com/iceman404/facemood_music_player/pulls)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
