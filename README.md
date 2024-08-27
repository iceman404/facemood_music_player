🎵 Face Mood Music Player 🎵
This project is a smart music player powered by a Raspberry Pi 4B 8GB, that detects a user's facial expression and plays music that matches their mood. The goal was to create an interactive and personalized music experience using computer vision, machine learning, and Python.

🚀 Features
Real-Time Mood Detection: Analyzes facial expressions using computer vision to determine 3 emotions: happiness, sadness, Surprised.
Automated Music Recommendation: Plays music that aligns with the detected mood from a predefined playlist.
Raspberry Pi Powered: Lightweight and designed to run smoothly on a Raspberry Pi, making it portable and energy-efficient. 
Customizable Playlists: Easily modify playlists and mood-matching logic to suit your preferences.

🎯 How It Works
Face Detection & Emotion Analysis: Uses OpenCV and Google's mediapipe library for accurate facial landmarks to detect faces.
Mood Classification: The detected emotions are mapped to specific 3 mood categories (Happy, Sad, Surprised). By default it is displays neutral.
Music Playback: Based on the classified mood, an appropriate playlist or song is selected and played using a Python-based audio player.

🛠️ Technologies Used
Python: Core programming language for scripting and logic.
OpenCV: For face detection and image processing.
TensorFlow/Keras: Pre-trained models for emotion recognition.
Mediapipe: Accurate facial landmarks which uses Tensorflow and the emotions are based on simple distance analysis rather than trained models to run on constrained edge devices.
Raspberry Pi 4B 8GB: The hardware platform, with easy setup and portability.

📷 Project Demo
Add a GIF or image of the project in action here.

💻 Installation & Setup
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/facemood_music_player.git
cd facemood_music_player

Install the required dependencies:

bash
Copy code
pip install -r requirements.txt

Set up your music playlists:
Organize your music files in folders according to moods (e.g., music/happy/, music/sad/, music/surprised/, etc.) and update the file paths in the code.

Run the application:

bash
Copy code
python facemood_player.py

Enjoy the mood-based music experience!

📂 Project Structure
bash
Copy code
facemood_music_player/
│
├── music/                # Music organized by mood categories
|  ├── happy/
|  ├── sad/
|  ├── surprised/
├── facemood_player.py    # Main script to run the application
├── requirements.txt      # Python dependencies
└── README.md             # This file

🎯 Customization
You can tweak the emotion-to-mood mapping and adjust the music selection logic by editing the facemood_player.py file. Feel free to extend the functionality by integrating other APIs or adding more emotion categories.

📖 Future Enhancements
Spotify Integration: Connect to a Spotify API to play personalized playlists(But expect delays depending upon your network connection).
Voice Control: Allow users to control the player with voice commands(Possibly accurate speech to text convertor using llm's)
Improved Accuracy: Fine-tune the emotion detection model for better accuracy based on the landmarks distances.

🤝 Contributing
Contributions are welcome! If you have ideas or improvements, feel free to submit a pull request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
