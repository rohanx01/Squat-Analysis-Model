# Squat Analyser

Squat Analyser is a GUI-based real-time computer vision application that uses MediaPipe and OpenCV to assess squat form. It can analyze squats using either a webcam or a video file, providing feedback on common squat issues such as excessive spine flexion, heels lifting off the ground, and knee positioning. The application also tracks repetitions and provides visual cues for correct form.

## Features

- **Real-time Analysis**: Processes video frames in real-time for instant feedback.
- **Webcam and Video File Support**: Choose between live webcam input or analyzing pre-recorded video files.
- **Form Feedback**: Identifies and highlights common squat issues:
  - Excessive spine flexion
  - Heels lifting off the ground
  - Knees not tracking properly over toes
  - Proper squat depth
- **Rep Counter**: Automatically counts repetitions based on knee-hip angle.
- **GUI-Based**: User-friendly interface built using Tkinter for easy interaction.
## Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe** <br>
    Human pose detection by the trained model:
    - <img src="https://ai.google.dev/static/mediapipe/images/solutions/pose_landmarks_index.png" width=500mm height=500mm>
- **Tkinter**

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/squat-analyser.git
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```sh
   python squat_analyser.py
   ```
2. Choose to use either the webcam or select a video file for analysis through the GUI.<br>
   If using a Video File, for best results make sure the **left side view** of your squat is captured
4. Enjoy the seamless real-time analysis of your squat.<br>
    Checkout the `Results` Folder for some sample videos

## Squat Science References
[The Real Science of Squat](/https://squatuniversity.com/2016/04/20/the-real-science-of-the-squat/)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.
