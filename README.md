# Squat Analysis Model
Squat Analyser is a GUI-based real-time computer vision application that uses MediaPipe and OpenCV to assess squat form. It can analyze squats using either a webcam or a video file, providing feedback on common squat issues such as excessive spine flexion, heels lifting off the ground, and knee positioning. The application also tracks repetitions and provides visual cues for correct form.
## Features
* **Real-time Analysis**: Processes video frames in real-time for instant feedback.
* **Webcam and Video File Support**: Choose between live webcam input or analyzing pre-recorded video files.
* **Form Feedback**: Identifies and highlights common squat issues:
  * Excessive spine flexion
  * Heels lifting off the ground
  * Knees not tracking properly over toes
  * Proper squat depth
* **Rep Counter**: Automatically counts repetitions based on knee-hip angle.
* **GUI-Based**: User-friendly interface built using Tkinter for easy interaction.
## Technologies Used
* Python
* OpenCV
* MediaPipe
* Tkinter

## Installation
1. Clone the repository
   `git clone https://github.com/yourusername/squat-analyser.git`
3. Install the required packages:
   `pip install -r requirements.txt`
## Usage
1. Run the application
   `python squat_analyser.py`

3. Choose to use either the webcam or select a video file for analysis through the GUI.
4. See the real-time feedback seemlessly.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contribution
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.





