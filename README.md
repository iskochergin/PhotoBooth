# 📸 Gesture-Activated Photo Booth

Welcome to the **Gesture-Activated Photo Booth** project! Capture memorable moments by simply making fun and unique gestures or facial expressions. Perfect for parties, events, or creating unforgettable memories! 🤳✨

## Features 🎯
- **Hand Gesture Recognition**: Detects various hand gestures to trigger photo capture. ✋
- **Facial Expression Recognition**: Identifies specific facial expressions to take photos. 😊
- **Multiple Gestures and Expressions**: Supports a variety of gestures and expressions for photo capture. 🤘👍
- **Automated Photo Capture**: Automatically takes photos when the correct gesture or expression is detected. 📸
- **Real-Time Processing**: Utilizes MediaPipe for efficient and real-time landmark detection.⚡️
- **Intelligent Image Capture**: Ensures crystal-clear photos by automatically discarding blurred images and capturing only when the subject is in sharp focus. 🔍📷

## Installation 🛠️

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/gesture-photo-booth.git
   cd gesture-photo-booth
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage 🚀

1. **Run the Photo Booth Application**
   ```bash
   python main.py
   ```
2. **Perform Gestures and Expressions**
   - Make a "V" sign ✌️
   - Perform the "OK" sign 👌
   - Create a fist ✊
   - Give a thumbs up 👍
   - Rock-n-Roll gesture 🤘
   - Smile 😊
   - Raise your eyebrows 🙆
   - Pucker your lips 😗
   - Make a heart sign 🫶
   - ENJOY SIGMA EMOJI!!! 🤫🧏

3. **Photo Capture**
   - The application will automatically take a photo when it detects any of the above gestures or expressions and save it in the folder "sessions\date_time" which is created for each session playing a nice sound! After processing all photos program will automatically delete blured ones.

4. **Some Photos**
   ![Photo 1](media/photo1.png)
   ![Photo 2](media/photo2.png)
   _The photos are displayed with highlighted emotions and gestures to demonstrate how the program works. However, it actually saves results without any highlights._  

## Credits 📝

This project is programmed by Ivan Kochergin and licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Contributing 🤝

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "Add Your Feature"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

## Acknowledgements 🙏

- **[MediaPipe](https://mediapipe.dev/)** for the powerful landmark detection.
- **[NumPy](https://numpy.org/)** and **[OpenCV](https://opencv.org/)** for providing essential tools for computer vision tasks.
