# 📸 Gesture-Activated Photo Booth

Welcome to the **Gesture-Activated Photo Booth** project! 🎉 Capture memorable moments by simply making fun and unique gestures or facial expressions. Perfect for parties, events, or creating unforgettable memories! 🤳✨

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Supported Gestures & Expressions](#supported-gestures--expressions)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Features
- **Hand Gesture Recognition**: Detects various hand gestures to trigger photo capture.
- **Facial Expression Recognition**: Identifies specific facial expressions to take photos.
- **Automated Photo Capture**: Automatically takes photos when the correct gesture or expression is detected.
- **Real-Time Processing**: Utilizes MediaPipe for efficient and real-time landmark detection.

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
   *Ensure you have Python 3.7 or higher.*

## Usage 🚀

1. **Run the Photo Booth Application**
   ```bash
   python main.py
   ```
2. **Perform Gestures and Expressions**
   - Make a "V" sign ✌️
   - Perform the "OK" sign 👌
   - Create a fist ✊
   - Give a thumbs up 👍 (Like gesture)
   - Rock-n-Roll gesture 🤘
   - Smile 😊
   - Raise your eyebrows 🙆
   - Pucker your lips 😗

3. **Photo Capture**
   - The application will automatically take a photo when it detects any of the above gestures or expressions.

## Supported Gestures & Expressions 📸

### Hand Gestures ✋
- **Victory ("V") Sign** ✌️
- **OK Sign** 👌
- **Fist** ✊
- **Like (Thumbs Up)** 👍
- **Rock-n-Roll** 🤘

### Facial Expressions 😊
- **Smile** 😊
- **Raised Eyebrows** 🙆
- **Puckered Lips** 😗

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use and modify the code as per the license terms.

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
- Inspired by the [gesture-detection-emojis](https://github.com/bdekraker/gesture-detection-emojis) project.

---

Enjoy capturing your unique moments with the Gesture-Activated Photo Booth! 📷✨ If you have any questions or need support, feel free to reach out. Happy snapping! 😄
