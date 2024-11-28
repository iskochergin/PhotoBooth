import cv2
import datetime
import mediapipe as mp
import numpy as np
import math
import os
from delete_blured import delete_blurred_images

SESSION_NAME = f'session_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.mkdir(SESSION_NAME)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


def calculate_distance(point1, point2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return math.sqrt((point1.x - point2.x) ** 2 +
                     (point1.y - point2.y) ** 2 +
                     (point1.z - point2.z) ** 2)


def calculate_normalized_distance(x1, y1, x2, y2, image_width, image_height):
    """Вычисляет нормализованное расстояние между двумя точками."""
    dx = (x1 - x2) * image_width
    dy = (y1 - y2) * image_height
    return math.sqrt(dx ** 2 + dy ** 2)


def calculate_scale(face_landmarks, image_width, image_height):
    """Вычисляет масштаб лица на основе межзрачкового расстояния."""
    LEFT_EYE = 159
    RIGHT_EYE = 386

    left_eye = face_landmarks.landmark[LEFT_EYE]
    right_eye = face_landmarks.landmark[RIGHT_EYE]

    interocular_distance = calculate_normalized_distance(
        left_eye.x, left_eye.y, right_eye.x, right_eye.y, image_width, image_height
    )

    return interocular_distance


def is_finger_extended(hand_landmarks, finger_tip, finger_pip):
    """Проверяет, вытянут ли палец"""
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_pip].y


def calculate_angle(landmark1, landmark2):
    """Вычисление угла между двумя точками"""
    return np.arctan2(landmark2.y - landmark1.y, landmark2.x - landmark1.x) * 180 / np.pi


def is_hand_reversed(hand_landmarks, handedness):
    """Проверка того, что ладонь перевернуто, этого надо избежать при определении жеста СТОП"""
    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

    angle_thumb = calculate_angle(wrist, thumb_tip)
    angle_index = calculate_angle(wrist, index_tip)

    if handedness.classification[0].label == 'Right':
        return angle_thumb > angle_index
    else:
        return angle_thumb < angle_index


def are_fingers_visible(hand_landmarks):
    """Проверка того, видны ли пальцы рук, нужно для определения жеста СТОП"""
    index_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

    if index_tip.y < index_pip.y and middle_tip.y < middle_pip.y:
        return True
    return False


def is_stop_signal(hand_landmarks):
    """Проверка положения руки в нужном состоянии для жеста СТОП"""
    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
    pinky_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    if (wrist.y > index_finger_tip.y and wrist.y > middle_finger_tip.y and index_finger_tip.y > middle_finger_tip.y and
            pinky_finger_tip.y > middle_finger_tip.y):
        return True
    return False


def is_v_gesture(hand_landmarks):
    """Распознавание жеста 'V' или 'Победа'."""
    is_index_up = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                     mp_hands.HandLandmark.INDEX_FINGER_DIP)
    is_middle_up = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                      mp_hands.HandLandmark.MIDDLE_FINGER_DIP)
    is_ring_down = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP,
                                          mp_hands.HandLandmark.RING_FINGER_DIP)
    is_pinky_down = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP,
                                           mp_hands.HandLandmark.PINKY_PIP)

    return is_index_up and is_middle_up and is_ring_down and is_pinky_down


def is_ok_gesture(hand_landmarks):
    """Распознавание жеста 'ОК'."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = calculate_distance(thumb_tip, index_tip)

    is_ok_distance = distance < 0.03

    is_middle_extended = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                            mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    is_ring_extended = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP,
                                          mp_hands.HandLandmark.RING_FINGER_PIP)
    is_pinky_extended = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP,
                                           mp_hands.HandLandmark.PINKY_PIP)

    return is_ok_distance and is_middle_extended and is_ring_extended and is_pinky_extended


def is_fist_gesture(hand_landmarks):
    """Распознавание жеста 'Сжатый кулак'."""
    is_fist = True
    for finger_tip, finger_pip in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        if is_finger_extended(hand_landmarks, finger_tip, finger_pip):
            is_fist = False
            break
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    is_thumb_down = thumb_tip.y > thumb_ip.y
    return is_fist and is_thumb_down


def is_like_gesture(hand_landmarks):
    """
    Распознавание жеста 'Лайк' - поднятый только большой палец.

    This function incorporates portions of the gesture detection logic from the gesture-detection-emojis project.
    Original source: https://github.com/bdekraker/gesture-detection-emojis/blob/main/face.py
    Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
    """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.y < wrist.y and index_finger_tip.y > thumb_tip.y


def is_rock_n_roll_gesture(hand_landmarks):
    """Распознавание жеста 'Рок-н-ролл' независимо от ориентации руки."""
    fingers = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False
    }

    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        fingers["index"] = True
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.PINKY_PIP].y:
        fingers["pinky"] = True

    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y + 0.02:
        fingers["middle"] = False
    else:
        fingers["middle"] = True

    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_PIP].y + 0.02:
        fingers["ring"] = False
    else:
        fingers["ring"] = True

    return fingers["index"] and fingers["pinky"] and not fingers["middle"] and not fingers["ring"]


# Функции для распознавания выражений лица
def detect_smile(face_landmarks, scale, smile_threshold=0.00035):
    """Определяет, улыбается ли лицо на основе ключевых точек лица."""
    UPPER_LIP = 13
    LOWER_LIP = 17

    upper_lip = face_landmarks.landmark[UPPER_LIP]
    lower_lip = face_landmarks.landmark[LOWER_LIP]

    lip_distance = lower_lip.y - upper_lip.y
    normalized_lip_distance = lip_distance / scale

    # print('УЛЫБКА: ', normalized_lip_distance)

    return normalized_lip_distance > smile_threshold, normalized_lip_distance


def detect_raised_eyebrows(face_landmarks, scale, eyebrow_threshold=0.00027):
    """Определяет, подняты ли брови на основе ключевых точек лица с учетом масштаба."""
    LEFT_EYEBROW = 70
    RIGHT_EYEBROW = 300
    LEFT_EYE = 159
    RIGHT_EYE = 386

    left_eyebrow = face_landmarks.landmark[LEFT_EYEBROW]
    right_eyebrow = face_landmarks.landmark[RIGHT_EYEBROW]
    left_eye = face_landmarks.landmark[LEFT_EYE]
    right_eye = face_landmarks.landmark[RIGHT_EYE]

    left_diff = left_eye.y - left_eyebrow.y
    right_diff = right_eye.y - right_eyebrow.y

    normalized_left_diff = left_diff / scale
    normalized_right_diff = right_diff / scale

    # print('БРОВИ:', 'left -', normalized_left_diff, 'right -', normalized_right_diff)

    return normalized_left_diff > eyebrow_threshold and normalized_right_diff > eyebrow_threshold


def detect_puckered_lips(face_landmarks, image_width, image_height, puckered_threshold=0.145):
    """
    Определяет, сформированы ли губы в форме "уточки" (duck lips) при условии,
    что человек выполняет действие, похожее на поцелуй.
    """
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    UPPER_LIP_LEFT = 13
    UPPER_LIP_RIGHT = 14
    LOWER_LIP_LEFT = 14
    LOWER_LIP_RIGHT = 17

    left_corner = face_landmarks.landmark[LEFT_MOUTH_CORNER]
    right_corner = face_landmarks.landmark[RIGHT_MOUTH_CORNER]
    upper_lip_left = face_landmarks.landmark[UPPER_LIP_LEFT]
    upper_lip_right = face_landmarks.landmark[UPPER_LIP_RIGHT]
    lower_lip_left = face_landmarks.landmark[LOWER_LIP_LEFT]
    lower_lip_right = face_landmarks.landmark[LOWER_LIP_RIGHT]

    center_x = (left_corner.x + right_corner.x) / 2
    center_y = (upper_lip_left.y + upper_lip_right.y + lower_lip_left.y + lower_lip_right.y) / 4

    dist_upper_left = calculate_normalized_distance(center_x, center_y, upper_lip_left.x, upper_lip_left.y, image_width,
                                                    image_height)
    dist_upper_right = calculate_normalized_distance(center_x, center_y, upper_lip_right.x, upper_lip_right.y,
                                                     image_width,
                                                     image_height)
    dist_lower_left = calculate_normalized_distance(center_x, center_y, lower_lip_left.x, lower_lip_left.y, image_width,
                                                    image_height)
    dist_lower_right = calculate_normalized_distance(center_x, center_y, lower_lip_right.x, lower_lip_right.y,
                                                     image_width,
                                                     image_height)

    avg_vertical_dist = (dist_upper_left + dist_upper_right + dist_lower_left + dist_lower_right) / 4
    horizontal_dist = calculate_normalized_distance(left_corner.x, left_corner.y, right_corner.x, right_corner.y,
                                                    image_width,
                                                    image_height)

    aspect_ratio = avg_vertical_dist / horizontal_dist if horizontal_dist != 0 else 0

    # print(f"УТОЧКА: {aspect_ratio:.4f}")

    if aspect_ratio > puckered_threshold:
        return True
    else:
        return False


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

snapshot_taken = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        original_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)

        smile_detected = False
        closed_smile_detected = False
        puckered_lips_detected = False
        raised_eyebrows_detected = False
        open_mouth = False

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                scale = calculate_scale(face_landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)
                if scale == 0:
                    scale = 1

                smile_detected, normalized_lip_distance = detect_smile(face_landmarks, scale)

                if detect_puckered_lips(face_landmarks, IMAGE_WIDTH,
                                        IMAGE_HEIGHT) and normalized_lip_distance < 0.00056:
                    puckered_lips_detected = True

                if detect_raised_eyebrows(face_landmarks, scale):
                    raised_eyebrows_detected = True

                if puckered_lips_detected:
                    smile_detected = False

                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                h, w, _ = frame.shape
                xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
                ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

                # Определяем цвет прямоугольника на основе обнаруженных выражений
                if (smile_detected or closed_smile_detected or puckered_lips_detected or
                        raised_eyebrows_detected or open_mouth):
                    color = (0, 0, 255)  # Красный цвет для обнаруженных искажений
                else:
                    color = (255, 0, 0)  # Синий цвет для нормального лица

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )

        hand_results = hands.process(rgb)
        gesture_detected = False
        detected_gesture = None

        if hand_results.multi_hand_landmarks:
            hand_data = list(zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness))
            for idx in range(len(hand_data)):
                hand_landmarks, handedness = hand_data[idx]

                if is_v_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "V Gesture"
                elif is_ok_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "OK Gesture"
                elif is_fist_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Fist"
                elif is_stop_signal(hand_landmarks) and not is_hand_reversed(hand_landmarks, handedness) \
                        and are_fingers_visible(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Stop Signal"
                elif is_rock_n_roll_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Rock-n-Roll Gesture"
                elif is_like_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Like"

                if gesture_detected:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )
                else:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )

        # Условие для сохранения снимка: обнаружено любое выражение лица или один из жестов
        if (smile_detected or closed_smile_detected or puckered_lips_detected or
                raised_eyebrows_detected or open_mouth or gesture_detected):
            if not snapshot_taken:
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{SESSION_NAME}/snapshot_{timestamp}.png"
                    cv2.imwrite(filename, original_frame)
                    print(f"Снимок сохранён как {filename}")
                    snapshot_taken = True
                except Exception as e:
                    print(f"Ошибка при сохранении снимка: {e}")
        else:
            snapshot_taken = False

        # Добавляем текст с названием жеста (если обнаружен)
        if gesture_detected and detected_gesture:
            cv2.putText(frame, detected_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        # Добавление текста для выражений лица
        y_position = 150
        if smile_detected:
            cv2.putText(frame, "Smile Detected", (50, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            y_position += 40
        if closed_smile_detected:
            cv2.putText(frame, "Closed Smile Detected", (50, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2, cv2.LINE_AA)
            y_position += 40
        if puckered_lips_detected:
            cv2.putText(frame, "Puckered Lips", (50, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 165, 255), 2, cv2.LINE_AA)
            y_position += 40
        if raised_eyebrows_detected:
            cv2.putText(frame, "Raised Eyebrows", (50, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2, cv2.LINE_AA)
            y_position += 40
        if open_mouth:
            cv2.putText(frame, "Open Mouth", (50, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 165, 255), 2, cv2.LINE_AA)
            y_position += 40

        cv2.imshow('PhotoBooth', frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Нажата клавиша 'q'. Выход из программы.")
            break

except Exception as e:
    print(f"Произошла ошибка: {e}")

finally:
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    print("Ресурсы освобождены")

    # Удаление размытых фотографий
    # delete_blurred_images(SESSION_NAME, threshold=50)
    # print("Размытые фотографии удалены")
    print("Программа завершена")
