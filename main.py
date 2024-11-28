import cv2
import datetime
import mediapipe as mp
import numpy as np
import math
import os
from delete_blured import delete_blurred_images

SESSION_NAME = f'session_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.mkdir(SESSION_NAME)

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


def is_finger_extended(hand_landmarks, finger_tip, finger_pip):
    """Проверяет, вытянут ли палец"""
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_pip].y


def calculate_angle(landmark1, landmark2):
    """Вычисление угла между двумя точками"""
    return np.arctan2(landmark2.y - landmark1.y, landmark2.x - landmark1.x) * 180 / np.pi


def is_stop_signal_reversed(hand_landmarks, handedness):
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


def is_shushing_gesture(hand_landmarks):
    """Распознавание жеста 'Жест молчания'."""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    return wrist.y - 0.05 < index_tip.y < wrist.y + 0.05


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
def detect_smile(face_landmarks):
    """Определяет, улыбается ли лицо на основе ключевых точек лица."""
    upper_lip = face_landmarks.landmark[61]
    lower_lip = face_landmarks.landmark[17]

    lip_distance = lower_lip.y - upper_lip.y

    return lip_distance > 0.035


def detect_puckered_lips(face_landmarks, puckered_threshold=0.02):
    """
    Определяет, сформированы ли губы в форме "уточки".

    :param face_landmarks: Ландмарки лица.
    :param puckered_threshold: Порог для определения "уточки".
    :return: True, если губы сформированы в форме "уточки".
    """
    # Получаем ландмарки
    upper_lip = face_landmarks.landmark[61]
    lower_lip = face_landmarks.landmark[17]
    center_mouth = face_landmarks.landmark[78]  # Центр рта (уточните индекс при необходимости)

    upper_distance = calculate_distance(center_mouth, upper_lip)
    lower_distance = calculate_distance(center_mouth, lower_lip)

    if abs(upper_distance - lower_distance) < puckered_threshold and upper_distance < 0.03:
        return True
    else:
        return False


def detect_raised_eyebrows(face_landmarks):
    """Определяет, подняты ли брови на основе ключевых точек лица."""
    left_eyebrow = face_landmarks.landmark[70]
    right_eyebrow = face_landmarks.landmark[300]
    left_eye = face_landmarks.landmark[159]
    right_eye = face_landmarks.landmark[386]

    left_diff = left_eye.y - left_eyebrow.y
    right_diff = right_eye.y - right_eyebrow.y

    return left_diff > 0.025 and right_diff > 0.025


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
                if detect_smile(face_landmarks):
                    smile_detected = True
                    # print("Улыбка обнаружена.")

                # Определение уточки
                if detect_puckered_lips(face_landmarks):
                    puckered_lips_detected = True
                    # print("Уточка обнаружена.")

                if detect_raised_eyebrows(face_landmarks):
                    raised_eyebrows_detected = True
                    # print("Брови подняты.")

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
                elif is_stop_signal(hand_landmarks) and not is_stop_signal_reversed(hand_landmarks, handedness) \
                        and are_fingers_visible(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Stop Signal"
                elif is_rock_n_roll_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Rock-n-Roll Gesture"

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
