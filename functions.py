import math
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands


def calculate_distance_xyz(point1, point2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return math.sqrt((point1.x - point2.x) ** 2 +
                     (point1.y - point2.y) ** 2 +
                     (point1.z - point2.z) ** 2)


def calculate_distance_xy(point1, point2):
    """Вычисляет расстояние между двумя точками на плоскости"""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


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


def calculate_angle_2points(landmark1, landmark2):
    """Вычисление угла между двумя точками"""
    return np.arctan2(landmark2.y - landmark1.y, landmark2.x - landmark1.x) * 180 / np.pi


def calculate_angle_3points(point1, point2, point3):
    """
    Рассчитывает угол между тремя точками.
    point1, point2, point3 - это объекты с координатами (x, y, z).
    """
    vector1 = [point1.x - point2.x, point1.y - point2.y, point1.z - point2.z]
    vector2 = [point3.x - point2.x, point3.y - point2.y, point3.z - point2.z]

    dot_product = sum([vector1[i] * vector2[i] for i in range(3)])
    norm1 = math.sqrt(sum([vector1[i] ** 2 for i in range(3)]))
    norm2 = math.sqrt(sum([vector2[i] ** 2 for i in range(3)]))

    # Косинус угла
    cos_angle = dot_product / (norm1 * norm2)
    # Ограничиваем значение cos_angle для избежания ошибок из-за погрешностей в вычислениях
    cos_angle = max(min(cos_angle, 1), -1)
    # Рассчитываем угол в радианах и переводим в градусы
    angle = math.degrees(math.acos(cos_angle))
    return angle


def is_hand_reversed(hand_landmarks, handedness):
    """Проверка того, что ладонь перевернута, этого надо избежать при определении жеста СТОП"""
    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

    angle_thumb = calculate_angle_2points(wrist, thumb_tip)
    angle_index = calculate_angle_2points(wrist, index_tip)

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
    distance = calculate_distance_xyz(thumb_tip, index_tip)

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
    landmarks = hand_landmarks.landmark

    finger_tips = {
        "thumb_tip": landmarks[mp_hands.HandLandmark.THUMB_TIP],
        "index_finger_tip": landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        "middle_finger_tip": landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        "ring_finger_tip": landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
        "pinky_tip": landmarks[mp_hands.HandLandmark.PINKY_TIP],
    }

    finger_joints = {
        "thumb_joint": landmarks[mp_hands.HandLandmark.THUMB_CMC],
        "index_finger_joint": landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        "middle_finger_joint": landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        "ring_finger_joint": landmarks[mp_hands.HandLandmark.RING_FINGER_MCP],
        "pinky_joint": landmarks[mp_hands.HandLandmark.PINKY_MCP],
    }
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_angle = calculate_angle_3points(finger_joints["thumb_joint"], finger_tips["thumb_tip"], wrist)

    is_fist = True
    for tip, joint in zip(finger_tips.keys(), finger_joints.keys()):
        if tip == "thumb_tip":
            continue
        if finger_tips[tip].y < finger_joints[joint].y:
            is_fist = False
            break

    # print('thumb_angle:', thumb_angle)
    return is_fist and thumb_angle > 12


def is_hear_gesture(hand_landmarks):
    """Распознавание жеста 'Сердце'."""

    landmarks = hand_landmarks.landmark

    finger_tips = {
        "thumb_tip": landmarks[mp_hands.HandLandmark.THUMB_TIP],
        "index_finger_tip": landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        "middle_finger_tip": landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        "ring_finger_tip": landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
        "pinky_tip": landmarks[mp_hands.HandLandmark.PINKY_TIP],
    }

    finger_joints = {
        "thumb_joint": landmarks[mp_hands.HandLandmark.THUMB_CMC],
        "index_finger_joint": landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        "middle_finger_joint": landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        "ring_finger_joint": landmarks[mp_hands.HandLandmark.RING_FINGER_MCP],
        "pinky_joint": landmarks[mp_hands.HandLandmark.PINKY_MCP],
    }
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_angle = calculate_angle_3points(finger_joints["thumb_joint"], finger_tips["thumb_tip"], wrist)

    is_fist = True
    com_dif = []
    for tip, joint in zip(finger_tips.keys(), finger_joints.keys()):
        if tip == "thumb_tip":
            continue
        if finger_tips[tip].y + 0.2 < finger_joints[joint].y:
            is_fist = False
            break
        # print(finger_tips[tip].y - finger_joints[joint].y, end=' ')
        com_dif.append(finger_tips[tip].y - finger_joints[joint].y)
    for a in com_dif:
        for b in com_dif:
            if abs(a - b) > 0.05:
                is_fist = False
                break

    return is_fist and 6 < thumb_angle < 12 and not is_fist_gesture(hand_landmarks) and not is_stop_signal(hand_landmarks)


def is_thumb_up(hand_landmarks):
    """
    Распознавание жеста 'Лайк' - поднятый только большой палец.

    This function incorporates portions of the gesture detection logic from the gesture-detection-emojis project.
    Original source: https://github.com/bdekraker/gesture-detection-emojis/blob/main/face.py
    Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
    """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

    if thumb_tip.y < wrist.y and index_finger_tip.y > thumb_tip.y:
        angle = calculate_angle_3points(thumb_mcp, thumb_tip, wrist)
        if angle < 4:
            return True
    return False


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def SIGMA_DETECTOR(hand_landmarks, face_landmarks, scale):
    """БЕЗ КОММЕНТАРИЕВ!!!"""

    landmarks = hand_landmarks.landmark
    finger_tips = {
        "thumb_tip": landmarks[mp_hands.HandLandmark.THUMB_TIP],
        "index_finger_tip": landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
    }

    landmarks = face_landmarks.landmark
    mouth_top = landmarks[78]
    mouth_bottom = landmarks[87]

    index_finger_tip = finger_tips["index_finger_tip"]

    mouth_center = Point(
        (mouth_top.x + mouth_bottom.x) / 2,
        (mouth_top.y + mouth_bottom.y) / 2
    )

    distance = calculate_distance_xy(index_finger_tip, mouth_center)

    if distance / scale < 0.0005:
        return True

    return False


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
