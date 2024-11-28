import cv2
import datetime
import mediapipe as mp
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
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


# Функции для распознавания жестов
def calculate_distance(point1, point2):
    """Вычисляет евклидово расстояние между двумя точками."""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)


def is_v_gesture(hand_landmarks):
    """Распознавание жеста 'V' или 'Победа'."""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    is_index_up = index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    is_middle_up = middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    is_ring_down = ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
    is_pinky_down = pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y

    return is_index_up and is_middle_up and is_ring_down and is_pinky_down


def is_ok_gesture(hand_landmarks):
    """Распознавание жеста 'ОК'."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    distance = calculate_distance(thumb_tip, index_tip)

    # Пороговое значение (настраивается)
    return distance < 0.05


def is_fist_gesture(hand_landmarks):
    """Распознавание жеста 'Сжатый кулак'."""
    for finger_tip, finger_pip in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        if hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_pip].y:
            return False
    # Проверяем большой палец
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    return thumb_tip.y > thumb_ip.y


def is_shushing_gesture(hand_landmarks):
    """Распознавание жеста 'Жест молчания'."""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    return index_tip.y < wrist.y + 0.1 and index_tip.y > wrist.y - 0.1


def is_stop_signal(hand_landmarks):
    """Распознавание жеста 'Стоп-сигнал'."""
    fingers = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False
    }

    # Проверяем, подняты ли все пальцы
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.THUMB_IP].y:
        fingers["thumb"] = True
    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        fingers["index"] = True
    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
        fingers["middle"] = True
    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_PIP].y:
        fingers["ring"] = True
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.PINKY_PIP].y:
        fingers["pinky"] = True

    return all(fingers.values())


def is_greeting_gesture(hand_landmarks):
    """Распознавание жеста 'Приветственный жест'."""
    # Рука поднята с легким помахом
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    return middle_tip.y < wrist.y - 0.1


def is_rock_n_roll_gesture(hand_landmarks):
    """Распознавание жеста 'Рок-н-ролл'."""
    fingers = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False
    }

    # Проверяем, подняты ли указательный и мизинец
    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        fingers["index"] = True
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.PINKY_PIP].y:
        fingers["pinky"] = True

    # Проверяем, что средний и безымянный пальцы согнуты
    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
        fingers["middle"] = False
    else:
        fingers["middle"] = True

    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_PIP].y:
        fingers["ring"] = False
    else:
        fingers["ring"] = True

    return fingers["index"] and fingers["pinky"] and not fingers["middle"] and not fingers["ring"]


def detect_smile(face_landmarks):
    """Определяет, улыбается ли лицо на основе ключевых точек лица."""
    # Используем ключевые точки на губах
    upper_lip = face_landmarks.landmark[61]
    lower_lip = face_landmarks.landmark[17]

    # Вычисляем расстояние между верхней и нижней губами
    lip_distance = lower_lip.y - upper_lip.y

    # Определяем пороговое значение для улыбки
    # Порог можно настроить в зависимости от камеры и условий освещения
    return lip_distance > 0.05  # Примерное значение, может потребовать настройки


# Инициализация видеопотока
cap = cv2.VideoCapture(0)

# Установка разрешения камеры (опционально)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Флаг для предотвращения многократного сохранения снимков
snapshot_taken = False

# Инициализация счетчика для подтверждения поцелуя
KISS_CONSECUTIVE_FRAMES = 3  # Количество последовательных кадров для подтверждения
kiss_counter = 0
kiss_detected_final = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        # Создаём копию оригинального кадра для сохранения без аннотаций
        original_frame = frame.copy()

        # Конвертируем изображение в RGB для MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обработка лиц
        face_results = face_mesh.process(rgb)

        smile_detected = False
        current_kiss_detected = False

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Определение улыбки
                if detect_smile(face_landmarks):
                    smile_detected = True
                    print("Улыбка обнаружена.")

                # Вычисляем и рисуем ограничивающий прямоугольник вокруг лица
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                h, w, _ = frame.shape
                xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
                ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

                if smile_detected or kiss_detected_final:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Красный цвет
                else:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Синий цвет

                # Рисуем сетку лицевых ключевых точек (опционально)
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )

        # Обработка рук
        hand_results = hands.process(rgb)

        gesture_detected = False
        detected_gesture = None  # Название обнаруженного жеста

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Распознавание жестов
                if is_v_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "V Gesture"
                elif is_ok_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "OK Gesture"
                elif is_fist_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Fist"
                elif is_shushing_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Shushing Gesture"
                elif is_stop_signal(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Stop Signal"
                elif is_greeting_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Greeting Gesture"
                elif is_rock_n_roll_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Rock-n-Roll Gesture"

                # Рисуем руки
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

        # Условие для сохранения снимка: обнаружена улыбка, подтвержден поцелуй или один из жестов
        if smile_detected or kiss_detected_final or gesture_detected:
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

        # Добавляем текст с состоянием поцелуя
        if kiss_detected_final:
            cv2.putText(frame, "Kiss Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 255), 2, cv2.LINE_AA)

        # Вывод аннотированного кадра
        cv2.imshow('Фотобудка', frame)

        # Отладочное сообщение
        # print("Кадр обработан.")

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
    delete_blurred_images(SESSION_NAME, threshold=200)
    print("Размытые фотографии удалены")
    print("Программа завершена")
