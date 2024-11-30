import cv2
import datetime
import os
from delete_blured import delete_blurred_images
import pygame
from functions import *

pygame.mixer.init()


def play_background_music():
    pygame.mixer.music.load('media/background_music.mp3')
    pygame.mixer.music.set_volume(0.2)
    pygame.mixer.music.play(-1)


def play_image_save_sound():
    image_save_sound = pygame.mixer.Sound('media/image_save_sound.wav')
    image_save_sound.play()


def play_sigma_sound():
    sigma_sound = pygame.mixer.Sound('media/sigma_sound.wav')
    sigma_sound.play()


if not os.path.exists("sessions"):
    os.mkdir("sessions")
SESSION_NAME = f'sessions/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.mkdir(SESSION_NAME)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=6,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=10,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

snapshot_taken = False
count_frames = 0
detected_emotions = []
same_detect_count = 0
no_sigma_frames = 10

play_background_music()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        original_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        count_frames += 1

        if count_frames < 180:
            cv2.putText(frame, "Show some emotion and freeze!", (140, 270), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, "        Press 'q' to exit", (140, 345), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 4, cv2.LINE_AA)

        smile_detected = False
        puckered_lips_detected = False
        raised_eyebrows_detected = False
        detected_emotions_new = []

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
                face_text = ""
                if (smile_detected or puckered_lips_detected or
                        raised_eyebrows_detected):
                    color = (0, 0, 255)  # Красный цвет для обнаруженных эмоций
                    if smile_detected:
                        if face_text:
                            face_text += '/'
                        face_text += "Smile"
                        detected_emotions_new.append("Smile")
                    if puckered_lips_detected:
                        if face_text:
                            face_text += '/'
                        face_text += "Puckered Lips"
                        detected_emotions_new.append("Puckered Lips")
                    if raised_eyebrows_detected:
                        if face_text:
                            face_text += '/'
                        face_text += "Raised Eyebrows"
                        detected_emotions_new.append("Raised Eyebrows")

                    cv2.putText(frame, face_text, (xmin, ymax + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    color = (0, 255, 0)  # Синий цвет для нормального лица

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                # mp_draw.draw_landmarks(
                #     frame,
                #     face_landmarks,
                #     mp_face_mesh.FACEMESH_TESSELATION,
                #     mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                #     mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                # )

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
                elif is_thumb_up(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Thumb up"
                elif is_hear_gesture(hand_landmarks):
                    gesture_detected = True
                    detected_gesture = "Heart"

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

                # Добавление текстовых меток для жестов
                if gesture_detected:
                    x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0]) + 15
                    cv2.putText(frame, detected_gesture, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    detected_emotions_new.append(detected_gesture)

        if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    scale = calculate_scale(face_landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)
                    if scale == 0:
                        scale = 1
                    if SIGMA_DETECTOR(hand_landmarks, face_landmarks, scale):
                        gesture_detected = True
                        detected_gesture = "SIGMA! +RESPECT"
                        if no_sigma_frames >= 10:
                            play_sigma_sound()
                        cv2.putText(frame, detected_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )
                        color = (0, 0, 255)
                        x_coords = [lm.x for lm in face_landmarks.landmark]
                        y_coords = [lm.y for lm in face_landmarks.landmark]
                        h, w, _ = frame.shape
                        xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
                        ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        detected_emotions_new.append(detected_gesture)
                        no_sigma_frames = 0
        if detected_gesture != "SIGMA! +RESPECT":
            no_sigma_frames += 1

        cv2.imshow('PhotoBooth', frame)

        if sorted(detected_emotions) == sorted(detected_emotions_new) and detected_emotions:
            same_detect_count += 1
        else:
            same_detect_count = 0
            detected_emotions = detected_emotions_new

        # Условие для сохранения снимка: обнаружено любое выражение лица или один из жестов
        if same_detect_count == 15 or (same_detect_count % 35 == 0 and same_detect_count != 0):
            if not snapshot_taken:
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{SESSION_NAME}/snapshot_{timestamp}.png"
                    cv2.imwrite(filename, original_frame)
                    print(f"Снимок сохранён как {filename}")
                    snapshot_taken = True
                    play_image_save_sound()
                except Exception as e:
                    print(f"Ошибка при сохранении снимка: {e}")
        else:
            snapshot_taken = False

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
