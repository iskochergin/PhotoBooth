import cv2
import os


def is_blurred(image_path, threshold=100):
    """Проверка того, размыта ли фотография. Основано на Лапласовой дисперсии"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < threshold


def delete_blurred_images(directory, threshold=100):
    """Проход по всем файлам в директории и удаление размытых"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if is_blurred(file_path, threshold):
            os.remove(file_path)
            print(f"Deleted blurred image: {filename}")
        else:
            print(f"Image is clear: {filename}")
