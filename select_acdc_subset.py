# select_acdc_subset.py (Версия 2 - отбор по количеству)
# Разместить в корневой папке проекта.
import json
import os
import glob
import random

# --- НАСТРОЙКИ ---
ACDC_ROOT_DIR = 'data/ACDC/'
ACDC_GT_BASE_DIR = os.path.join(ACDC_ROOT_DIR, 'gt_detection')
ACDC_RGB_ANON_BASE_DIR = os.path.join(ACDC_ROOT_DIR, 'rgb_anon') # Нужен для проверки существования файла

TARGET_NUMBER_OF_IMAGES = 800  # Желаемое количество изображений в подмножестве
OUTPUT_SELECTED_SUBSET_FILE = f'selected_acdc_images_{TARGET_NUMBER_OF_IMAGES}imgs.json'

USE_AGGREGATED_GT_FILES = True
# --- КОНЕЦ НАСТРОЕК ---

def get_gt_detection_files():
    """Находит все релевантные *_gt_detection.json файлы для train и val."""
    files_to_scan = []
    if USE_AGGREGATED_GT_FILES:
        agg_train = os.path.join(ACDC_GT_BASE_DIR, "instancesonly_train_gt_detection.json")
        agg_val = os.path.join(ACDC_GT_BASE_DIR, "instancesonly_val_gt_detection.json")
        if os.path.exists(agg_train): files_to_scan.append(agg_train)
        else: print(f"Предупреждение: Агрегированный {agg_train} не найден.")
        
        if os.path.exists(agg_val): files_to_scan.append(agg_val)
        else: print(f"Предупреждение: Агрегированный {agg_val} не найден.")
        
        if files_to_scan:
            print(f"Используются агрегированные gt_detection файлы: {files_to_scan}")
            return files_to_scan
        else:
            print("Агрегированные файлы не найдены, переход к поиску по погодным условиям.")
    
    weather_conditions = ["fog", "night", "rain", "snow"]
    splits = ["train", "val"]
    for weather in weather_conditions:
        weather_path = os.path.join(ACDC_GT_BASE_DIR, weather)
        if not os.path.isdir(weather_path): continue
        for split_type in splits:
            gt_file = os.path.join(weather_path, f"instancesonly_{weather}_{split_type}_gt_detection.json")
            if os.path.exists(gt_file): files_to_scan.append(gt_file)
    print(f"Используются индивидуальные gt_detection файлы ({len(files_to_scan)} шт.): {files_to_scan[:5] + ['...'] if len(files_to_scan) > 5 else files_to_scan}")
    return files_to_scan

def main():
    print(f"Этап 1: Выбор Подмножества из {TARGET_NUMBER_OF_IMAGES} Изображений ACDC...")

    gt_json_files = get_gt_detection_files()
    if not gt_json_files:
        print("Не найдено JSON файлов аннотаций для обработки. Прерывание.")
        return

    # Используем словарь для хранения уникальных кандидатов по file_name_relative,
    # чтобы избежать дубликатов, если изображение упоминается в нескольких JSON.
    unique_image_candidates_map = {} 
    print("Сбор информации об изображениях-кандидатах...")

    for json_path in gt_json_files:
        print(f"  Сканирование {os.path.basename(json_path)}...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            images_info = data.get('images', [])
            if not images_info:
                print(f"    В файле {os.path.basename(json_path)} не найдена секция 'images' или она пуста.")
                continue

            for img_entry in images_info:
                image_id = img_entry.get('id') # ID изображения внутри этого JSON файла
                relative_file_name = img_entry.get('file_name') # Относительный путь, как в JSON
                
                if image_id is None or relative_file_name is None:
                    continue

                # Проверяем, существует ли файл на диске, прежде чем добавить в кандидаты
                full_disk_path = os.path.join(ACDC_RGB_ANON_BASE_DIR, relative_file_name)
                if not os.path.exists(full_disk_path):
                    # print(f"    Файл изображения {full_disk_path} не найден. Пропуск.")
                    continue
                
                # Если такого file_name еще нет, добавляем
                if relative_file_name not in unique_image_candidates_map:
                    unique_image_candidates_map[relative_file_name] = {
                        "image_id_in_source_json": image_id, 
                        "file_name_relative": relative_file_name,
                        "source_json_file_path": json_path 
                    }
        except Exception as e:
            print(f"    Ошибка при обработке файла {json_path}: {e}")
    
    all_unique_image_candidates = list(unique_image_candidates_map.values())

    if not all_unique_image_candidates:
        print("Не найдено ни одного уникального изображения-кандидата. Проверьте пути и содержимое JSON файлов.")
        return

    print(f"Найдено всего уникальных изображений-кандидатов: {len(all_unique_image_candidates)}")

    # Отбор подмножества по количеству
    random.shuffle(all_unique_image_candidates) # Перемешиваем для случайного выбора
    
    if len(all_unique_image_candidates) < TARGET_NUMBER_OF_IMAGES:
        print(f"Предупреждение: Доступно только {len(all_unique_image_candidates)} уникальных изображений, что меньше желаемых {TARGET_NUMBER_OF_IMAGES}.")
        print(f"Будут использованы все доступные изображения.")
        selected_images_info = all_unique_image_candidates
    else:
        selected_images_info = all_unique_image_candidates[:TARGET_NUMBER_OF_IMAGES]
    
    print(f"Выбрано изображений для подмножества: {len(selected_images_info)}")

    # Сохранение списка выбранных изображений
    try:
        with open(OUTPUT_SELECTED_SUBSET_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(selected_images_info, f_out, indent=4)
        print(f"Список выбранных изображений ({len(selected_images_info)} шт.) сохранен в: {OUTPUT_SELECTED_SUBSET_FILE}")
    except Exception as e:
        print(f"Ошибка сохранения списка выбранных изображений: {e}")

if __name__ == '__main__':
    main()