import json
import os
import shutil
import random
import glob
import yaml

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ACDC_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'ACDC')
ACDC_CATEGORIES_FILE = os.path.join(ACDC_DATA_DIR, 'categories.json')
ACDC_GT_BASE_DIR = os.path.join(ACDC_DATA_DIR, 'gt_detection_trainval', 'gt_detection') 
ACDC_RGB_ANON_BASE_DIR = os.path.join(ACDC_DATA_DIR, 'rgb_anon_trainvaltest', 'rgb_anon')

print(PROJECT_ROOT_DIR, ACDC_CATEGORIES_FILE, ACDC_DATA_DIR, ACDC_GT_BASE_DIR, ACDC_RGB_ANON_BASE_DIR)

TARGET_TOTAL_IMAGES = 12000     # Изображений около 1600, но чтобы наверняка выбрать вообще все, что есть, можно указать заведомо большее число, чем предполагаемое количество изображений

TRAIN_RATIO_FINAL_YOLO = 0.8 
VAL_RATIO_FINAL_YOLO = 0.2   
TEST_RATIO_FINAL_YOLO = 0.0  

OUTPUT_YOLO_DATASET_NAME = f'ACDC_YOLO_PreservedSplit_{TARGET_TOTAL_IMAGES}imgs'
OUTPUT_YOLO_DATASET_ROOT_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', OUTPUT_YOLO_DATASET_NAME)

YOLO_CLASS_MAPPING_FROM_NAMES = {
    "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
    "motorcycle": 5, "bicycle": 6, 
}
USE_AGGREGATED_GT_FILES = True

def coco_bbox_to_yolo(x_min, y_min, width, height, img_width, img_height):
    x_center = (x_min + width / 2.0) / img_width
    y_center = (y_min + height / 2.0) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return x_center, y_center, norm_width, norm_height

def build_category_mappings():
    if not os.path.exists(ACDC_CATEGORIES_FILE):
        print(f"Ошибка: Файл категорий {ACDC_CATEGORIES_FILE} не найден!")
        return None, None
    with open(ACDC_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        acdc_categories_data = json.load(f)
    acdc_id_to_yolo_id = {}
    max_yolo_id = -1
    if YOLO_CLASS_MAPPING_FROM_NAMES:
        max_yolo_id = max(YOLO_CLASS_MAPPING_FROM_NAMES.values())
    final_yolo_class_names_ordered = [""] * (max_yolo_id + 1) 
    for acdc_cat_info in acdc_categories_data:
        acdc_id = acdc_cat_info['id']
        acdc_name = acdc_cat_info['name']
        if acdc_name in YOLO_CLASS_MAPPING_FROM_NAMES:
            yolo_id = YOLO_CLASS_MAPPING_FROM_NAMES[acdc_name]
            acdc_id_to_yolo_id[acdc_id] = yolo_id
            if 0 <= yolo_id < len(final_yolo_class_names_ordered):
                final_yolo_class_names_ordered[yolo_id] = acdc_name
            else: 
                print(f"Критическая ошибка: YOLO ID {yolo_id} для '{acdc_name}' вне диапазона.")
    for i, name in enumerate(final_yolo_class_names_ordered):
        is_id_supposed_to_be_used = False
        for mapped_yolo_id in YOLO_CLASS_MAPPING_FROM_NAMES.values():
            if i == mapped_yolo_id:
                is_id_supposed_to_be_used = True
                break
        if not name and is_id_supposed_to_be_used:
            expected_acdc_name_for_id = "[неизвестно]"
            for acdc_name_key, yolo_id_val in YOLO_CLASS_MAPPING_FROM_NAMES.items():
                if yolo_id_val == i:
                    expected_acdc_name_for_id = acdc_name_key
                    break
            print(f"Ошибка: Для YOLO ID {i} (ожидалось для ACDC '{expected_acdc_name_for_id}') не найдено.")
            return None, None
    final_yolo_class_names_ordered = [name for name in final_yolo_class_names_ordered if name]
    if not final_yolo_class_names_ordered or len(final_yolo_class_names_ordered) != len(YOLO_CLASS_MAPPING_FROM_NAMES):
        print("Ошибка: Количество итоговых классов YOLO не совпадает с ожидаемым из YOLO_CLASS_MAPPING_FROM_NAMES.")
        return None, None
    print("Успешно построены отображения категорий.")
    print(f"  ACDC ID -> YOLO ID: {acdc_id_to_yolo_id}")
    print(f"  Имена классов YOLO (для dataset.yaml): {final_yolo_class_names_ordered}")
    return acdc_id_to_yolo_id, final_yolo_class_names_ordered

def get_source_gt_detection_files_with_split_info():
    source_files_info = [] # Список словарей {'path': '...', 'original_split': 'train'/'val'}
    
    if USE_AGGREGATED_GT_FILES:
        agg_train_path = os.path.join(ACDC_GT_BASE_DIR, "instancesonly_train_gt_detection.json")
        agg_val_path = os.path.join(ACDC_GT_BASE_DIR, "instancesonly_val_gt_detection.json")
        if os.path.exists(agg_train_path):
            source_files_info.append({'path': agg_train_path, 'original_split': 'train'})
        if os.path.exists(agg_val_path):
            source_files_info.append({'path': agg_val_path, 'original_split': 'val'})
        
        if len(source_files_info) == 2: # Если оба агрегированных найдены
            print(f"Используются агрегированные gt_detection файлы: {[info['path'] for info in source_files_info]}")
            return source_files_info
        elif source_files_info: # Если найден только один, это странно, но продолжим с ним и индивидуальными
            print(f"Найден только один агрегированный файл: {[info['path'] for info in source_files_info]}. Поиск индивидуальных...")
        else: # Ни одного агрегированного
            print("Агрегированные файлы не найдены. Поиск индивидуальных файлов...")

    weather_conditions = ["fog", "night", "rain", "snow"]
    acdc_splits = ["train", "val"]
    for weather in weather_conditions:
        weather_path = os.path.join(ACDC_GT_BASE_DIR, weather)
        if not os.path.isdir(weather_path): continue
        for split_type in acdc_splits:
            gt_file = os.path.join(weather_path, f"instancesonly_{weather}_{split_type}_gt_detection.json")
            if os.path.exists(gt_file):
                is_already_covered_by_aggregated = False
                if USE_AGGREGATED_GT_FILES:
                    agg_file_for_split = os.path.join(ACDC_GT_BASE_DIR, f"instancesonly_{split_type}_gt_detection.json")
                    if any(info['path'] == agg_file_for_split for info in source_files_info):
                        is_already_covered_by_aggregated = True
                
                if not is_already_covered_by_aggregated:
                    source_files_info.append({'path': gt_file, 'original_split': split_type})
    
    unique_source_files_info = []
    seen_paths = set()
    for info in source_files_info:
        if info['path'] not in seen_paths:
            unique_source_files_info.append(info)
            seen_paths.add(info['path'])
            
    if unique_source_files_info:
        print(f"Файлы JSON для сканирования ({len(unique_source_files_info)}):")
        for info in unique_source_files_info: print(f" - {info['path']} (исходный сплит: {info['original_split']})")
    return unique_source_files_info


def collect_all_image_data(source_files_info_list):
    all_image_entries = [] # {file_name, width, height, source_image_path, annotations, original_acdc_split}
    unique_file_names = set()
    print("\nСбор данных из исходных JSON файлов ACDC...")
    for source_file_entry in source_files_info_list:
        json_path = source_file_entry['path']
        original_split = source_file_entry['original_split']
        print(f"  Обработка JSON: {os.path.basename(json_path)} (исходный ACDC сплит: {original_split})")
        try:
            with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
            images_in_json = {img['id']: img for img in data.get('images', [])}
            annotations_by_img_id = {}
            for ann in data.get('annotations', []):
                img_id = ann['image_id']
                annotations_by_img_id.setdefault(img_id, []).append(ann)

            for img_id, img_details in images_in_json.items():
                relative_file_name = img_details['file_name']
                if relative_file_name in unique_file_names: continue # Уже обработали это изображение из другого JSON
                
                full_disk_path = os.path.join(ACDC_RGB_ANON_BASE_DIR, relative_file_name)
                if not os.path.exists(full_disk_path): continue
                
                all_image_entries.append({
                    "file_name": relative_file_name, "width": img_details['width'], "height": img_details['height'],
                    "source_image_path": full_disk_path, 
                    "annotations": annotations_by_img_id.get(img_id, []),
                    "original_acdc_split": original_split 
                })
                unique_file_names.add(relative_file_name)
        except Exception as e: print(f"    Ошибка при обработке {json_path}: {e}")
    print(f"Собрана информация для {len(all_image_entries)} уникальных изображений.")
    return all_image_entries


def save_yolo_split(split_name, image_data_list, acdc_id_to_yolo_id_map):
    output_imgs_dir = os.path.join(OUTPUT_YOLO_DATASET_ROOT_DIR, 'images', split_name)
    output_labels_dir = os.path.join(OUTPUT_YOLO_DATASET_ROOT_DIR, 'labels', split_name)
    os.makedirs(output_imgs_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    copied_images_count = 0
    written_annotations_count = 0
    print(f"\n--- Сохранение выборки: {split_name} ({len(image_data_list)} изображений) ---")

    for img_entry in image_data_list:
        relative_path = img_entry['file_name']
        source_img_path = img_entry['source_image_path']
        img_w, img_h = img_entry['width'], img_entry['height']
        annotations = img_entry['annotations']

        yolo_flat_img_filename = relative_path.replace('/', '_').replace('\\', '_')
        yolo_flat_label_filename = os.path.splitext(yolo_flat_img_filename)[0] + '.txt'
        dest_img_path = os.path.join(output_imgs_dir, yolo_flat_img_filename)
        dest_label_path = os.path.join(output_labels_dir, yolo_flat_label_filename)

        try:
            shutil.copy(source_img_path, dest_img_path)
            copied_images_count += 1
        except Exception as e:
            print(f"    Ошибка копирования {source_img_path} -> {dest_img_path}: {e}")
            continue

        with open(dest_label_path, 'w', encoding='utf-8') as f_label:
            for ann in annotations:
                acdc_cat_id = ann.get('category_id')
                coco_bbox = ann.get('bbox')
                if acdc_cat_id is None or coco_bbox is None or acdc_cat_id not in acdc_id_to_yolo_id_map:
                    continue
                yolo_class_id = acdc_id_to_yolo_id_map[acdc_cat_id]
                x_min, y_min, bbox_w, bbox_h = coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3]
                if bbox_w <= 0 or bbox_h <= 0: continue
                yolo_x, yolo_y, yolo_w_n, yolo_h_n = coco_bbox_to_yolo(x_min, y_min, bbox_w, bbox_h, img_w, img_h)
                f_label.write(f"{yolo_class_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w_n:.6f} {yolo_h_n:.6f}\n")
                written_annotations_count += 1
    print(f"  Для {split_name}: скопировано изображений: {copied_images_count}, записано аннотаций: {written_annotations_count}")
    return copied_images_count, written_annotations_count


def main():
    print(f"Запуск подготовки датасета ACDC для YOLO (Версия 6 - с сохранением сплитов)...")
    if os.path.exists(OUTPUT_YOLO_DATASET_ROOT_DIR):
        print(f"Очистка: {OUTPUT_YOLO_DATASET_ROOT_DIR}")
        shutil.rmtree(OUTPUT_YOLO_DATASET_ROOT_DIR)
    
    if abs(TRAIN_RATIO_FINAL_YOLO + VAL_RATIO_FINAL_YOLO + TEST_RATIO_FINAL_YOLO - 1.0) > 1e-9:
        print("Ошибка: Сумма RATIO для YOLO сплитов должна быть 1.0")
        return

    acdc_id_to_yolo_id, final_yolo_class_names = build_category_mappings()
    if not acdc_id_to_yolo_id or not final_yolo_class_names: return

    source_files = get_source_gt_detection_files_with_split_info()
    if not source_files: return

    all_images = collect_all_image_data(source_files)
    if not all_images: return
    
    random.shuffle(all_images)


    if len(all_images) > TARGET_TOTAL_IMAGES:
        print(f"Отобрано {TARGET_TOTAL_IMAGES} из {len(all_images)} доступных уникальных изображений.")
        selected_pool = all_images[:TARGET_TOTAL_IMAGES]
    else:
        print(f"Используются все {len(all_images)} доступные уникальные изображения (меньше или равно {TARGET_TOTAL_IMAGES}).")
        selected_pool = all_images
    
    num_selected_pool = len(selected_pool)

    acdc_train_images = [img for img in selected_pool if img['original_acdc_split'] == 'train']
    acdc_val_images = [img for img in selected_pool if img['original_acdc_split'] == 'val']

    yolo_train_list = acdc_train_images
    yolo_val_list = acdc_val_images
    yolo_test_list = []

    print(f"Отобрано из ACDC train: {len(yolo_train_list)} изображений.")
    print(f"Отобрано из ACDC val: {len(yolo_val_list)} изображений.")
    
    num_for_yolo_test = int(TEST_RATIO_FINAL_YOLO * num_selected_pool)

    if num_for_yolo_test > 0:
        print(f"Требуется {num_for_yolo_test} изображений для YOLO тестовой выборки.")
        temp_pool_for_test = yolo_train_list + yolo_val_list
        random.shuffle(temp_pool_for_test)
        
        if num_for_yolo_test > len(temp_pool_for_test):
            print(f"Предупреждение: Недостаточно изображений ({len(temp_pool_for_test)}) для формирования тестовой выборки из {num_for_yolo_test}. Используются все доступные.")
            yolo_test_list = temp_pool_for_test
            yolo_train_list = [] # Все ушло в тест
            yolo_val_list = []
        else:
            yolo_test_list = temp_pool_for_test[:num_for_yolo_test]
            test_file_names = {img['file_name'] for img in yolo_test_list}
            yolo_train_list = [img for img in acdc_train_images if img['file_name'] not in test_file_names]
            yolo_val_list = [img for img in acdc_val_images if img['file_name'] not in test_file_names]
        print(f"Сформирована YOLO тестовая выборка: {len(yolo_test_list)} изображений.")

    print(f"Итоговое распределение для YOLO: Train={len(yolo_train_list)}, Val={len(yolo_val_list)}, Test={len(yolo_test_list)}")

    total_imgs = 0
    total_anns = 0
    
    img_c, ann_c = save_yolo_split("train", yolo_train_list, acdc_id_to_yolo_id)
    total_imgs += img_c; total_anns += ann_c
    
    img_c, ann_c = save_yolo_split("val", yolo_val_list, acdc_id_to_yolo_id)
    total_imgs += img_c; total_anns += ann_c

    if TEST_RATIO_FINAL_YOLO > 0.0 and yolo_test_list:
        img_c, ann_c = save_yolo_split("test", yolo_test_list, acdc_id_to_yolo_id)
        total_imgs += img_c; total_anns += ann_c

    dataset_yaml_path = os.path.join(OUTPUT_YOLO_DATASET_ROOT_DIR, 'dataset.yaml')

    # path_in_docker_for_yaml = r'/datasets/acdc_final' 
    path_in_docker_for_yaml = './acdc_final'
    # path_in_docker_for_yaml = os.path.join('/datasets', OUTPUT_YOLO_DATASET_NAME) # Путь как его увидит Docker

    yaml_content = {
        'path': path_in_docker_for_yaml, 
        'train': os.path.join('images', 'train'), 
        'val': os.path.join('images', 'val'),     
        'nc': len(final_yolo_class_names),
        'names': final_yolo_class_names
    }
    if TEST_RATIO_FINAL_YOLO > 0.0 and yolo_test_list:
        yaml_content['test'] = os.path.join('images', 'test')
    try:
        with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True, Dumper=yaml.CDumper if hasattr(yaml, 'CDumper') else yaml.Dumper)
        print(f"\nФайл dataset.yaml успешно создан: {dataset_yaml_path}")
        print(f"  (Убедитесь, что 'path' в нем: '{path_in_docker_for_yaml}' корректен для Docker)")
    except Exception as e: print(f"\nОшибка при создании dataset.yaml: {e}")

    print(f"\n--- Подготовка датасета завершена ---")
    print(f"Всего изображений в итоговом датасете YOLO: {total_imgs}")
    print(f"Всего аннотаций в итоговом датасете YOLO: {total_anns}")
    print(f"Структура создана в: {OUTPUT_YOLO_DATASET_ROOT_DIR}")

if __name__ == '__main__':
    main()