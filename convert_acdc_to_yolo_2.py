# convert_acdc_to_yolo.py
import json
import os
import shutil
import random
import yaml # Для генерации dataset.yaml

# --- НАСТРОЙКИ ---
ACDC_ROOT_DIR = 'data/ACDC/' 
ACDC_CATEGORIES_FILE = 'categories.json'                                                                # Убедись, что путь правильный
ACDC_RGB_ANON_BASE_DIR = os.path.join(ACDC_ROOT_DIR, 'rgb_anon')

SELECTED_IMAGES_SUBSET_FILE = 'selected_acdc_images_800imgs.json' 

try:
    img_count_str = os.path.splitext(SELECTED_IMAGES_SUBSET_FILE)[0].split('imgs')[0].split('_')[-1]
    OUTPUT_YOLO_DATASET_ROOT_DIR = f'data/ACDC_YOLO_Dataset_{img_count_str}imgs/'                       # Имя корневой папки для датасета YOLO
    
except IndexError: 
    print("Предупреждение: не удалось извлечь кол-во изображений. Используется стандартное имя.")
    OUTPUT_YOLO_DATASET_ROOT_DIR = f'data/ACDC_YOLO_Dataset/'

# Пропорции для разделения на train/val/test
# Сумма должна быть равна 1.0. Если test_ratio = 0, то папка test создаваться не будет.
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.0 # Если тестовый набор не нужен на этом этапе

YOLO_CLASS_MAPPING_FROM_NAMES = {
    "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
    "motorcycle": 5, "bicycle": 6, 
}
# --- КОНЕЦ НАСТРОЕК ---

def coco_bbox_to_yolo(x_min, y_min, width, height, img_width, img_height):
    # ... (код без изменений) ...
    x_center = (x_min + width / 2.0)
    y_center = (y_min + height / 2.0)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return x_center_norm, y_center_norm, norm_width, norm_height

def build_category_mappings():
    # ... (код без изменений, как в Версии 4.3) ...
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
            print(f"Ошибка: Для YOLO ID {i} (ожидалось для ACDC класса '{expected_acdc_name_for_id}') не найдено.")
            return None, None
    final_yolo_class_names_ordered = [name for name in final_yolo_class_names_ordered if name]
    print("Успешно построены отображения категорий.")
    print(f"  ACDC ID -> YOLO ID: {acdc_id_to_yolo_id}")
    print(f"  Имена классов YOLO (для dataset.yaml и utils.py): {final_yolo_class_names_ordered}")
    return acdc_id_to_yolo_id, final_yolo_class_names_ordered

def main():
    print(f"Запуск конвертации и разделения ACDC подмножества в формат YOLO (Версия 5)...")
    print(f"Корневая папка для датасета YOLO: {OUTPUT_YOLO_DATASET_ROOT_DIR}")

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        print("Ошибка: Сумма TRAIN_RATIO, VAL_RATIO и TEST_RATIO должна быть равна 1.0")
        return

    acdc_id_to_yolo_id, final_yolo_class_names = build_category_mappings()
    if acdc_id_to_yolo_id is None or final_yolo_class_names is None:
        return

    if not os.path.exists(SELECTED_IMAGES_SUBSET_FILE):
        print(f"Ошибка: Файл {SELECTED_IMAGES_SUBSET_FILE} не найден!")
        return
    
    with open(SELECTED_IMAGES_SUBSET_FILE, 'r', encoding='utf-8') as f_subset:
        selected_images_list_from_json = json.load(f_subset) # Это список словарей

    if not selected_images_list_from_json:
        print("Список выбранных изображений пуст.")
        return

    print(f"Загружено {len(selected_images_list_from_json)} записей о выбранных изображениях из {SELECTED_IMAGES_SUBSET_FILE}")

    # Перемешиваем список выбранных изображений для случайного разделения
    random.shuffle(selected_images_list_from_json)

    # Разделяем список selected_images_list_from_json на train, val, test
    num_total_selected = len(selected_images_list_from_json)
    num_train = int(TRAIN_RATIO * num_total_selected)
    num_val = int(VAL_RATIO * num_total_selected)
    # num_test = num_total_selected - num_train - num_val # Оставшиеся идут в тест

    train_selected_images = selected_images_list_from_json[:num_train]
    val_selected_images = selected_images_list_from_json[num_train : num_train + num_val]
    test_selected_images = selected_images_list_from_json[num_train + num_val :]

    print(f"Разделение на выборки: Train={len(train_selected_images)}, Val={len(val_selected_images)}, Test={len(test_selected_images)}")

    # --- Словарь для хранения путей к созданным .txt файлам и подсчета аннотаций ---
    # Ключ: yolo_flat_label_filename, Значение: полный путь к .txt файлу
    # Это нужно, чтобы не открывать файл на дозапись много раз, если на изображении много объектов,
    # и чтобы отслеживать, для каких изображений уже созданы файлы (для подсчета processed_yolo_images_count)
    created_label_files_paths = {} 
    total_annotations_converted_globally = 0
    
    # --- Обработка каждой выборки (train, val, test) ---
    for split_name, selected_image_infos_for_split in [
        ("train", train_selected_images), 
        ("val", val_selected_images), 
        ("test", test_selected_images)
    ]:
        if not selected_image_infos_for_split: # Если для какого-то сплита 0 файлов
            print(f"Для выборки '{split_name}' нет изображений, пропуск.")
            continue
        
        if split_name == "test" and TEST_RATIO == 0.0: # Не создаем test, если не просили
             print(f"TEST_RATIO = 0, выборка 'test' не будет создана.")
             continue

        output_split_imgs_dir = os.path.join(OUTPUT_YOLO_DATASET_ROOT_DIR, 'images', split_name)
        output_split_labels_dir = os.path.join(OUTPUT_YOLO_DATASET_ROOT_DIR, 'labels', split_name)
        os.makedirs(output_split_imgs_dir, exist_ok=True)
        os.makedirs(output_split_labels_dir, exist_ok=True)
        
        print(f"\n--- Обработка выборки: {split_name} ({len(selected_image_infos_for_split)} изображений) ---")

        # Группируем изображения для текущего сплита по их исходному JSON
        grouped_current_split_images = {}
        for sel_img_info in selected_image_infos_for_split:
            source_json = sel_img_info['source_json_file_path']
            if source_json not in grouped_current_split_images:
                grouped_current_split_images[source_json] = []
            grouped_current_split_images[source_json].append({
                "image_id": sel_img_info['image_id_in_source_json'], 
                "file_name": sel_img_info['file_name_relative']      
            })

        for source_json_path, images_to_process_from_this_json in grouped_current_split_images.items():
            # print(f"  Анализ исходного JSON: {os.path.basename(source_json_path)} для выборки {split_name}")
            normalized_source_json_path = os.path.normpath(source_json_path)
            if not os.path.exists(normalized_source_json_path): continue
            
            try:
                with open(normalized_source_json_path, 'r', encoding='utf-8') as f_gt:
                    gt_data = json.load(f_gt)
            except Exception: continue

            required_image_ids = {img_info['image_id'] for img_info in images_to_process_from_this_json}
            current_json_image_id_to_details = {
                img['id']: {
                    "file_name": img['file_name'], "width": img['width'], "height": img['height']
                } for img in gt_data.get('images', []) if img.get('id') in required_image_ids
            }
            annotations_list = gt_data.get('annotations', [])
            if not annotations_list: continue

            for ann in annotations_list:
                image_id_ann = ann.get('image_id')
                if image_id_ann not in required_image_ids or image_id_ann not in current_json_image_id_to_details:
                    continue

                img_details = current_json_image_id_to_details[image_id_ann]
                img_relative_path = img_details['file_name']
                acdc_cat_id = ann.get('category_id')
                coco_bbox = ann.get('bbox')

                if acdc_cat_id is None or coco_bbox is None or acdc_cat_id not in acdc_id_to_yolo_id:
                    continue
                
                yolo_class_id = acdc_id_to_yolo_id[acdc_cat_id]
                img_w, img_h = img_details['width'], img_details['height']
                x_min, y_min, bbox_w, bbox_h = coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3]

                if bbox_w <= 0 or bbox_h <= 0: continue

                yolo_x, yolo_y, yolo_w_norm, yolo_h_norm = coco_bbox_to_yolo(x_min, y_min, bbox_w, bbox_h, img_w, img_h)
                
                source_img_full_path = os.path.join(ACDC_RGB_ANON_BASE_DIR, img_relative_path)
                yolo_flat_img_filename = img_relative_path.replace('/', '_').replace('\\', '_')
                yolo_flat_label_filename = os.path.splitext(yolo_flat_img_filename)[0] + '.txt'
                
                # Пути для сохранения в текущий сплит
                output_img_path_in_split = os.path.join(output_split_imgs_dir, yolo_flat_img_filename)
                output_label_path_in_split = os.path.join(output_split_labels_dir, yolo_flat_label_filename)

                if yolo_flat_label_filename not in created_label_files_paths: # Изображение обрабатывается первый раз (копируем)
                    if os.path.exists(source_img_full_path):
                        try:
                            shutil.copy(source_img_full_path, output_img_path_in_split)
                            created_label_files_paths[yolo_flat_label_filename] = output_label_path_in_split 
                        except Exception as e:
                            print(f"    Ошибка копирования {source_img_full_path} -> {output_img_path_in_split}: {e}")
                            continue
                    else:
                        print(f"    Предупреждение: Исходное изображение НЕ НАЙДЕНО: {source_img_full_path}")
                        continue
                
                try:
                    with open(output_label_path_in_split, 'a', encoding='utf-8') as f_label:
                        f_label.write(f"{yolo_class_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w_norm:.6f} {yolo_h_norm:.6f}\n")
                    total_annotations_converted_globally += 1
                except Exception as e:
                    print(f"    Ошибка записи в {output_label_path_in_split}: {e}")
        
    # --- Генерация dataset.yaml ---
    dataset_yaml_content = {
        'path': os.path.abspath(OUTPUT_YOLO_DATASET_ROOT_DIR), # Абсолютный путь к корню датасета YOLO
        'train': os.path.join('images', 'train'), # Относительно 'path'
        'val': os.path.join('images', 'val'),     # Относительно 'path'
        'nc': len(final_yolo_class_names),
        'names': final_yolo_class_names
    }
    if TEST_RATIO > 0.0 and len(test_selected_images) > 0 :
        dataset_yaml_content['test'] = os.path.join('images', 'test')

    dataset_yaml_path = os.path.join(OUTPUT_YOLO_DATASET_ROOT_DIR, 'dataset.yaml')
    try:
        with open(dataset_yaml_path, 'w', encoding='utf-8') as f_yaml:
            yaml.dump(dataset_yaml_content, f_yaml, sort_keys=False, allow_unicode=True)
        print(f"\nФайл dataset.yaml успешно создан: {dataset_yaml_path}")
    except Exception as e:
        print(f"\nОшибка при создании dataset.yaml: {e}")

    print(f"\n--- Конвертация и разделение завершены ---")
    print(f"Всего уникальных изображений скопировано в выборки: {len(created_label_files_paths)}")
    print(f"Всего успешно конвертировано и записано аннотаций: {total_annotations_converted_globally}")
    print(f"Структура датасета YOLO создана в: {OUTPUT_YOLO_DATASET_ROOT_DIR}")
    print(f"  images/train/, labels/train/")
    print(f"  images/val/,   labels/val/")
    if TEST_RATIO > 0.0 and len(test_selected_images) > 0:
        print(f"  images/test/,  labels/test/")

if __name__ == '__main__':
    main()