# convert_acdc_to_yolo.py
# Разместить в корневой папке проекта.
import json
import os
import shutil

# --- НАСТРОЙКИ ---
ACDC_ROOT_DIR = 'data/ACDC/' 
ACDC_CATEGORIES_FILE = 'categories.json'
ACDC_RGB_ANON_BASE_DIR = os.path.join(ACDC_ROOT_DIR, 'rgb_anon')

# Имя файла, созданного скриптом select_acdc_subset.py
SELECTED_IMAGES_SUBSET_FILE = 'selected_acdc_images_800imgs.json' # Убедись, что имя верное

# Обновляем имя выходной папки, чтобы оно было уникальным и понятным
# Извлекаем количество из имени файла SELECTED_IMAGES_SUBSET_FILE
try:
    img_count_str = os.path.splitext(SELECTED_IMAGES_SUBSET_FILE)[0].split('imgs')[0].split('_')[-1]
    OUTPUT_YOLO_BASE_DIR = f'data/ACDC_subset_{img_count_str}imgs_yolo_format/'
except IndexError: # На случай, если имя файла не соответствует ожидаемому шаблону
    print("Предупреждение: не удалось извлечь количество изображений из имени файла для OUTPUT_YOLO_BASE_DIR. Используется стандартное имя.")
    OUTPUT_YOLO_BASE_DIR = f'data/ACDC_subset_yolo_format/'


OUTPUT_YOLO_IMGS_DIR = os.path.join(OUTPUT_YOLO_BASE_DIR, 'images')
OUTPUT_YOLO_LABELS_DIR = os.path.join(OUTPUT_YOLO_BASE_DIR, 'labels')

YOLO_CLASS_MAPPING_FROM_NAMES = {
    "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
    "motorcycle": 5, "bicycle": 6, 
    # "train": 5, # Если train нужен, сместить ID motorcycle и bicycle
}
# --- КОНЕЦ НАСТРОЕK ---

def coco_bbox_to_yolo(x_min, y_min, width, height, img_width, img_height):
    x_center = (x_min + width / 2.0)
    y_center = (y_min + height / 2.0)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return x_center_norm, y_center_norm, norm_width, norm_height


def chain_folders_creation(path_to: str):
    if path_to.startswith('/'):
        folders_2create_in_order = path_to.replace('\\', '/').split('/')[1:]

        for counter, folder_name in enumerate(folders_2create_in_order):
            os.makedirs(f'{"/".join(folders_2create_in_order[:counter])}/{folder_name}', exist_ok=True)
        
        return True
    return False


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
            # Это означает, что ID должен был быть использован, но для него не нашлось имени.
            # Проверяем, какое имя ожидалось для этого ID из YOLO_CLASS_MAPPING_FROM_NAMES
            expected_acdc_name_for_id = "[неизвестно]"
            for acdc_name_key, yolo_id_val in YOLO_CLASS_MAPPING_FROM_NAMES.items():
                if yolo_id_val == i:
                    expected_acdc_name_for_id = acdc_name_key
                    break
            print(f"Ошибка: Для YOLO ID {i} (ожидалось для ACDC класса '{expected_acdc_name_for_id}') не найдено соответствующее имя в {ACDC_CATEGORIES_FILE} или класс отсутствует в YOLO_CLASS_MAPPING_FROM_NAMES.")
            return None, None
    final_yolo_class_names_ordered = [name for name in final_yolo_class_names_ordered if name]
    print("Успешно построены отображения категорий.")
    print(f"  ACDC ID -> YOLO ID: {acdc_id_to_yolo_id}")
    print(f"  Имена классов YOLO (для dataset.yaml и utils.py): {final_yolo_class_names_ordered}")
    return acdc_id_to_yolo_id, final_yolo_class_names_ordered

def main():
    print("Этап 2: Конвертация ВЫБРАННОГО Подмножества ACDC в формат YOLO...")
    # Создаем базовые выходные директории images и labels, если их нет
    os.makedirs(OUTPUT_YOLO_IMGS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_YOLO_LABELS_DIR, exist_ok=True)

    acdc_id_to_yolo_id, final_yolo_class_names = build_category_mappings()
    if acdc_id_to_yolo_id is None or final_yolo_class_names is None:
        print("Ошибка отображения категорий. Прерывание.")
        return

    if not os.path.exists(SELECTED_IMAGES_SUBSET_FILE):
        print(f"Ошибка: Файл {SELECTED_IMAGES_SUBSET_FILE} не найден!")
        return
    
    with open(SELECTED_IMAGES_SUBSET_FILE, 'r', encoding='utf-8') as f_subset:
        selected_images_list = json.load(f_subset)

    if not selected_images_list:
        print("Список выбранных изображений пуст.")
        return

    print(f"Загружено {len(selected_images_list)} записей из {SELECTED_IMAGES_SUBSET_FILE}")

    grouped_selected_images = {}
    for sel_img_info in selected_images_list:
        source_json = sel_img_info['source_json_file_path']
        if source_json not in grouped_selected_images:
            grouped_selected_images[source_json] = []
        grouped_selected_images[source_json].append({
            "image_id": sel_img_info['image_id_in_source_json'], 
            "file_name": sel_img_info['file_name_relative']      
        })

    total_annotations_converted_globally = 0
    processed_yolo_images_count = 0

    for source_json_path, images_to_process_from_this_json in grouped_selected_images.items():
        print(f"\nОбработка исходного JSON: {os.path.basename(source_json_path)}")
        normalized_source_json_path = os.path.normpath(source_json_path)
        if not os.path.exists(normalized_source_json_path):
            print(f"  Предупреждение: Исходный JSON {normalized_source_json_path} не найден. Пропуск.")
            continue
        
        try:
            with open(normalized_source_json_path, 'r', encoding='utf-8') as f_gt:
                gt_data = json.load(f_gt)
        except Exception as e:
            print(f"  Ошибка чтения JSON из {normalized_source_json_path}: {e}. Пропуск.")
            continue

        required_image_ids_from_this_json = {img_info['image_id'] for img_info in images_to_process_from_this_json}
        current_json_image_id_to_details = {
            img['id']: {
                "file_name": img['file_name'], 
                "width": img['width'], 
                "height": img['height']
            } for img in gt_data.get('images', []) if img.get('id') in required_image_ids_from_this_json
        }

        annotations_list = gt_data.get('annotations', [])
        if not annotations_list:
            print(f"  В файле {os.path.basename(normalized_source_json_path)} нет 'annotations'.")
            continue

        annotations_for_selected_images_in_this_json = 0
        for ann in annotations_list:
            image_id_from_annotation = ann.get('image_id')
            if image_id_from_annotation not in required_image_ids_from_this_json or \
               image_id_from_annotation not in current_json_image_id_to_details:
                continue

            img_details = current_json_image_id_to_details[image_id_from_annotation]
            img_relative_path = img_details['file_name'] # Например "fog/train/GP010475/GP010475_frame_001043_rgb_anon.png"
            acdc_category_id = ann.get('category_id')
            coco_bbox = ann.get('bbox')

            if acdc_category_id is None or coco_bbox is None or acdc_category_id not in acdc_id_to_yolo_id:
                continue
            
            yolo_class_id = acdc_id_to_yolo_id[acdc_category_id]
            img_width = img_details['width']
            img_height = img_details['height']
            x_min, y_min, bbox_width, bbox_height = coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3]

            if bbox_width <= 0 or bbox_height <= 0: continue

            yolo_x, yolo_y, yolo_w, yolo_h = coco_bbox_to_yolo(x_min, y_min, bbox_width, bbox_height, img_width, img_height)
            
            source_image_full_path = os.path.join(ACDC_RGB_ANON_BASE_DIR, img_relative_path)
            
            # Имя файла для YOLO датасета (заменяем / или \ на _, чтобы избежать создания подпапок внутри images/labels)
            # Это сохранит уникальность имен, если file_name_relative содержал пути
            # img_relative_path_cleaned = img_relative_path.replace('/', '_').replace('\\', '_')
            yolo_flat_image_filename = img_relative_path.replace('/', '_').replace('\\', '_')
            yolo_flat_label_filename = os.path.splitext(yolo_flat_image_filename)[0] + '.txt'
            
            output_image_full_path = os.path.join(OUTPUT_YOLO_IMGS_DIR, yolo_flat_image_filename)
            output_label_full_path = os.path.join(OUTPUT_YOLO_LABELS_DIR, yolo_flat_label_filename)

            # Копируем изображение только один раз
            if not os.path.exists(output_image_full_path):
                if os.path.exists(source_image_full_path):
                    try:
                        # Папки OUTPUT_YOLO_IMGS_DIR и OUTPUT_YOLO_LABELS_DIR уже созданы в начале main()
                        shutil.copy(source_image_full_path, output_image_full_path)
                        processed_yolo_images_count +=1 
                    except Exception as e:
                        print(f"    Ошибка копирования {source_image_full_path} -> {output_image_full_path}: {e}")
                        continue 
                else:
                    print(f"    Предупреждение: Исходное изображение НЕ НАЙДЕНО: {source_image_full_path} (для {img_relative_path}).")
                    continue 
            
            try:
                # Папка OUTPUT_YOLO_LABELS_DIR уже создана
                with open(output_label_full_path, 'a', encoding='utf-8') as f_label:
                    f_label.write(f"{yolo_class_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
                annotations_for_selected_images_in_this_json += 1
            except Exception as e:
                print(f"    Ошибка записи в {output_label_full_path}: {e}")
        
        total_annotations_converted_globally += annotations_for_selected_images_in_this_json
        if annotations_for_selected_images_in_this_json > 0:
            print(f"  Для {os.path.basename(normalized_source_json_path)} обработано {annotations_for_selected_images_in_this_json} аннотаций.")

    print(f"\n--- Конвертация выбранного подмножества завершена ---")
    print(f"Всего скопировано уникальных изображений: {processed_yolo_images_count}")
    print(f"Всего успешно конвертировано аннотаций: {total_annotations_converted_globally}")
    print(f"Итоговые изображения для YOLO: {OUTPUT_YOLO_IMGS_DIR}")
    print(f"Итоговые метки для YOLO:     {OUTPUT_YOLO_LABELS_DIR}")
    print(f"\nВАЖНО: Список имен классов для dataset.yaml (и для utils.py):")
    print(final_yolo_class_names)

if __name__ == '__main__':
    main()
    # chain_folders_creation('/aboba/1/2/3/4')