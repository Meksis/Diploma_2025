# Weather_Distortions_Recognition/src/development/train.py
from ultralytics import YOLO
import os
import argparse
import json
import time
import yaml

EXPERIMENTS_LOG_FILE = "/diploma_code/runs/experiments_log.json" 

def log_experiment(exp_details):
    log_data = []
    os.makedirs(os.path.dirname(EXPERIMENTS_LOG_FILE), exist_ok=True)
    if os.path.exists(EXPERIMENTS_LOG_FILE):
        try:
            with open(EXPERIMENTS_LOG_FILE, 'r', encoding='utf-8') as f: log_data = json.load(f)
        except json.JSONDecodeError: log_data = []
    log_data.append(exp_details)
    try:
        with open(EXPERIMENTS_LOG_FILE, 'w', encoding='utf-8') as f: json.dump(log_data, f, indent=4, ensure_ascii=False)
        print(f"Детали эксперимента сохранены в {EXPERIMENTS_LOG_FILE}")
    except Exception as e: print(f"Ошибка при сохранении лога: {e}")

def main(args):
    model = YOLO(args.model_base) 

    if not os.path.exists(args.dataset_yaml):
        print(f"Ошибка: Файл датасета {args.dataset_yaml} не найден.")
        return
    
    augmentation_params = {}
    if args.hyp_augmentations:
        if os.path.exists(args.hyp_augmentations):
            try:
                with open(args.hyp_augmentations, 'r') as f:
                    augmentation_params = yaml.safe_load(f)
                print(f"  Загружены параметры аугментации из: {args.hyp_augmentations}")
                print(f"  Параметры аугментации: {augmentation_params}")
            except Exception as e:
                print(f"Ошибка загрузки YAML файла аугментаций {args.hyp_augmentations}: {e}. Используются дефолты.")
        else:
            print(f"Ошибка: Файл гиперпараметров аугментации {args.hyp_augmentations} не найден. Используются дефолты.")
    else:
        print(f"  Файл гиперпараметров аугментации не указан. Используются дефолтные аугментации Ultralytics (или их отсутствие, если augment=False).")


    print(f"\n--- Запуск обучения: {args.exp_name} в проекте {args.project} ---")
    print(f"  Начальный Learning Rate (lr0): {args.lr0}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Оптимизатор: {args.optimizer}")

    train_kwargs = {
        "data": args.dataset_yaml,
        "epochs": args.epochs,
        "imgsz": args.img_size,
        "batch": args.batch,
        "project": args.project,
        "name": args.exp_name,
        "device": args.device,
        "patience": args.patience,
        "model": args.model_base, 
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "box": args.box_loss_gain,
        "cls": args.cls_loss_gain,
        "dfl": args.dfl_loss_gain,
        "optimizer": args.optimizer,
        "close_mosaic": args.close_mosaic_epochs,
        "augment": args.use_general_augment_flag 
    }


    if augmentation_params:
        train_kwargs.update(augmentation_params)
        if 'augment' not in augmentation_params: 
             train_kwargs['augment'] = True 

    final_train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None or k in augmentation_params}


    print(f"  Финальные параметры для model.train(): {final_train_kwargs}")

    results = model.train(**final_train_kwargs)
    
    print("Обучение завершено.")
    print(f"Модель и результаты сохранены в: {results.save_dir}") 

    log_entry = {
        "experiment_name": args.exp_name, "project_dir": args.project,
        "run_dir": str(results.save_dir), "dataset_yaml": args.dataset_yaml,
        "base_model": args.model_base, "epochs": args.epochs, "batch_size": args.batch,
        "img_size": args.img_size, "device": args.device, 
        "hyp_augmentations_file": args.hyp_augmentations,
        "augmentation_params_used": augmentation_params if augmentation_params else "Defaults/CLI",
        "lr0": args.lr0, "lrf": args.lrf, "momentum": args.momentum, "wd": args.weight_decay,
        "optimizer": args.optimizer,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "final_mAP50_95": float(results.box.map) if hasattr(results, 'box') and hasattr(results.box, 'map') else None,
        "final_mAP50": float(results.box.map50) if hasattr(results, 'box') and hasattr(results.box, 'map50') else None,
    }
    log_experiment(log_entry)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт обучения модели YOLO.")

    parser.add_argument('--dataset_yaml', type=str, required=True)
    parser.add_argument('--model_base', type=str, default='yolov8n.pt') 
    parser.add_argument('--project', type=str, default="/diploma_code/runs/detect_experiments")
    parser.add_argument('--exp_name', type=str, default='experiment')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--patience', type=int, default=20)
    
    parser.add_argument('--hyp_augmentations', type=str, default=None, help='Путь к YAML файлу ТОЛЬКО с параметрами аугментации.')
    
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_epochs', type=float, default=3.0)
    parser.add_argument('--warmup_momentum', type=float, default=0.8)
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1)
    parser.add_argument('--box_loss_gain', type=float, default=7.5, dest='box_loss_gain')
    parser.add_argument('--cls_loss_gain', type=float, default=0.5, dest='cls_loss_gain')
    parser.add_argument('--dfl_loss_gain', type=float, default=1.5, dest='dfl_loss_gain')
    parser.add_argument('--optimizer', type=str, default='auto', choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'])
    parser.add_argument('--use_general_augment_flag', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='Общий флаг для вкл/выкл стандартных аугментаций Ultralytics, если --hyp_augmentations не указан.')
    parser.add_argument('--close_mosaic_epochs', type=int, default=10, dest='close_mosaic_epochs')

    cli_args = parser.parse_args()
    main(cli_args)