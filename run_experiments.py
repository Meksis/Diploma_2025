import subprocess
import os
import itertools
import time

DATASET_YAML_PATH_IN_CONTAINER = "../datasets/acdc_final/dataset.yaml" 
BASE_MODEL_NAME_FOR_ULTRALYTICS = "/models/yolo12m.pt"
AUGMENTATION_HYP_FILE_IN_CONTAINER = "/datasets/acdc_final/augmentations.yaml" 
   
IMAGE_SIZE = 640
DEVICE = "0"
PATIENCE = 20
BASE_PROJECT_DIR_FOR_TRAIN_PY = "/diploma_code/runs/BEST_AUTO_ML_QUALIFICATION" 
# BASE_PROJECT_DIR_FOR_TRAIN_PY = "/diploma_code/runs/final_hyperparam_runs_inside_docker_auto_dl" # Новое имя папки

learning_rates = [0.001] 
batch_sizes = [4, 8] 
epoch_counts = [60, 100] 
# epoch_counts = [20, 40, 60] 
# weight_decays = [0.0005, 0.0001]
weight_decays = [0.0005]
# optimizers = ['auto']
optimizers = ['SGD']

def main():
    if not os.path.exists(os.path.dirname(BASE_PROJECT_DIR_FOR_TRAIN_PY)):
        try:
            os.makedirs(os.path.dirname(BASE_PROJECT_DIR_FOR_TRAIN_PY), exist_ok=True)
        except OSError as e:
            print(f"Не удалось создать родительскую директорию для проектов: {e}")
       
    experiment_run_number = 0
    
    param_combinations = list(itertools.product(
        learning_rates, batch_sizes, epoch_counts, weight_decays, optimizers
    ))
    total_experiments = len(param_combinations)
    print(f"Всего запланировано экспериментов: {total_experiments}")

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # path_to_train_py = os.path.join(current_script_dir, './train.py')
    path_to_train_py = os.path.join(current_script_dir, 'Weather_Distortions_Recognition/src/development/train.py')
    if not os.path.exists(path_to_train_py):
        path_to_train_py = "train.py" 
        print(f"Предупреждение: train.py не найден по относительному пути, используется '{path_to_train_py}'.")

    for lr, batch, epochs, wd, optimizer_choice in param_combinations:
        experiment_run_number += 1
        
        project_name_for_run = (
            f"lr{str(lr).replace('.', 'p')}"
            f"_b{batch}"
            f"_e{epochs}"
            f"_wd{str(wd).replace('.', 'p')}"
            f"_opt_{optimizer_choice}"
        )
        current_project_path = os.path.join(BASE_PROJECT_DIR_FOR_TRAIN_PY, project_name_for_run)
        run_name = "train_run" 

        print(f"\n--- Запуск эксперимента {experiment_run_number}/{total_experiments} ---")
        print(f"  Параметры: lr0={lr}, batch={batch}, epochs={epochs}, weight_decay={wd}, optimizer={optimizer_choice}")
        print(f"  Модель: {BASE_MODEL_NAME_FOR_ULTRALYTICS} (автозагрузка)")
        print(f"  Файл аугментаций: {AUGMENTATION_HYP_FILE_IN_CONTAINER}")
        print(f"  Project Path: {current_project_path}")
        print(f"  Run Name: {run_name}")

        command_parts = [
            "python", path_to_train_py,
            "--dataset_yaml", DATASET_YAML_PATH_IN_CONTAINER,
            "--model_base", BASE_MODEL_NAME_FOR_ULTRALYTICS, 
            "--project", current_project_path,
            "--exp_name", run_name,          
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--img_size", str(IMAGE_SIZE),
            "--device", DEVICE,
            "--patience", str(PATIENCE),
            "--hyp_augmentations", AUGMENTATION_HYP_FILE_IN_CONTAINER,
            "--lr0", str(lr),
            "--weight_decay", str(wd),
            "--optimizer", optimizer_choice
        ]
        
        print(f"  Выполнение команды: {' '.join(command_parts)}")
        try:
            process = subprocess.run(command_parts, check=True, text=True, capture_output=False)
            print(f"--- Эксперимент {experiment_run_number} успешно завершен ---")
        except subprocess.CalledProcessError as e:
            print(f"!!! Ошибка в эксперименте {experiment_run_number}: {e} !!!")
            print(f"    Команда была: {' '.join(e.cmd)}")
            if hasattr(e, 'stdout') and e.stdout: print(f"    STDOUT: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr: print(f"    STDERR: {e.stderr}")
        except Exception as e:
            print(f"!!! Непредвиденная ошибка в эксперименте {experiment_run_number}: {e} !!!")
        
        print("-" * 50)
        time.sleep(2)

if __name__ == "__main__":
    print("Запуск скрипта перебора экспериментов ИЗНУТРИ КОНТЕЙНЕРА (с автозагрузкой модели).")
    print(f"Файл аугментаций '{AUGMENTATION_HYP_FILE_IN_CONTAINER}' будет использоваться.")
    main()