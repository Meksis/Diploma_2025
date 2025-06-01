import shutil, os, json
from random import choice

class Utils:
    def pick_data_files(total: int, train_p: int, test_p: int, val_p: int) -> bool:
        ds_path = ''

        prefix = 'YOLO_data/datasets'
        shutil.rmtree(prefix, True)
        os.makedirs(prefix)

        n_train, n_test, n_val = round(total * train_p), round(total * test_p), round(total * val_p)

        already_picked_images = []

        for dir_name, n_items in {'train': n_train, 'test': n_test, 'val': n_val}.items():
            os.makedirs(f'{prefix}/{dir_name}/images'), os.makedirs(f'{prefix}/{dir_name}/labels')

            for _ in range(n_items):
                image = fr"{ds_path}/dataset/imgs/{choice(os.listdir(f'{ds_path}/dataset/imgs'))}"

                while image in already_picked_images:
                    image = fr"{ds_path}/dataset/imgs/{choice(os.listdir(f'{ds_path}/dataset/imgs'))}"
                
                txt = f'{ds_path}/dataset/txt/{image.split('/')[-1].split('.')[-2]}.txt'


                shutil.copy(image, f'{prefix}/{dir_name}/images/{image.split('/')[-1]}')
                shutil.copy(txt, f'{prefix}/{dir_name}/labels/{txt.split('/')[-1]}')

        '''
            pick_data_files(200, .7, .15, .15)
        '''
        return True
        print('Успешно разбили на выборки и перенесли файлы')

    
    def save_new_experiment_name(name: str, results_path: str) -> bool:
        try:
            with open('experiments.json', 'r', encoding='utf-8') as file:
                experiments = json.load(file)

            experiments.append({name: results_path})

            with open('experiments.json', 'w', encoding='utf-8') as file:
                json.dump(experiments, file, indent=4)

            return True
        
        
        except Exception as e:
            print(f"Ошибка: {e}")
            return False