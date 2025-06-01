import pandas as pd
import json

with open('./runs/experiments_log.json', 'r') as file:
    runs_data = json.load(file)

df = pd.DataFrame.from_dict(runs_data)

df['final_mAP50_95'] = df['final_mAP50_95'].replace(0.0, float('nan'))
df['final_mAP50'] = df['final_mAP50'].replace(0.0, float('nan'))

df_successful = df.dropna(subset=['final_mAP50_95', 'final_mAP50'])

df_sorted_map50_95 = df_successful.sort_values(by='final_mAP50_95', ascending=False)

df_sorted_map50 = df_successful.sort_values(by='final_mAP50', ascending=False)

print("--- Топ экспериментов по final_mAP50_95 ---")
print(df_sorted_map50_95[['lr0', 'batch_size', 'epochs', 'wd', 'optimizer', 'final_mAP50_95', 'final_mAP50']].head())

print("\n--- Топ экспериментов по final_mAP50 ---")
print(df_sorted_map50[['lr0', 'batch_size', 'epochs', 'wd', 'optimizer', 'final_mAP50_95', 'final_mAP50']].head())

print("\n--- Результаты для SGD ---")
sgd_results = df_successful[df_successful['optimizer'] == 'SGD']
print(sgd_results[['lr0', 'batch_size', 'epochs', 'wd', 'final_mAP50_95', 'final_mAP50']].sort_values(by='final_mAP50_95', ascending=False))


print("\n--- Результаты для AdamW/'auto' ---")
adamw_results = df_successful[df_successful['optimizer'] != 'SGD'] #
print(adamw_results[['lr0', 'batch_size', 'epochs', 'wd', 'final_mAP50_95', 'final_mAP50']].sort_values(by='final_mAP50_95', ascending=False).head())


original_nan_run = pd.DataFrame(runs_data)
original_nan_run = original_nan_run[
    (original_nan_run['lr0'] == 0.005) &
    (original_nan_run['batch_size'] == 6) &
    (original_nan_run['epochs'] == 60) &
    (original_nan_run['wd'] == 0.0005) &
    (original_nan_run['optimizer'] == 'auto')
]
print("Исходная запись для NaN-run из лога:")
print(original_nan_run[['lr0', 'batch_size', 'epochs', 'wd', 'optimizer', 'final_mAP50_95', 'final_mAP50']])