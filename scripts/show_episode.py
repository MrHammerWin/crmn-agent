"""Read a saved real trajectory; never silently call a model when inspecting results."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episodes', type=Path)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.episodes.read_text().splitlines() if s.strip()]
    row = rows[args.index]
    print(f"{row['benchmark']} / {row['task']} / {row['variant']} / {row['model']}")
    print(f"Статус: {row['status']}; успех: {row['success']}; шагов: {row['steps']}")
    for turn in row['trajectory']:
        print(f"\nШАГ {turn['step'] + 1}\nЦель: {turn['mission']}\nНаблюдение: {turn['observation']}")
        diagnostics = turn['diagnostics']
        if 'responsibilities' in diagnostics:
            print('Веса карты и плана:', diagnostics['responsibilities'])
            print('Ошибка восстановления:', diagnostics['cycle_errors'])
            print('Ошибка последствий:', diagnostics['prediction_ema'])
            print('Ошибка награды:', diagnostics['reward_error_ema'])
        print('Действие:', turn['action'])
        print('Результат:', turn['next_observation'])


if __name__ == '__main__':
    main()
