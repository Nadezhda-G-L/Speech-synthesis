"""Build the filtered RUSLAN metadata — entry point for lab 1.

Reads the corpus metadata (two columns), runs every utterance through the normalizer,
drops whatever the classifier rejects, and writes the result in LJSpeech format
(three columns)::

    000000_RUSLAN|С тревожным чувством берусь я за перо.|С тревожным чувством берусь я за перо.
                 ^ raw text                             ^ normalized text

The third column is your contribution: the original corpus does not have one. For most
rows it will equal the second, and that is expected — punctuation cleanup changes little.

Run from the lab directory::

    python preprocess_ruslan.py
"""

import csv
import os

import pandas as pd

# Убедитесь, что путь к модулям корректен
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем BERT-классификатор
from text_filter import load_trained_model
from text_normalizer import TextNormalizer

INPUT_PATH = "../../data/ruslan_dataset/metadata_RUSLAN_22200.csv"
OUTPUT_PATH = "../../data/metadata_RUSLAN_22200_normalized.csv"

# quoting=csv.QUOTE_NONE is required in both directions: the corpus text contains
# « » „ “ ” ' and pandas would otherwise read them as field delimiters.
CSV_KWARGS = {"sep": "|", "quoting": csv.QUOTE_NONE}


if __name__ == "__main__":
    # Проверяем существование входного файла
    if not os.path.exists(INPUT_PATH):
        print(f"Ошибка: файл {INPUT_PATH} не найден!")
        exit(1)
    
    # Загружаем предобученную модель
    print("Loading trained BERT model...")
    text_filter = load_trained_model("./bert_model.pth")
    normalizer = TextNormalizer()

    raw = pd.read_csv(INPUT_PATH, names=["id", "raw"], **CSV_KWARGS)
    
    # Проверка на пустые строки
    raw = raw.dropna(subset=['raw'])
    raw = raw[raw['raw'].str.strip() != '']

    raw["nrm"] = raw["raw"].apply(normalizer.normalize)
    
    # Применяем фильтр
    def filter_text(text):
        try:
            result = text_filter.filter(text)
            return result
        except Exception as e:
            print(f"Error processing text: {text[:50]}... Error: {e}")
            return 0  # По умолчанию отбрасываем
    
    raw["filtered"] = raw["nrm"].apply(filter_text)
    clean = raw[raw["filtered"] == 1]

    print(f"Num rows before cleaning: {len(raw)}; after cleaning: {len(clean)}")

    # Проверка, что файл не пуст
    if len(clean) == 0:
        print("Внимание: после фильтрации не осталось строк!")
    else:
        clean[["id", "raw", "nrm"]].to_csv(
            OUTPUT_PATH, index=False, header=False, **CSV_KWARGS
        )
        print(f"Файл сохранен: {OUTPUT_PATH}")