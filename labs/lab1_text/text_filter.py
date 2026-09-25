"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv
import pathlib
import re
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Пути к файлам 
SCRIPT_DIR = pathlib.Path(__file__).parent
DEV_SET_PATH = SCRIPT_DIR / "data" / "dev_sentences.csv"

# Путь для сохранения весов модели
CHECKPOINT_PATH = SCRIPT_DIR / "checkpoints" / "text_filter.pkl"

class TextFilter:
    """Decides whether an utterance is usable as a training example."""

    def __init__(self):
        """Train the classifier on the dev set or load from checkpoint."""
        
        # Если модель уже обучена и сохранена — загружаем её, чтобы избежать переобучения и случайности
        if CHECKPOINT_PATH.exists():
            print("Загрузка сохраненной модели...")
            data = joblib.load(CHECKPOINT_PATH)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.threshold = data['threshold']
            return

        # 1. Загружаем обучающую выборку
        df = pd.read_csv(DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0)
        
        # 2. Очистка от пропусков
        df['text'] = df['text'].fillna('').astype(str)
        
        # Добавляем синтетические данные
        # Модель видела только 300 примеров. Ей нужно больше примеров "мусора" с цифрами и латиницей,
        # чтобы она не путала их с хорошим текстом.
        good_examples = df[df['is_normalized'] == 1]
        bad_examples = df[df['is_normalized'] == 0]
        
        # Создаем искусственные "плохие" примеры, добавляя мусор к хорошим фразам
        synth_text = []
        synth_labels = []
        
        # Фиксируем seed для повторяемости результатов
        np.random.seed(42)
        random.seed(42)
        
        # 50 синтетических примеров "Плохо - Латиница"
        trash_words = ["MacBook", "iPhone", "app", "test", "error", "null"]
        for i in range(50):
            # Теперь мы используем детерминированный выбор по индексу (i % length),
            # чтобы каждый раз брать именно те же слова и фразы, а не случайные.
            base_text = good_examples.iloc[i % len(good_examples)]['text']
            trash_word = trash_words[i % len(trash_words)]
            synth_text.append(f"{trash_word} {base_text}")
            synth_labels.append(0)
            
        # 50 синтетических примеров "Плохо - Цифры"
        trash_digits = ["100", "2026", "999", "000"]
        for i in range(50):
            base_text = good_examples.iloc[i % len(good_examples)]['text']
            trash_digit = trash_digits[i % len(trash_digits)]
            synth_text.append(f"{base_text} {trash_digit}")
            synth_labels.append(0)
            
        # Добавляем это к исходному датафрейму
        df_augmented = pd.DataFrame({
            'text': list(df['text']) + synth_text,
            'is_normalized': list(df['is_normalized']) + synth_labels
        })

        # 3. Разделяем признаки и метки
        X = df_augmented['text']
        y = df_augmented['is_normalized'].astype(int).values
        
        # 4. Создаем модель
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        # Сохраняем знаки препинания в признаках для ML
        self.vectorizer = TfidfVectorizer(max_features=15000, token_pattern=r"(?u)\S+", ngram_range=(1, 2))
        
        # 5. Обучаем модель на расширенном наборе
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

        # Порог вероятности. 0.45 позволяет сохранить больше фраз (как Wi-Fi или 22-го),
        # но не пропустить явный мусор. С 0.5 остается только около 7000 фраз
        self.threshold = 0.45

        # Сохраняем веса модели в файл (checkpoints), чтобы в следующий раз не обучать её заново
        os.makedirs(CHECKPOINT_PATH.parent, exist_ok=True)
        joblib.dump({'model': self.model, 'vectorizer': self.vectorizer, 'threshold': self.threshold}, CHECKPOINT_PATH)
        print(f"Модель сохранена в: {CHECKPOINT_PATH}")

    def filter(self, text: str) -> int:
        """Classify a single utterance.
        Returns:
            1 if normalized, 0 if not.
        """
        if not text:
            return 1 
        
        # 1. Жесткие правила для явного мусора (символы, ссылки, email)
        # Мы оставляем только самое очевидное, остальное решит модель (благодаря синтетическим данным)
        self.junk_re = re.compile(r'[*#@&€$%]')
        self.url_re = re.compile(r'(http|www|ftp)')
        
        if self.junk_re.search(text):
            return 0
        if self.url_re.search(text):
            return 0
        
        # 2. Передаем текст модели
        X = self.vectorizer.transform([text])
        
        # Получаем вероятность, что фраза нормализована (класс 1)
        prob = self.model.predict_proba(X)[0][1]
        
        # 3. Возвращаем результат по порогу
        # Если вероятность >= 0.45, значит модель "уверена наполовину" или больше, что это ок.
        return 1 if prob >= self.threshold else 0

if __name__ == "__main__":
    textfilter = TextFilter()
    dev_files = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0
    )
    
    dev_files["predicted"] = dev_files["text"].apply(textfilter.filter)

    prc = precision_score(dev_files["is_normalized"], dev_files["predicted"])
    rec = recall_score(dev_files["is_normalized"], dev_files["predicted"])
    f1 = f1_score(dev_files["is_normalized"], dev_files["predicted"])
    
    print(f"F1 Score is {f1:.4f}, Precision is {prc:.4f}, Recall is {rec:.4f}")
    print(f"Сохранено: {(dev_files['predicted']==1).sum()}, Отброшено: {(dev_files['predicted']==0).sum()}")