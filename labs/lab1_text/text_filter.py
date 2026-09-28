"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv
import pathlib
import re
import random
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).parent
DEV_SET_PATH = SCRIPT_DIR / "data" / "dev_sentences.csv"

class TextFilter:
    """Decides whether an utterance is usable as a training example."""

    def __init__(self):
        """Train the classifier on the dev set."""
        if not DEV_SET_PATH.exists():
            raise FileNotFoundError(f"Не найден файл: {DEV_SET_PATH}")

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

        np.random.seed(42)
        random.seed(42)

        # Список паттернов мусора. Чередование позиций (сначала в начале, потом в конце)
        # помогает модели понять, что шум не привязан к конкретной части фразы.
        trash_patterns = [
            "(( {0}", "{0} ((",
            ")) {0}", "{0} ))",
            "((( {0}", "{0} (((",
            "))) {0}", "{0} )))",
            "!! {0}", "{0} !!",
            "?? {0}", "{0} ??",
            "!!! {0}", "{0} !!!",
            "??? {0}", "{0} ???",
            "Ммм, {0}", "{0} ммм",
            "ага, {0}", "{0} ага",
            "хаха, {0}", "{0} хаха",
            "угу, {0}", "{0} угу",
            "Гмм, {0}", "{0} гмм",
            "Хмм, {0}", "{0} хмм",
            "MacBook {0}", "{0} MacBook",
            "iPhone {0}", "{0} iPhone",
            "test {0}", "{0} test",
            "app {0}", "{0} app",
            "error {0}", "{0} error",
            "100 {0}", "{0} 100",
            "2026 {0}", "{0} 2026",
            "999 {0}", "{0} 999",
            "null {0}", "{0} null",
        ]

        # Детерминированный перебор: берём фразы и мусор строго по индексу
        for i, pattern in enumerate(trash_patterns):
            # Циклический доступ без random.choice() и .sample()
            base_text = good_examples.iloc[i % len(good_examples)]['text']
            synth_text.append(pattern.format(base_text))
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
        
        # Получаем вероятность, что фраза нормизована (класс 1)
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

    # Ищем примеры, где модель ошиблась

    # 1. False Negative (Модель удалила НОРМУ)
    # Метка 1 (Норма), а модель сказала 0 (Мусор)
    false_negatives = dev_files[(dev_files["is_normalized"] == 1) & (dev_files["predicted"] == 0)]

    # 2. False Positive (Модель пропустила МУСОР)
    # Метка 0 (Мусор), а модель сказала 1 (Норма)
    false_positives = dev_files[(dev_files["is_normalized"] == 0) & (dev_files["predicted"] == 1)]

    print("\n========================================")
    print("АНАЛИЗ ОШИБОК МОДЕЛИ")
    print("========================================")

    print(f"\nМОДЕЛЬ ОШИБАЕТСЯ, УДАЛЯЯ НОРМУ:")
    print(f"Всего таких примеров: {len(false_negatives)}")
    print("-" * 40)
    # Выводим первые 5 примеров для отчета
    for text in false_negatives["text"].head(5):
        print(f" - '{text}'")

    print(f"\nМОДЕЛЬ ПРОПУСКАЕТ МУСОР:")
    print(f"Всего таких примеров: {len(false_positives)}")
    print("-" * 40)
    # Выводим первые 5 примеров для отчета
    for text in false_positives["text"].head(5):
        print(f" - '{text}'")