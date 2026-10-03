"""
Pause Predictor with BERT Embeddings
"""
import csv
import re
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings("ignore")

PAUSE_PREDICTOR_DATA = '../../data/ruslan_dataset/RUSLAN_pause_metadata.csv'

# Базовые наборы символов и слов
PUNCT_CHARS = set('.,!?:;\u2014"\'')
FILLER_WORDS = {'ну', 'короче', 'значит', 'вот', 'типа', 'как', 'в общем', 'итак', 'слышь', 'понимаешь', 'то', 'так'}
FUNCTION_WORDS = {'в', 'и', 'на', 'как', 'но', 'те', 'бы', 'ли', 'а', 'о', 'об', 'от', 'к', 'за', 'при', 'с', 'у', 'под', 'над'}

class PausePredictor:
    def __init__(self):
        # Загрузка BERT модели для получения эмбеддингов
        self.tokenizer = AutoTokenizer.from_pretrained('../../labs/lab1_text/models/rubert_tiny2')
        self.bert_model = AutoModel.from_pretrained('../../labs/lab1_text/models/rubert_tiny2')
        self.bert_model.eval()
        
        # Модели классификации и регрессии
        self.pause_clf = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            objective='binary',
            metric='binary_logloss',
            verbose=-1
        )
        self.duration_reg = LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        self.threshold = 0.45
        self.is_trained = False

    def get_bert_embeddings(self, tokens):
        """Получение BERT эмбеддингов для токенов"""
        embeddings = []
        
        for token in tokens:
            # Токенизация с помощью BERT токенизатора
            inputs = self.tokenizer(token, return_tensors='pt', truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Используем усредненное последнее скрытое состояние
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()  # Усредняем по токенам
                embeddings.append(embedding.flatten())
        
        return np.array(embeddings)

    def extract_features(self, tokens):
        """Извлечение признаков из токенов"""
        if len(tokens) == 0:
            return np.zeros((0, 25 + 128), dtype=np.float32)  # +128 для BERT эмбеддингов

        N = len(tokens)
        features = []
        is_punct = [len(t) > 0 and t[-1] in PUNCT_CHARS for t in tokens]

        for i, token in enumerate(tokens):
            # Обработка типов данных
            if isinstance(token, np.ndarray):
                token = str(token.item()) if token.size > 0 else ""
            elif not isinstance(token, str):
                token = str(token)

            # Очистка и анализ токена
            clean = re.sub(r'[^\w]', '', token.lower())
            is_word = len(clean) > 0 and clean[0].isalpha()
            len_tok = len(token)
            len_clean = len(clean)
            p = token[-1] if len(token) > 0 else ""

            # Поиск ближайших пунктуационных знаков
            next_punct_idx = next((j for j in range(i + 1, N) if is_punct[j]), -1)
            dist_to_next = (next_punct_idx - i) if next_punct_idx != -1 else 20

            prev_punct_idx = next((j for j in range(i - 1, -1, -1) if is_punct[j]), -1)
            dist_to_prev = (i - prev_punct_idx) if prev_punct_idx != -1 else 20

            # Проверка типов слов
            clean_alpha = re.sub(r'[^а-яёa-z]', '', clean)
            is_func = clean_alpha in FUNCTION_WORDS
            is_filler = (
                clean_alpha in FILLER_WORDS or
                (clean_alpha.startswith('что') or clean_alpha.startswith('как')) or
                (clean_alpha in ['короче', 'ну', 'вот', 'типа', 'значит', 'кажется', 'вернее'])
            )
            is_short_word = len_clean <= 2 and is_word
            is_short_punct = len(clean) < 6 and p in PUNCT_CHARS

            # Контекстные фичи
            prev_word = tokens[i-1].lower() if i > 0 else ""
            next_word = tokens[i+1].lower() if i < N-1 else ""

            prev_is_filler = prev_word in FILLER_WORDS or prev_word.startswith('что') or prev_word.startswith('как')
            next_is_filler = next_word in FILLER_WORDS or next_word.startswith('что') or next_word.startswith('как')

            # Фичи для контекста
            after_comma = len(tokens[i-1]) > 0 and tokens[i-1][-1] == ',' if i > 0 else False
            before_punct = len(tokens[i+1]) > 0 and tokens[i+1][-1] in '.!?' if i < N-1 else False

            # Фича: длина фразы до следующего знака препинания
            phrase_length = next_punct_idx - i if next_punct_idx != -1 else N - i

            # Основные признаки
            base_feats = [
                len_tok, len_clean,
                float(p == ','), float(p == '.'), float(p == '!'), float(p == '?'),
                float(p == ';'), float(p == ':'),
                float('—' in token or '-' in token),
                float(any(c in token for c in '()[]{}')),
                float(p in PUNCT_CHARS),
                float(is_word), float(is_short_word), float(len_clean > 8),
                float(token.isupper()), float(token.islower()), float(token.isdigit()),
                float(len(tokens[i-1]) > 0 and tokens[i-1][-1] in PUNCT_CHARS) if i > 0 else 0.0,
                float(len(tokens[i+1]) > 0 and tokens[i+1][-1] in PUNCT_CHARS) if i < N-1 else 0.0,
                float(dist_to_next > 0),
                min(dist_to_next, 10) / 10.0,
                min(dist_to_prev, 10) / 10.0,
                min((dist_to_next + dist_to_prev) / 2, 10) / 10.0,
                float(is_func), float(is_filler), float(is_short_punct),
                float(prev_is_filler), float(next_is_filler),
                float(after_comma), float(before_punct),
                float(phrase_length <= 3), float(phrase_length <= 6), float(phrase_length > 6)
            ]
            
            features.append(base_feats)

        # Получаем BERT эмбеддинги для всех токенов
        bert_embeddings = self.get_bert_embeddings(tokens)
        
        # Комбинируем базовые фичи с эмбеддингами
        final_features = []
        for i, base_feat in enumerate(features):
            if i < len(bert_embeddings):
                # Добавляем BERT эмбеддинги к базовым фичам
                combined = base_feat + bert_embeddings[i].tolist()
                final_features.append(combined)
            else:
                # Если эмбеддингов нет, используем только базовые фичи
                final_features.append(base_feat)
        
        return np.array(final_features, dtype=np.float32)

    def _find_threshold(self, y_true, y_proba):
        """Поиск оптимального порога для классификации"""
        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.20, 0.65, 0.005):
            pred = (y_proba > thr).astype(int)
            if pred.sum() == 0: continue
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1: best_f1, best_thr = f1, thr
        return best_thr

    def train(self, train_df):
        """Обучение модели"""
        print("Training Predictor with BERT Embeddings...")
        sentences = train_df.groupby('id', sort=False)

        X_all, y_pause_all, X_dur_all, y_dur_all = [], [], [], []

        for uid, group in sentences:
            if group.empty: continue
            mask = group['is_last_word'].values == 0
            if mask.sum() == 0: continue

            feats = self.extract_features(group['label_raw'].values)
            y_p = group['is_pause_after'].values[mask]
            X_p = feats[mask]
            y_d = group['pause_duration'].values[mask]

            X_all.append(X_p)
            y_pause_all.append(y_p)

            pos_mask = y_p == 1
            if pos_mask.sum() > 0:
                X_dur_all.append(X_p[pos_mask])
                durations_pos = y_d[pos_mask]
                y_dur_all.append(durations_pos[durations_pos > 0.01])

        X_train = np.vstack(X_all)
        y_train = np.concatenate(y_pause_all)
        print(f"Full Training Shape: {X_train.shape}, Positives: {y_train.sum()}")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        sample_weights = np.ones_like(y_tr, dtype=np.float32)
        if n_pos > 0:
            sample_weights[y_tr == 1] = n_neg / n_pos * 1.3

        # Обучение классификатора
        self.pause_clf.fit(X_tr, y_tr, sample_weight=sample_weights)
        y_proba_val = self.pause_clf.predict_proba(X_val)[:, 1]
        self.threshold = self._find_threshold(y_val, y_proba_val)

        print(f"Optimized Threshold: {self.threshold:.2f} (Val F1: {f1_score(y_val, (y_proba_val > self.threshold).astype(int)):.3f})")

        if X_dur_all:
            X_dur_train = np.vstack(X_dur_all)
            y_dur_train = np.concatenate(y_dur_all)
            if len(y_dur_train) > 10:
                self.duration_reg.fit(X_dur_train, y_dur_train)

        self.is_trained = True
        print("Done.\n")

    def predict(self, tokens):
        """Предсказание пауз для токенов"""
        if not self.is_trained or len(tokens) == 0:
            return np.zeros(len(tokens), dtype=int), np.zeros(len(tokens), dtype=float)

        feats = self.extract_features(tokens)
        proba = self.pause_clf.predict_proba(feats)[:, 1]

        # Фильтрация ложных срабатываний
        is_pause = (proba > self.threshold).astype(int)

        # Уточнение предсказаний для fillers
        for i in range(len(tokens)):
            if is_pause[i] == 1:
                clean = re.sub(r'[^\w]', '', tokens[i].lower())
                clean_alpha = re.sub(r'[^а-яёa-z]', '', clean)

                # Проверка на fillers с контекстом
                if clean_alpha in ['короче', 'ну', 'вот', 'типа', 'значит', 'кажется', 'вернее']:
                    # Если предыдущее слово не запятая и не filler - убираем паузу
                    if i > 0:
                        prev_token = tokens[i-1]
                        prev_clean = re.sub(r'[^\w]', '', prev_token.lower())
                        prev_clean_alpha = re.sub(r'[^а-яёa-z]', '', prev_clean)

                        # Если предыдущее слово не контекстное - убираем паузу
                        if not (prev_clean_alpha in ['ну', 'вот', 'типа', 'значит', 'кажется', 'вернее'] or
                               prev_token.endswith(',')):
                            is_pause[i] = 0
                    else:
                        # Первое слово - убираем паузу для fillers
                        is_pause[i] = 0

        durations = np.zeros(len(tokens), dtype=float)
        pos_idx = np.where(is_pause == 1)[0]
        if len(pos_idx) > 0:
            preds = self.duration_reg.predict(feats[pos_idx])
            durations[pos_idx] = preds

        # Обрезка значений для MAE
        N = len(tokens)
        is_punct_pred = [len(t) > 0 and t[-1] in PUNCT_CHARS for t in tokens]
        for idx in pos_idx:
            if idx < N:
                next_punct_idx = next((j for j in range(idx + 1, N) if is_punct_pred[j]), N)
                phrase_len = next_punct_idx - idx

                # Уточним диапазоны:
                if phrase_len <= 1:
                    durations[idx] = np.clip(durations[idx], 0.02, 0.08)
                elif phrase_len <= 3:
                    durations[idx] = np.clip(durations[idx], 0.05, 0.15)
                elif phrase_len <= 6:
                    durations[idx] = np.clip(durations[idx], 0.10, 0.30)
                else:
                    durations[idx] = np.clip(durations[idx], 0.15, 0.80)

        return is_pause, durations

def calc_metrics(sub_df, name):
    """Вычисление метрик качества"""
    if sub_df['is_pause_after'].sum() == 0 or sub_df['is_pause_hat'].sum() == 0:
        print(f"{name}: PRC:0.0 REC:0.0 F1:0.0 MAE:NaN\n"); return

    rec = recall_score(sub_df.is_pause_after, sub_df.is_pause_hat, zero_division=0)
    prc = precision_score(sub_df.is_pause_after, sub_df.is_pause_hat, zero_division=0)
    f1 = f1_score(sub_df.is_pause_after, sub_df.is_pause_hat, zero_division=0)

    tp_mask = (sub_df.is_pause_after == 1) & (sub_df.is_pause_hat == 1)
    mae = mean_absolute_error(
        sub_df.loc[tp_mask, 'pause_duration'],
        sub_df.loc[tp_mask, 'pause_duration_hat']
    ) if tp_mask.sum() > 0 else np.nan

    print(f"{name}: PRC:{prc:.3f} REC:{rec:.3f} F1:{f1:.3f} MAE:{mae:.4f}\n")

def test_pause_predictor():
    """Основная функция тестирования"""
    print("Loading data...")
    pause_df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)

    pp = PausePredictor()
    pp.train(pause_df[pause_df.set == 'train'])

    print("Making predictions...")
    unique_ids = pause_df['id'].unique()
    is_pause_hat = np.zeros(len(pause_df), dtype=int)
    pause_dur_hat = np.zeros(len(pause_df), dtype=float)

    # Сбор ошибок для анализа
    errors = []

    for uid in tqdm.tqdm(unique_ids, desc="Predicting"):
        group = pause_df.loc[pause_df['id'] == uid]
        is_last = group['is_last_word'].values

        update_indices = group.index[is_last == 0].to_numpy()
        tokens_for_pred = group['label_raw'].values[is_last == 0]
        true_labels = group['is_pause_after'].values[is_last == 0]

        if len(tokens_for_pred) == 0: continue

        pred_pause, pred_dur = pp.predict(tokens_for_pred)
        is_pause_hat[update_indices] = pred_pause
        pause_dur_hat[update_indices] = pred_dur

        # Сравнение с истинными значениями
        for word, true_val, pred_val, pred_d in zip(tokens_for_pred, true_labels, pred_pause, pred_dur):
            if true_val != pred_val:
                err_type = "FP (Ложная пауза)" if pred_val == 1 else "FN (Пропущена пауза)"
                errors.append({
                    'Word': word,
                    'ID': uid,
                    'Set': group['set'].iloc[0],
                    'True': true_val,
                    'Pred': pred_val,
                    'Type': err_type,
                    'Duration': pred_d
                })

    pause_df['is_pause_hat'] = is_pause_hat
    pause_df['pause_duration_hat'] = pause_dur_hat

    # Метрики
    calc_metrics(pause_df[(pause_df.set == 'train') & (pause_df.is_last_word == 0)], "TRAIN")
    calc_metrics(pause_df[(pause_df.set == 'test') & (pause_df.is_last_word == 0)], "TEST")

    # Анализ ошибок
    if len(errors) > 0:
        print("\n" + "="*40)
        print("АНАЛИЗ ОШИБОК ПРЕДСКАЗАНИЯ")
        print("="*40)
        df_err = pd.DataFrame(errors)
        print(f"Всего ошибок: {len(errors)}")

        fp_df = df_err[df_err['Type'] == "FP (Ложная пауза)"]
        if not fp_df.empty:
            print(f"\nЛожные паузы (FP): {len(fp_df)} предсказаний")
            print("Топ-10 слов, где модель ставит ПАУЗУ, где её нет:")
            print(fp_df['Word'].value_counts().head(10))
            print("\nПримеры (ID | Слово | Предсказанная дл-ть):")
            print(fp_df[['ID', 'Word', 'Duration']].head(5).to_string(index=False))
        fn_df = df_err[df_err['Type'] == "FN (Пропущена пауза)"]
        if not fn_df.empty:
            print(f"\nПропущенные паузы (FN): {len(fn_df)} предсказаний")
            print("Топ-10 слов, где модель НЕ СТАВИТ ПАУЗУ, где она есть:")
            print(fn_df['Word'].value_counts().head(10))
            print("\nПримеры (ID | Слово):")
            print(fn_df[['ID', 'Word']].head(5).to_string(index=False))

    else:
        print("\nОшибок нет!")

if __name__ == '__main__':
    test_pause_predictor()