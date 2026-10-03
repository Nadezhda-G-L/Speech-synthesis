"""Build pause predictor training data — lab 2.
Joins the lab 1 normalized metadata with the MFA word alignment and writes one row per
word to `data/RUSLAN_pause_metadata.csv`::
    id|label|label_raw|duration|is_last_word|is_pause_after|pause_duration|set
Utterances whose id ends in 0 or 5 go to `test`, the rest to `train`.
Run from the lab directory::
    python prepare_training_data.py
"""
import csv
import glob
import numpy as np
import os
import pandas as pd
from praatio import textgrid
import tqdm

# --- Конфигурация путей ---
# Путь к нормализованным метаданным из Лабораторной 1
RUSLAN_META = '../../data/ruslan_dataset/metadata_RUSLAN_22200_normalized.csv'
# Путь к папке с файлами разметки (TextGrid) от MFA
ALIGN_DIR = '../../data/RUSLAN_align_v2/'
# Куда сохранится итоговый CSV для обучения модели
RESULT_PATH = '../../data/ruslan_dataset/RUSLAN_pause_metadata.csv'

def read_text_grids(ruslan: pd.DataFrame, align_root: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:   
    """Читает MFA TextGrid файлы для каждой утвари из `ruslan`.
    
    Args:
        ruslan: Метаданные с колонками `id` и `nrm`.
        align_root: Директория с файлами `{id}.TextGrid`.
    Returns:
        Word intervals (`label`, `duration`, `id`; для пауз label = ``""``), 
        phone intervals (аналогично; паузы = ``"<SIL>"``), и один 
        склейка фонем на строку метаданных.
    """
    word_docs = []
    phn_docs = []
    phoneme_sequences = []
    
    for wave_id, nrm in tqdm.tqdm(ruslan[['id', 'nrm']].values):
        try:
            # includeEmptyIntervals=True, чтобы читать паузы (которые могут быть нулевой длительности)
            tg = textgrid.openTextgrid(os.path.join(align_root, wave_id+'.TextGrid'), includeEmptyIntervals=True)
        except Exception as e:
            print(f"Ошибка чтения TextGrid для {wave_id}: {e}")
            phoneme_sequences.append('')
            continue
            
        # Обработка тиров - проверяем тип tg.tiers
        try:
            # Проверяем, что tg.tiers - это кортеж с тирами
            if hasattr(tg, 'tiers'):
                tiers = tg.tiers
                # Если это кортеж, преобразуем в словарь
                if isinstance(tiers, tuple) and len(tiers) >= 2:
                    # Предполагаем, что первый элемент - тир слов, второй - тир фонем
                    words_tier = tiers[0]
                    phones_tier = tiers[1] if len(tiers) > 1 else None
                else:
                    # Если это не кортеж или слишком короткий, пробуем другой способ
                    words_tier = None
                    phones_tier = None
                    # Попробуем получить тиры по имени
                    if hasattr(tg, 'tierNames'):
                        tier_names = tg.tierNames
                        if 'words' in tier_names:
                            words_tier = tg.tiers[tier_names.index('words')]
                        if 'phones' in tier_names:
                            phones_tier = tg.tiers[tier_names.index('phones')]
            else:
                words_tier = None
                phones_tier = None
                
        except Exception as e:
            print(f"Ошибка при обработке тиров для {wave_id}: {e}")
            words_tier = None
            phones_tier = None

        # Если не удалось получить тиры через стандартный способ, пробуем альтернативный
        if words_tier is None:
            # Попробуем получить тиры по индексам
            try:
                if hasattr(tg, 'tiers') and len(tg.tiers) >= 2:
                    words_tier = tg.tiers[0]
                    phones_tier = tg.tiers[1]
                elif hasattr(tg, 'tiers') and len(tg.tiers) == 1:
                    words_tier = tg.tiers[0]
                    phones_tier = None
                else:
                    words_tier = None
                    phones_tier = None
            except:
                words_tier = None
                phones_tier = None

        # Если тиры не найдены, пропускаем
        if words_tier is None:
            print(f"Пропуск {wave_id}: тиры не найдены")
            phoneme_sequences.append('')
            continue

        # Собираем данные по словам
        try:
            if hasattr(words_tier, 'entries'):
                for i in words_tier.entries:
                    word_docs.append({'label': i.label, 'duration': i.end - i.start, 'id': wave_id})
            elif hasattr(words_tier, '__iter__'):
                for i in words_tier:
                    word_docs.append({'label': i.label, 'duration': i.end - i.start, 'id': wave_id})
        except Exception as e:
            print(f"Ошибка при обработке слов для {wave_id}: {e}")

        # Собираем данные по фонемам
        try:
            if phones_tier is not None and hasattr(phones_tier, 'entries'):
                phoneme_sequences.append(' '.join([p.label for p in phones_tier.entries]))
                for i in phones_tier.entries:
                    if i.label == '':
                        phn_docs.append({'label': '<SIL>', 'duration': i.end - i.start, 'id': wave_id})
                    else:
                        phn_docs.append({'label': i.label, 'duration': i.end - i.start, 'id': wave_id})
            elif phones_tier is not None and hasattr(phones_tier, '__iter__'):
                phoneme_sequences.append(' '.join([p.label for p in phones_tier]))
                for i in phones_tier:
                    if i.label == '':
                        phn_docs.append({'label': '<SIL>', 'duration': i.end - i.start, 'id': wave_id})
                    else:
                        phn_docs.append({'label': i.label, 'duration': i.end - i.start, 'id': wave_id})
            else:
                phoneme_sequences.append('')
        except Exception as e:
            print(f"Ошибка при обработке фонем для {wave_id}: {e}")
            phoneme_sequences.append('')
                    
    word_df = pd.DataFrame(word_docs)
    phn_df = pd.DataFrame(phn_docs)
    return word_df, phn_df, phoneme_sequences

def align_text_and_textgrid(tokens: pd.DataFrame, text: str) -> pd.DataFrame:
    """Привязывает оригинальную форму текста (с регистром и пунктуацией) к токенам выравнивания.
    
    MFA возвращает нижний регистр без пунктуации. Эта функция восстанавливает 
    исходный вид слова из `label_raw`.
    
    Args:
        tokens: Интервалы слов.
        text: Нормализованный текст.
    Returns:
        `tokens` с добавленной колонкой `label_raw`.
    """
    raw_tokens = []
    text_lower = text.lower()
    previous_word = -1
    
    for t, d, i in tokens[['label', 'duration', 'id']].values:
        if t == '':  # Если токен пустой, это пауза
            raw_tokens.append('<SIL>')
            continue
            
        # Пытаемся найти слово в тексте (ищем в нижнем регистре)
        splits = text_lower.split(t, maxsplit=1)
        
        if len(splits) == 1:  
            # Если слово не найдено в тексте, это значит, что текст и аудио расходятся
            # Для устойчивости считаем это паузой/мусором, но не останавливаем весь процесс
            raw_tokens.append('<SIL>') 
            continue
            
        # Вычисляем, какой кусок исходного текста соответствует этому токену
        # Важно: мы берем текст до точки разреза и добавляем само слово
        # Это позволяет сохранить пунктуацию (запятые, точки) "внутри" или "после" слова
        current_length = len(splits[0] + t)
        
        if previous_word == -1:
            # Для первого слова берем текст от самого начала
            raw_tokens.append(text[:current_length].strip())
        else:
            # Для последующих слов берем остаток от предыдущего слова
            raw_tokens[previous_word] += text[:len(splits[0])].strip()
            raw_tokens.append(text[len(splits[0]):current_length])
            
        # Сдвигаем текст вперед для следующей итерации
        text_lower = splits[1]
        text = text[len(splits[0] + t):]
        previous_word = len(raw_tokens) - 1
        
    if len(text) and previous_word >= 0:
        # Если в конце осталось что-то (например, точка или многоточие), добавляем к последнему слову
        raw_tokens[previous_word] += text.strip()
        
    tokens['label_raw'] = raw_tokens
    return tokens

def add_pause_labels(align: pd.DataFrame) -> pd.DataFrame:
    """Проставляет метки о паузах после каждого слова.
    
    Args:
        align: Выровненные токены.
    Returns:
        DataFrame с колонками: is_last_word, is_pause_after, pause_duration.
    """
    is_last_word = []
    pause_after = []
    pause_duration = []
    last_word_idx = -1  # Индекс последнего встреченного слова
    
    for idx, (label, dur) in enumerate(align[['label', 'duration']].values):
        if label == '':  
            # Если видим пустой лейбл, значит это пауза.
            # Если перед ней было слово, помечаем это слово как "перед паузой"
            if last_word_idx >= 0:
                pause_after[last_word_idx] = True
                pause_duration[last_word_idx] = dur
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
        else:  
            # Это слово. Паузы после него пока нет.
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
            last_word_idx = idx
            
    # Помечаем последнее слово как финальное в предложении
    if last_word_idx >= 0:
        is_last_word[last_word_idx] = True
        
    align['is_last_word'] = is_last_word
    align['is_pause_after'] = pause_after
    align['pause_duration'] = pause_duration
    return align

def main() -> None:
    """Основная функция: читает данные, выравнивает, проставляет паузы, сохраняет."""
    print("1. Загрузка метаданных...")
    # Читаем файл из Лабораторной 1 (нормализованный)
    ruslan = pd.read_csv(f'{RUSLAN_META}', sep='|', names=['id', 'raw', 'nrm'], quoting=csv.QUOTE_NONE)
    
    print("Чтение файлов TextGrid...")
    word_df, _, _ = read_text_grids(ruslan, ALIGN_DIR)

    # Дополнительная нормализация для улучшения совпадений
    word_df.label = word_df.label.str.replace('‐', '-')  # разные виды дефисов
    word_df.label = word_df.label.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‐', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('’', "'")        # MFA заменяет ’ на '
    ruslan.nrm = ruslan.nrm.str.replace('\\((.*?)\\)', '[bracketed]', regex=True)  
    ruslan.nrm = ruslan.nrm.str.replace('\\<(.*?)\\>', '[bracketed]', regex=True)
    
    # Выравнивание токенов из TextGrid с текстом, включая пунктуацию
    print("Выравнивание токенов с текстом...")
    aligns = []
    for n, i in tqdm.tqdm(ruslan[['nrm', 'id']].values):
        tokens = word_df[word_df.id == i]
        if len(tokens) > 0:
            aligns.append(align_text_and_textgrid(tokens, n))
    
    # Создание меток для обучения предиктора пауз
    print("Проставление меток пауз...")
    aligns = [add_pause_labels(a) for a in aligns]
    
    # Сборка финального DataFrame
    print("Сборка финальной таблицы...")
    pause_df = pd.concat(aligns, ignore_index=True)

    # Удаление токенов пауз и невалидных записей
    pause_df = pause_df[pause_df.label_raw != '<SIL>']  # Паузы уже учтены в метках слов
    pause_df = pause_df[pause_df.label_raw.notna()]    # Отбрасываем сбойные выравнивания
    pause_df = pause_df.reset_index(drop=True)         # Сброс индекса после фильтрации

    # Приведение логических полей к целым числам
    pause_df.is_last_word = pause_df.is_last_word.astype(int)
    pause_df.is_pause_after = pause_df.is_pause_after.astype(int)
    
    # Детерминированное деление на train и test: ID, делящийся на 5 без остатка → test
    print("Деление на обучающую и тестовую выборки...")
    pause_df['set'] = 'train'
    pause_df.loc[pause_df.id.str.split('_', expand=True)[0].astype(int) % 5 == 0, 'set'] = 'test'
    
    # Сохранение итогового датасета
    print(f"Сохранение данных в {RESULT_PATH}")
    pause_df.to_csv(f'{RESULT_PATH}', sep='|', index=False, header=True, quoting=csv.QUOTE_NONE)
    
    # Вывод статистики
    print(f"\nСтатистика подготовленного датасета:")
    print(f"Всего строк: {len(pause_df)}")
    print(f"Обучающая выборка: {len(pause_df[pause_df.set=='train'])}")
    print(f"Тестовая выборка: {len(pause_df[pause_df.set=='test'])}")
    print(f"Пауз отмечено: {pause_df.is_pause_after.sum()} ({pause_df.is_pause_after.mean()*100:.1f}%)")
    print(f"Средняя длительность паузы: {pause_df[pause_df.is_pause_after==1].pause_duration.mean():.3f}с")

if __name__ == '__main__':
    main()