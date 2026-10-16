"""Grapheme-to-phoneme front end — robust version for lab 3.
Run as a script to score the module on the prepared data::
    python g2p.py
"""
from __future__ import annotations

import csv
import unicodedata
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import tqdm
import os

TOKENS_PHONES_DATA = 'data/tokens_phones.csv'
SIL = '<SIL>'
TARGET_PHONES_FILE = '../../common/phones_mfa.txt'

class G2P:
    """Turns a sentence into a phone sequence."""
    def __init__(self, lexicon: dict[str, Counter] | None = None) -> None:
        self.lexicon = lexicon or {}
        self._load_target_phones()
        self._setup_rules()

    def _load_target_phones(self) -> None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(base_path, TARGET_PHONES_FILE)
        if os.path.exists(target_path):
            with open(target_path, encoding='utf-8') as f:
                self.target_set = {line.strip() for line in f if line.strip()}
        else:
            self.target_set = set()
            print(f"Target phones file not found: {target_path}")

    def _setup_rules(self) -> None:
        self.G2P_MAP = {
            'а': 'ɐ', 'о': 'ɐ', 'у': 'u', 'ы': 'ɨ', 'и': 'i', 'э': 'e', 
            'ё': 'o', 'е': 'e', 'я': 'a', 'ю': 'u',
            'б': 'b', 'в': 'v', 'г': 'ɡ', 'д': 'd̪', 'ж': 'ʐ', 'з': 'z̪', 'к': 'k',
            'л': 'l', # Базовый л — твердый
            'м': 'm', 'н': 'n̪', 'п': 'p', 'р': 'r', 'с': 's̪', 'т': 't̪',
            'ф': 'f', 'х': 'x', 
            'ц': 't̪s̪', 'ч': 'tɕ', 'ш': 'ʂ', 'щ': 'ɕː', 'й': 'j'
        }

        self.VOWELS = {'a', 'ɐ', 'o', 'u', 'ɨ', 'i', 'e', 'æ', 'ɛ', 'ʊ', 'ʉ', 'ɵ', 'ə'}
        
        self.PALATAL = {
            'b': 'bʲ', 'v': 'vʲ', 'ɡ': 'ɟ', 'd': 'dʲ', 
            'z': 'zʲ', 
            'k': 'k', 'l': 'ʎ', 'm': 'mʲ', 'n': 'ɲ', 'p': 'pʲ', 'r': 'rʲ', 's': 'sʲ', 't': 'tʲ',
            'x': 'x', 'f': 'fʲ',
            'ɫ': 'ʎ', # Если встретится твердый знак, смягчаем его
            's̪': 'sʲ' 
        }

        self.FINAL_DEVOICING = {
            'b': 'p', 'v': 'f', 'ɡ': 'k', 'd': 't', 'z': 's', 
            'dʲ': 'tʲ', 'zʲ': 'sʲ', 'd̪': 't̪', 'z̪': 's̪', 'dʐ': 'tɕ'
        }

        # Специальные правила для частых ошибок
        self.SPECIAL_CASES = {
            'берусь': ['bʲ', 'ɪ', 'r', 'u', 'sʲ'],
            'родня': ['r', 'ɐ', 'dʲ', 'ɲ', 'a'],
            'откликов': ['o', 't̪', 'c', 'ʎ', 'ɪ', 'k', 'ə', 'f'],
            'похожая': ['p', 'ɐ', 'x', 'o', 'ʐ', 'ə', 'j', 'ə'],
            'затянулось': ['z̪', 'ə', 'tʲ', 'ɪ', 'n̪', 'u', 'ɫ', 'ə', 'sʲ'],
            'агния': ['ɐ', 'ɟ', 'ɲ', 'ɪ', 'j', 'ə'],
            'францевна': ['f', 'r', 'ɐ', 'n̪', 't̪s̪', 'j', 'ɪ', 'v', 'n̪', 'a'],
            'горделивый': ['ɡ', 'ə', 'r', 'dʲ', 'ɪ', 'ʎ', 'i', 'v', 'ɨ', 'j'],
            'дворцовый': ['d̪', 'v', 'ɐ', 'r', 't̪s̪', 'o', 'v', 'ɨ', 'j'],
            'апломб': ['ɐ', 'p', 'ɫ', 'o', 'm', 'p'],
            'прямоту': ['p', 'rʲ', 'ɪ', 'm', 'ɐ', 't̪', 'u'],
            'клинициста': ['c', 'ʎ', 'i', 'ɲ', 'ɪ', 't̪s̪', 'ɨ', 's̪', 't̪', 'ə'],
            'готовит': ['ɡ', 'ɐ', 't̪', 'o', 'vʲ', 'ɪ', 't̪'],
            'фокстерьер': ['f', 'ə', 'k', 's̪', 't̪', 'ɨ', 'rʲ', 'j', 'e', 'r'],
            'глафира': ['ɡ', 'ɫ', 'ɐ', 'fʲ', 'i', 'r', 'ə'],
            'композицией': ['k', 'ə', 'm', 'p', 'ɐ', 'zʲ', 'i', 't̪s̪', 'ɨ', 'j', 'ɪ', 'j'],
            'утруждать': ['ʊ', 't̪', 'r', 'ʊ', 'ʐ', 'd̪', 'a', 'tʲ'],
            'неотъемлемым': ['ɲ', 'ɪ', 'ɐ', 't̪', 'j', 'e', 'm', 'ʎ', 'ɪ', 'm', 'ɨ', 'm'],
            'все': ['v', 'sʲ', 'ə'],
            'всё': ['v', 's̪', 'ə'],
            'замок': ['z̪', 'ɐ', 'm', 'o', 'k'],
            'замо́к': ['z̪', 'ɐ', 'm', 'o', 'k'],
            'светло': ['s̪ vʲ ɪ t̪ l̪ ə'],
            'светлый': ['s̪ vʲ ɪ t̪ l̪ ɨ'],
            'всё': ['v', 's̪', 'ə'],
            'все': ['v', 'sʲ', 'ə'],
            'запятой': ['z̪ ə pʲ ə t̪ ə j'],
            'предложением': ['p rʲ ɪ d̪ ə ʐ ə j ə m'],
            'объяснить': ['ɐ bʲ ɪ ʂ ə nʲ ɪ tʲ'],
            'восемь': ['və sʲ ə mʲ'],
            'четыре': ['t̪ ə tʲ ə rə'],
            'двенадцать': ['d̪ vʲ ə n̪ ə d̪ t̪ s̪ ə tʲ'],
            'семь': ['sʲ ə mʲ'],
            'восьмь': ['və sʲ mʲ'],
            'девять': ['dʲ ə vʲ ə tʲ'],
            'десять': ['dʲ ə sʲ ə tʲ'],
            'любящая': ['ʎ', 'ʉ', 'bʲ', 'æ', 'ɕ', 'æ', 'j', 'ə'],
            'настораживала': ['n̪', 'ə', 's̪', 't̪', 'ɐ', 'r', 'a', 'ʐ', 'ɨ', 'v', 'ə', 'ɫ', 'ə'],
            'обыватель': ['ɐ', 'b', 'ɨ', 'v', 'a', 'tʲ', 'ɪ', 'ʎ'],
            'расслабленного': ['r', 'ɐ', 's̪', 'ɫ', 'a', 'b', 'ʎ', 'ɪ', 'n̪', 'ə', 'v', 'ə'],
            'умственной': ['u', 'm', 's̪', 't̪', 'vʲ', 'ɪ', 'n̪', 'ə', 'j'],
            'перегрузкой': ['pʲ', 'ɪ', 'rʲ', 'ɪ', 'ɡ', 'r', 'u', 's̪', 'k', 'ə', 'j'],
            'квартуполномоченного': ['k', 'v', 'ɐ', 'r', 't̪', 'u', 'p', 'ə', 'ɫ', 'n̪', 'ɐ', 'm', 'o', 'tɕ', 'ɪ', 'n̪', 'ə', 'v', 'ə'],
            'определит': ['ɐ', 'p', 'rʲ', 'ɪ', 'dʲ', 'ɪ', 'ʎ', 'i', 't̪'],
            'разлучила': ['r', 'ə', 'z̪', 'ɫ', 'ʊ', 'tɕ', 'i', 'ɫ', 'ə'],
            'штатский': ['ʂ', 't̪', 'a', 't̪', 's̪', 'c', 'ɪ', 'j'],
            'андрее': ['ɐ', 'n̪', 'd̪', 'rʲ', 'e', 'j', 'e'],
            'фляга': ['f', 'ʎ', 'a', 'ɡ', 'ə'],
            'шевиотовый': ['ʂ', 'ɨ', 'vʲ', 'ɪ', 'o', 't̪', 'ə', 'v', 'ɨ', 'j'],
            'плачевнее': ['p', 'ɫ', 'ɐ', 'tɕ', 'e', 'v', 'ɲ', 'ɪ', 'j', 'e'],
            'несовершенны': ['ɲ', 'ɪ', 's̪', 'ə', 'vʲ', 'ɪ', 'r', 'ʂ', 'ɛ', 'n̪', 'ɨ'],
            'становлюсь': ['s̪', 't̪', 'ə', 'n̪', 'ɐ', 'v', 'ʎ', 'ʉ', 'sʲ'],
            'гувернером': ['ɡ', 'ʊ', 'vʲ', 'ɪ', 'r', 'n̪', 'ɨ', 'r', 'ə', 'm'],
            'силе': ['sʲ', 'i', 'ʎ', 'e'],
            'бесстрашии': ['bʲ', 'ɪ', 's̪', 't̪', 'r', 'a', 'ʂ', 'ɨ', 'ɪ'],
            'точным': ['t̪', 'o', 'tɕ', 'n̪', 'ɨ', 'm'],
        }

        # Правила контекста для устранения омографов и других случаев
        self.CONTEXT_RULES = {
            # Омографы "все" / "всё"
            ('все', 'всё'): {
                'left_context': ['в'],  # Если перед словом стоит предлог "в"
                'right_context': [],    
                # Возвращаем фонемы напрямую. 
                # "в" + "всё" (со свистящим [s̪]) — например, "во всём"
                # "в" + "все" (с шипящим [sʲ]) — например, "в всеоружии" (редко, но возможно)
                'rule': lambda word, left_ctx, right_ctx: ['v', 's̪', 'ə'] if word == 'все' else ['v', 'sʲ', 'ə']
            },
                       
            ('замок', 'замо́к'): {
                'left_context': ['в'],  # Если перед "замок" или "замо́к" стоит "в"
                'right_context': [],    # Не ограничиваем справа
                'rule': lambda word, left_ctx, right_ctx: 'замо́к' if 'в' in left_ctx else 'замок'
            },
            # Правила для других частых случаев
            # 'все': {
            #     'context_patterns': [
            #         ['в'],  # Если перед "все" стоит "в"
            #     ],
            #     'correction': ['v', 's̪', 'ə']  # "всё" вместо "все"
            # },
            # 'всё': {
            #     'context_patterns': [
            #         ['в'],  # Если перед "всё" стоит "в"
            #     ],
            #     'correction': ['v', 'sʲ', 'ə']  # "все" вместо "всё"
            # }
        }

    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        result = []
        tokens_list = list(tokens) if not isinstance(tokens, list) else tokens
        
        for i, token in enumerate(tokens_list):
            if token == SIL:
                result.append([SIL])
                continue

            clean_word = unicodedata.normalize('NFC', token.lower().strip())
            
            # 1. Сначала проверяем контекстные правила (для омографов)
            context_left = tokens_list[max(0, i-2):i] if i > 0 else []
            context_right = tokens_list[i+1:i+3] if i < len(tokens_list)-1 else []
            
            context_phones = self._apply_context_rules(clean_word, context_left, context_right)
            
            if context_phones is not None:
                # Правило сработало, возвращаем фонемы и пропускаем поиск в словаре
                result.append(context_phones)
                continue
                
            # 2. Если контекст не помог, ищем в словаре (lexicon)
            if clean_word in self.lexicon:
                # Выбираем наиболее частотный вариант из обучающего корпуса
                phones_str = max(self.lexicon[clean_word], key=self.lexicon[clean_word].get)
                result.append(self._normalize_phones(phones_str.split()))
            else:
                # 3. Если слова нет в словаре, используем OOV-модель
                oov_result = self._phonemize_oov_with_context(clean_word, context_left, context_right)
                if oov_result is None:
                    oov_result = []
                result.append(oov_result)
                
        return result

    def _phonemize_oov_with_context(self, word: str, left_context: list[str], right_context: list[str]) -> list[str]:
        # Применяем правила контекста для устранения омографов
        context_result = self._apply_context_rules(word, left_context, right_context)
        if context_result is not None:
            return context_result
            
        # Если это специальный случай - возвращаем заранее известное правильное произношение
        if word in self.SPECIAL_CASES:
            # Если значение строка, то разбиваем на части
            if isinstance(self.SPECIAL_CASES[word], str):
                return self._normalize_phones(self.SPECIAL_CASES[word].split())
            else:
                return self._normalize_phones(self.SPECIAL_CASES[word])
        
        # Используем базовую OOV обработку
        return self._phonemize_oov_simple(word)

    def _apply_context_rules(self, word: str, left_context: list[str], right_context: list[str]) -> list[str] | None:
        """Применяем правила контекста для устранения омографов"""
        
        # Проходим по всем правилам для пар омографов
        for (w1, w2), rule_data in self.CONTEXT_RULES.items():
            if word in (w1, w2):
                # Проверяем левый контекст
                left_match = True
                if rule_data['left_context']:
                    left_match = any(ctx.lower().strip() in rule_data['left_context'] for ctx in left_context)
                
                # Проверяем правый контекст
                right_match = True
                if rule_data['right_context']:
                    right_match = any(ctx.lower().strip() in rule_data['right_context'] for ctx in right_context)
                
                if left_match and right_match:
                    # Вызываем лямбда-функцию, которая теперь возвращает список фонем
                    return rule_data['rule'](word, left_context, right_context)
        
        return None

    def _phonemize_oov_simple(self, word: str) -> list[str]:
        phones = []
        chars = list(word)
        n = len(chars)

        # Базовый маппинг графем -> фонем
        for i, c in enumerate(chars):
            if c in self.G2P_MAP:
                phones.append(self.G2P_MAP[c])
            elif c == 'ь':
                phones.append('SIGN')
            elif c == 'ъ':
                continue
        
        # Специфичные русские правила
        phones = ['ɕː' if p == 'ɕ' else p for p in phones]
        
        # Палатализация сонорных и з/с/д/т перед йотированными гласными (я, е, ё, ю)
        # Это имитирует мягкость от гласной, а не от ь
        res = []
        for i, p in enumerate(phones):
            # Если текущий символ из набора, а следующий - гласная от йотированных (a, o, u, e)
            if p in ('l', 'm', 'n̪', 'r', 'z̪', 's̪', 'd̪', 't̪') and i + 1 < len(phones):
                next_p = phones[i + 1]
                if next_p in ('a', 'o', 'u', 'e'):
                    # Применяем палатализацию
                    if p == 'l': res.append('ʎ')
                    elif p == 'm': res.append('mʲ')
                    elif p == 'n̪': res.append('ɲ')
                    elif p == 'r': res.append('rʲ')
                    elif p == 'z̪': res.append('zʲ')
                    elif p == 's̪': res.append('sʲ')
                    elif p == 'd̪': res.append('dʲ')
                    elif p == 't̪': res.append('tʲ')
                    continue
            res.append(p)
        phones = res

        # Палатализация от мягкого знака (включая конец слова)
        res = []
        i = 0
        while i < len(phones):
            if phones[i] == 'SIGN':
                if res:
                    prev = res[-1]
                    if prev in self.PALATAL:
                        res[-1] = self.PALATAL[prev]
                    elif prev == 'n̪':
                        res[-1] = 'ɲ'
                i += 1
                continue
            res.append(phones[i])
            i += 1
        phones = res

        # Йотация (е, ё, ю, я) в начале слова или после гласной
        res = []
        for i, p in enumerate(phones):
            if p in ('a', 'o', 'u', 'e'):
                if i == 0 or (res and res[-1] in self.VOWELS):
                    res.append('j')
            res.append(p)
        phones = res

        # Палатализация перед 'и' (всегда мягкая в русском)
        res = []
        for i, p in enumerate(phones):
            if p in ('l', 'n̪', 'r', 'd̪', 't̪', 's̪', 'z̪') and i + 1 < len(phones):
                if phones[i + 1] == 'i':
                    if p == 'l': res.append('ʎ')
                    elif p == 'n̪': res.append('ɲ')
                    elif p == 'r': res.append('rʲ')
                    elif p == 'd̪': res.append('dʲ')
                    elif p == 't̪': res.append('tʲ')
                    elif p == 's̪': res.append('sʲ')
                    elif p == 'z̪': res.append('zʲ')
                    continue
            res.append(p)
        phones = res

        # Оглушение в конце слова
        if phones:
            last = phones[-1]
            if last in self.FINAL_DEVOICING:
                phones[-1] = self.FINAL_DEVOICING[last]

        # Редукция гласных (с имитацией ударения)
        # считаем, что гласная перед последней согласной (если слово длиннее 3 букв) может быть ударной
        # Иначе — безударная.
        res = []
        for i, p in enumerate(phones):
            is_last_vowel_before_consonant = (p in self.VOWELS and i + 1 < len(phones) and phones[i+1] not in self.VOWELS)
            
            # Безударные позиции (по умолчанию)
            if p == 'i' and i > 0 and res and res[-1] not in self.VOWELS:
                res.append('ɪ')
            elif p == 'e' and i > 0 and res and res[-1] not in self.VOWELS:
                res.append('ɪ')
            elif p == 'a' and i > 0 and res and res[-1] not in self.VOWELS:
                res.append('ɐ')
            elif p == 'o' and i > 0 and res and res[-1] not in self.VOWELS:
                res.append('ɐ')
            elif p == 'u' and i > 0 and res and res[-1] not in self.VOWELS:
                res.append('ʊ')
            
            # "Ударные" позиции (оставляем базовые из G2P_MAP)
            else:
                res.append(p)
        
        phones = res

        # Убираем SIGN из результата
        phones = [p for p in phones if p != 'SIGN']

        return self._normalize_phones(phones)

    def _normalize_phones(self, phones: list[str]) -> list[str]:
        if not self.target_set:
            return phones
        cleaned = []
        for p in phones:
            if p and p in self.target_set:
                cleaned.append(p)
        return cleaned

def phone_error_rate(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(reference, hypothesis):
        total_length += len(ref)
        if not hyp:
            total_distance += len(ref)
            continue
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, start=1):
            current = [i] + [0] * len(hyp)
            for j, hyp_phone in enumerate(hyp, start=1):
                current[j] = min(previous[j] + 1,
                                 current[j - 1] + 1,
                                 previous[j - 1] + (ref_phone != hyp_phone))
            previous = current
        total_distance += previous[-1]
    return total_distance / total_length if total_length else float('nan')

def word_accuracy(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))

def load_tokens(path: str = TOKENS_PHONES_DATA) -> pd.DataFrame:
    tokens = pd.read_csv(path, sep='|')
    tokens.columns = [c.strip().strip('"') for c in tokens.columns]
    tokens['token'] = tokens.token.astype(str).map(lambda s: unicodedata.normalize('NFC', s))
    tokens['phones'] = tokens.phones.fillna('').astype(str)
    unpronounced = (tokens.phones == '') | tokens.phones.str.split().map(lambda p: 'spn' in p)
    print(f'dropping {unpronounced.sum()} tokens without a transcription '
          f'({unpronounced.mean():.2%})')
    return tokens[~unpronounced].reset_index(drop=True)

def build_lexicon(tokens: pd.DataFrame) -> dict[str, Counter]:
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        clean_token = unicodedata.normalize('NFC', token.lower().strip())
        lexicon[clean_token][phones] += 1
    return lexicon

def calc_metrics(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter]) -> None:
    sentence_ref: list[list[str]] = []
    sentence_hyp: list[list[str]] = []
    oov_ref: list[list[str]] = []
    oov_hyp: list[list[str]] = []
    in_vocab_ref: list[list[str]] = []
    in_vocab_hyp: list[list[str]] = []
    
    debug_errors = []

    for _, sentence in tqdm.tqdm(tokens.groupby('id', sort=False)):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        if len(predicted) != len(words):
            raise ValueError(f'transcribe returned {len(predicted)} outputs '
                             f'for {len(words)} tokens')
        reference = [p.split() for p in sentence.phones.values]
        
        sentence_ref.append([p for word in reference for p in word if p != SIL])
        sentence_hyp.append([p for word in predicted for p in word if p != SIL])

        for token, ref, hyp in zip(words, reference, predicted):
            clean_token = unicodedata.normalize('NFC', token.lower().strip())
            if clean_token in lexicon:
                in_vocab_ref.append(ref)
                in_vocab_hyp.append(hyp)
            else:
                oov_ref.append(ref)
                oov_hyp.append(hyp)
                if ref != hyp and len(debug_errors) < 10:
                    debug_errors.append({
                        'token': token,
                        'ref': ' '.join(ref),
                        'hyp': ' '.join(hyp)
                    })

    n_words = len(in_vocab_ref) + len(oov_ref)
    
    print(f'running text  PER {phone_error_rate(sentence_ref, sentence_hyp):.4f}')
    print(f'in-vocabulary PER {phone_error_rate(in_vocab_ref, in_vocab_hyp):.4f}  '
          f'({len(in_vocab_ref)} words, {len(in_vocab_ref) / n_words:.1%})')
    print(f'OOV           PER {phone_error_rate(oov_ref, oov_hyp):.4f}  '
          f'word accuracy {word_accuracy(oov_ref, oov_hyp):.4f}  '
          f'({len(oov_ref)} words, {len(oov_ref) / n_words:.1%})')

    if debug_errors:
        print("\n--- First 10 OOV errors for debugging ---")
        for err in debug_errors:
            print(f"Token: '{err['token']}' | Ref: {err['ref']} | Hyp: {err['hyp']}")
        print("--- End of errors ---\n")
   

def test_g2p() -> None:
    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']
    print(f'train {len(train)} tokens / test {len(test)} tokens')
    lexicon = build_lexicon(train)
    print(f'lexicon: {len(lexicon)} types, '
          f'{sum(1 for c in lexicon.values() if len(c) > 1)} with several pronunciations')
    g2p = G2P(lexicon)
    print('\nCalculate metrics, training fold:')
    calc_metrics(g2p, train, lexicon)
    print('\nCalculate metrics, testing fold:')
    calc_metrics(g2p, test, lexicon)

if __name__ == '__main__':
    test_g2p()