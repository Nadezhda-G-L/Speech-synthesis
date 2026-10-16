"""
Grapheme-to-phoneme front end — Hybrid version (Lexicon + PyTorch Seq2Seq).
Run as a script to score the module on the prepared data::
    python g2p.py
"""
from __future__ import annotations

import csv
import unicodedata
import random
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENS_PHONES_DATA = 'data/tokens_phones.csv'
SIL = '<SIL>'
TARGET_PHONES_FILE = '../../common/phones_mfa.txt'

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        # Сцепляем последнее скрытое состояние (прямое и обратное)
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return nn.functional.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.9): # Изменено: 0.9
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        
        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs

def build_vocab(items: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    """Создание словаря соответствия 'символ -> индекс' и обратно."""
    item_to_idx = {item: idx for idx, item in enumerate(sorted(set([i for sublist in items for i in sublist])))}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    return item_to_idx, idx_to_item

class G2PDataset(Dataset):
    """Набор данных для обучения модели G2P."""
    def __init__(self, data, char2idx, phone2idx, sos_token, eos_token, pad_token):
        self.data = data
        self.char2idx = char2idx
        self.phone2idx = phone2idx
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        left, word, right, phones = self.data[idx]
        # Контекст: левое слово, текущее слово, правое слово
        context_chars = list((left + " " + word + " " + right).lower())
        src_indexes = [self.char2idx.get(tok, self.char2idx['<unk>']) for tok in context_chars]
        src_tensor = torch.LongTensor(src_indexes)

        trg_indexes = [self.sos_token] + [self.phone2idx.get(p, self.phone2idx['<unk>']) for p in phones] + [self.eos_token]
        trg_tensor = torch.LongTensor(trg_indexes)
        
        return src_tensor, trg_tensor

def collate_fn(batch, char2idx, phone2idx):
    """Сборка батча и дополнение последовательностей (padding)."""
    src_batch, trg_batch = zip(*batch)
    
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=char2idx['<pad>'])
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, padding_value=phone2idx['<pad>'])
    
    return src_batch, trg_batch


class G2P:
    """
        Модуль преобразования текста в фонемы (Grapheme-to-Phoneme).
        Использует гибридный подход:
        1. Поиск в словаре (lexicon) для известных слов.
        2. Нейросетевая модель (Seq2Seq) для неизвестных слов (OOV).
    """
    def __init__(self, lexicon: dict[str, Counter] | None = None, model_path: str = 'models/g2p_seq2seq.pt') -> None:
        """
        Args:
            lexicon: Словарь произношений {word: Counter({phones: count})}.
            model_path: Путь для сохранения/загрузки нейросети.
        """
        self.lexicon = lexicon or {}
        self._load_target_phones()
        self.model_path = model_path
        
        self.nn_model = None
        self.datasets_ready = False
        self.dataset = None

    def _load_target_phones(self) -> None:
        """Загрузка целевого инвентаря фонем из файла."""
        base_path = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(base_path, TARGET_PHONES_FILE)
        if os.path.exists(target_path):
            with open(target_path, encoding='utf-8') as f:
                self.target_set = {line.strip() for line in f if line.strip()}
        else:
            self.target_set = set()
            print(f"Файл не найден: {target_path}")

    def _prepare_nn_data(self, tokens: pd.DataFrame):
        """
        Подготовка выборки для обучения нейросети.
        Извлекает контекст (соседние слова) для каждого токена.
        """
        nn_data = []
        
        for sentence_id, sentence_group in tokens.groupby('id', sort=False):
            sentence_group = sentence_group.sort_values('token_id')
            sentence_tokens = sentence_group['token'].tolist()
            sentence_phones = sentence_group['phones'].tolist()
            
            for i, (token, phones_str) in enumerate(zip(sentence_tokens, sentence_phones)):
                clean_token = unicodedata.normalize('NFC', token.lower().strip())
                
                left = sentence_tokens[i-1].lower() if i > 0 else '<pad>'
                right = sentence_tokens[i+1].lower() if i < len(sentence_tokens)-1 else '<pad>'
                
                phones = phones_str.split()
                nn_data.append((left, clean_token, right, phones))
        
        if not nn_data:
            raise ValueError("NN Dataset is empty.")
        # Создание словарей    
        all_chars = [list(w[0] + w[1] + w[2]) for w in nn_data]
        all_phones = [w[3] for w in nn_data]
        
        self.char2idx, self.idx2char = build_vocab(all_chars)
        self.phone2idx, self.idx2phone = build_vocab(all_phones)
        
        self.char2idx['<pad>'] = len(self.char2idx)
        self.char2idx['<unk>'] = len(self.char2idx)
        self.phone2idx['<pad>'] = len(self.phone2idx)
        self.phone2idx['<sos>'] = len(self.phone2idx)
        self.phone2idx['<eos>'] = len(self.phone2idx)
        self.phone2idx['<unk>'] = len(self.phone2idx)
        
        self.dataset = G2PDataset(nn_data, self.char2idx, self.phone2idx, 
                                  self.phone2idx['<sos>'], self.phone2idx['<eos>'], self.phone2idx['<pad>'])
        self.datasets_ready = True

    def _init_nn_model(self, force_retrain: bool = False):
        """
        Инициализация или загрузка обученной модели.
        Args:
            force_retrain: Если True, переобучает модель с нуля.
        """
        if self.nn_model is None or force_retrain:
            if not self.datasets_ready:
                raise RuntimeError("NN Data not prepared.")
            
            input_dim = len(self.char2idx)
            output_dim = len(self.phone2idx)
            
            enc_emb_dim = 64
            enc_hid_dim = 256 # Изменено: 256
            dec_hid_dim = 256 # Изменено: 256
            dec_emb_dim = 64
            dropout = 0.45 # Изменено: 0.45
            
            attn = Attention(enc_hid_dim, dec_hid_dim)
            enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, dropout)
            dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dropout, attn)
            
            self.nn_model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
            
            if os.path.exists(self.model_path) and not force_retrain:
                self.nn_model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
                self.nn_model.eval()
                print("Loaded pre-trained OOV model.")
            else:
                self._train_nn_model()

    def _train_nn_model(self):
        """Тренировка модели на OOV-словах с использованием внимания."""
        print("Training OOV Seq2Seq model...")
        optimizer = optim.Adam(
            self.nn_model.parameters(), 
            lr=0.0001,
            weight_decay=1e-5 
        )
        criterion = nn.CrossEntropyLoss(ignore_index=self.phone2idx['<pad>'])
        
        train_dataloader = DataLoader(
            self.dataset, 
            batch_size=32, 
            shuffle=True, 
            collate_fn=lambda batch: collate_fn(batch, self.char2idx, self.phone2idx)
        )
        
        val_dataloader = DataLoader(
            self.dataset, 
            batch_size=32,
            collate_fn=lambda batch: collate_fn(batch, self.char2idx, self.phone2idx)
        )
        
        best_valid_loss = float('inf')
        
        for epoch in range(100):
            self.nn_model.train()
            epoch_loss = 0
            
            for i, (src, trg) in enumerate(train_dataloader):
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                
                optimizer.zero_grad()
                output = self.nn_model(src, trg)
                
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                
                loss = criterion(output, trg)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm=2.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            self.nn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, trg in val_dataloader:
                    src, trg = src.to(DEVICE), trg.to(DEVICE)
                    output = self.nn_model(src, trg, 0.0)
                    
                    output_dim = output.shape[-1]
                    output = output[1:].view(-1, output_dim)
                    trg = trg[1:].view(-1)
                    val_loss += criterion(output, trg).item()

            # Сохранение лучшей модели
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(self.nn_model.state_dict(), self.model_path)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_dataloader):.4f} | Val Loss: {val_loss/len(val_dataloader):.4f}")
        
        self.nn_model.eval()
        print("Training finished.")

    def _predict_oov(self, left: str, word: str, right: str) -> list[str]:
        """
        Предсказание фонем для слова, отсутствующего в словаре.
        Использует обученную Seq2Seq модель.
        """
        if self.nn_model is None:
            raise RuntimeError("NN Model not initialized.")
            
        self.nn_model.eval()
        with torch.no_grad():
            context = (left + " " + word + " " + right).lower()
            src_indexes = [self.char2idx.get(tok, self.char2idx['<unk>']) for tok in context]
            src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(DEVICE)
            
            encoder_outputs, hidden = self.nn_model.encoder(src_tensor)
            
            trg_indexes = [self.phone2idx['<sos>']]
            
            for t in range(50):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(DEVICE)
                output, hidden, _ = self.nn_model.decoder(trg_tensor, hidden, encoder_outputs)
                
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                
                if pred_token == self.phone2idx['<eos>'] or len(trg_indexes) > 50:
                    break
            
            predicted_phones = [self.idx2phone[idx] for idx in trg_indexes[1:-1]]
            return self._normalize_phones(predicted_phones)

    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        """
        Транскрибирует последовательность токенов.
        Args:
            tokens: Список токенов, возможно включающий <SIL>.
        Returns:
            Список списков фонем для каждого токена.
        """
        # Если модель еще не готова, подготавливаем данные (используем словарь для этого)
        if self.nn_model is None:
            if not self.datasets_ready:
                # Берем одно первое слово для инициализации структуры данных
                temp_tokens = pd.DataFrame({'token': [tokens[0]], 'phones': [['<unk>']]})
                self._prepare_nn_data(temp_tokens)
            self._init_nn_model()
            
        result = []
        tokens_list = list(tokens) if not isinstance(tokens, list) else tokens
        
        for i, token in enumerate(tokens_list):
            if token == SIL:
                result.append([SIL])
                continue

            clean_word = unicodedata.normalize('NFC', token.lower().strip())
             # Словарный поиск (базовое решение)
            if clean_word in self.lexicon:
                phones_str = max(self.lexicon[clean_word], key=self.lexicon[clean_word].get)
                result.append(self._normalize_phones(phones_str.split()))
            else:
                # Нейросетевой предсказатель (для OOV)
                left_ctx = tokens_list[i-1].lower() if i > 0 else ""
                right_ctx = tokens_list[i+1].lower() if i < len(tokens_list)-1 else ""
                
                phones = self._predict_oov(left_ctx, clean_word, right_ctx)
                result.append(phones)
                
        return result

    def _normalize_phones(self, phones: list[str]) -> list[str]:
        """
        Очистка выходной последовательности фонем.
        Оставляет только те, что присутствуют в целевом инвентаре MFA.
        """
        if not self.target_set:
            return phones
        return [p for p in phones if p and p in self.target_set]

def phone_error_rate(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Расчет PER на уровне корпуса (суммируя ошибки)."""
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
    """Доля слов с точным совпадением фонемной транскрипции."""
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))

def load_tokens(path: str = TOKENS_PHONES_DATA) -> pd.DataFrame:
    """Загрузка и фильтрация сырых данных разметки."""
    tokens = pd.read_csv(path, sep='|')
    tokens.columns = [c.strip().strip('"') for c in tokens.columns]
    tokens['token'] = tokens.token.astype(str).map(lambda s: unicodedata.normalize('NFC', s))
    tokens['phones'] = tokens.phones.fillna('').astype(str)
    unpronounced = (tokens.phones == '') | tokens.phones.str.split().map(lambda p: 'spn' in p)
    print(f'dropping {unpronounced.sum()} tokens without a transcription '
          f'({unpronounced.mean():.2%})')
    return tokens[~unpronounced].reset_index(drop=True)

def build_lexicon(tokens: pd.DataFrame) -> dict[str, Counter]:
    """Построение словаря произношений из обучающей выборки."""
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        clean_token = unicodedata.normalize('NFC', token.lower().strip())
        lexicon[clean_token][phones] += 1
    return lexicon

def calc_metrics(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter]) -> None:
    """Расчет метрик для заданной выборки."""
    if not g2p.datasets_ready:
        g2p._prepare_nn_data(tokens)
        g2p._init_nn_model()
        
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
    """Запуск полного пайплайна обучения и оценки."""
    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']
    
    # обрезаем для ускорения обучения
    train_nn_sample = train.sample(n=25000, random_state=42) 
    
    print(f'train {len(train)} tokens / test {len(test)} tokens')
    lexicon = build_lexicon(train)
    print(f'lexicon: {len(lexicon)} types, '
          f'{sum(1 for c in lexicon.values() if len(c) > 1)} with several pronunciations')
    
    g2p = G2P(lexicon)
    
    print('\nCalculate metrics, training fold:')
    calc_metrics(g2p, train_nn_sample, lexicon)
    
    print('\nCalculate metrics, testing fold:')
    calc_metrics(g2p, test, lexicon)

if __name__ == '__main__':
    test_g2p()