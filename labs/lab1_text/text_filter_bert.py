"""BERT-based Normalized / non-normalized classifier - CORRECTED VERSION"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "./models/rubert_tiny2"  # Локальная модель
DEV_SET_PATH = "data/dev_sentences.csv"
BATCH_SIZE = 16
EPOCHS = 20  
LEARNING_RATE = 2e-5

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
        label = self.labels[idx]
        
        # Добавляем немного предобработки
        if len(text) < 2:
            text = "пустой текст"
            
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextFilterBERT:
    def __init__(self):
        print(f"Loading model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=2
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_trained = False

    def prepare_data(self, texts, labels):
        dataset = TextDataset(texts, labels, self.tokenizer)
        return dataset

    def train(self, train_texts, train_labels):
        """Обучение модели с класс-вейтингом и более тонкой настройкой"""
        
        # Вычисляем веса классов для баланса
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(train_labels), 
                                           y=train_labels)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        train_dataset = self.prepare_data(train_texts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Оптимизатор и scheduler
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Обучение
        self.model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                if i % 20 == 0:
                    print(f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss:.4f}')
        
        self.is_trained = True

    def predict(self, texts):
        """Предсказание для новых текстов"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    str(text).strip(),
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.append(preds.cpu().item())
        
        return predictions

    def evaluate(self, texts, true_labels):
        """Оценка модели"""
        predictions = self.predict(texts)
        f1 = f1_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predictions': predictions
        }

def main():
    # Загрузка данных
    print("Loading data...")
    df = pd.read_csv(DEV_SET_PATH, sep="|", encoding="utf-8")
    
    # Анализ датасета
    print("=== Анализ датасета ===")
    print(f"Общее количество записей: {len(df)}")
    print(f"Нормализованные: {df['is_normalized'].sum()}")
    print(f"Ненормализованные: {len(df) - df['is_normalized'].sum()}")
    print(f"Соотношение: {df['is_normalized'].mean():.2%}")
    
    # Подготовка данных
    texts = df["text"].tolist()
    labels = df["is_normalized"].tolist()
    
    # Разделение на train/test (80/20)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Создание и обучение модели
    print("Initializing BERT model...")
    textfilter = TextFilterBERT()
    
    print("Training model on TRAIN set...")
    textfilter.train(train_texts, train_labels)
    
    # Оценка на TEST выборке
    print("Evaluating model on TEST set...")
    test_results = textfilter.evaluate(test_texts, test_labels)
    
    print(f"\n=== РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ ===")
    print(f"F1 Score: {test_results['f1']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    
    # Оценка на TRAIN выборке
    train_results = textfilter.evaluate(train_texts, train_labels)
    print(f"\n=== РЕЗУЛЬТАТЫ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ ===")
    print(f"F1 Score: {train_results['f1']:.4f}")
    print(f"Precision: {train_results['precision']:.4f}")
    print(f"Recall: {train_results['recall']:.4f}")
    
    # Сохранение модели
    print("\nSaving model...")
    torch.save(textfilter.model.state_dict(), 'bert_model.pth')
    print("Model saved as bert_model.pth")
    
    return test_results

if __name__ == "__main__":
    results = main()