"""BERT-based Normalized / non-normalized classifier - INFERENCE VERSION"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "./models/rubert_tiny2"  # Локальная модель

class TextFilter:
    def __init__(self):
        print(f"Loading model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=2
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # Устанавливаем в режим инференса

    def filter(self, text: str) -> int:
        """Classify a single utterance using BERT model.
        Returns:
            1 if the text is normalized and the utterance can be used for training;
            0 if it contains something the speaker pronounced differently.
        """
        if not text or len(str(text).strip()) < 2:
            return 0
            
        try:
            # Подготовка текста для модели
            inputs = self.tokenizer(
                str(text).strip(),
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Получение предсказания
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=-1)
                prediction = preds.cpu().item()
                
            return prediction
            
        except Exception as e:
            print(f"Error processing text '{text[:50]}...': {e}")
            return 0  # По умолчанию отбрасываем

# Функция для загрузки модели из файла
def load_trained_model(model_path="./bert_model.pth"):
    """Загрузка предобученной модели"""
    model = TextFilter()
    model.model.load_state_dict(torch.load(model_path, map_location=model.device))
    model.model.eval()
    return model

if __name__ == "__main__":
    # Для тестирования
    print("Testing BERT filter...")
    filter_instance = TextFilter()
    print("BERT model loaded successfully!")