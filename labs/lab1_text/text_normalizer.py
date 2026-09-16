"""Russian text normalizer — skeleton for lab 1.
Brings corpus text into a form usable for training a speech synthesizer.
"""
import re
import unicodedata

class TextNormalizer:
    """Normalizes text in Russian.
        "!.."           -> "!"
        "«цитата»"      -> '"цитата"'
        "текст * мусор" -> "текст мусор"
        "де‑факто"      -> "де-факто"      # U+2011 -> ordinary hyphen
    """
    def __init__(self):
        """Prepare the normalizer's resources."""
        # 1. Убираем пробелы перед знаками препинания (спасает фразы типа "Спасибо , что")
        self.space_before_punct = re.compile(r'\s+([.,!?;:])')
        
        # 2. Гарантируем ровно один пробел после знаков препинания (спасает фразы типа "по набережной,и ветер")
        self.space_after_punct = re.compile(r'([.,!?;:])\s+')
        
        # 3. Устранение дубликатов и смешанных последовательностей (!. -> !, ??? -> ?)
        self.multi_punct = re.compile(r'([.!?])\s*([.!?]+)')
        self.dup_punct = re.compile(r'([.!?])\1+')
        
        # 4. Замена всех видов тире и дефисов на обычный "-"
        self.special_hyphens = re.compile(r'[\u2013\u2014\u2011\u2010]')
        
        # 5. Удаление технических мусорных символов
        self.junk = re.compile(r'[*#@&]+')
        
        # 6. Приведение всех видов кавычек к стандартному виду
        self.smart_quotes = re.compile(r"[\u201C\u201D\u00AB\u00BB\u201A\u201E\u2018\u2019]")

    def normalize(self, text: str) -> str:
        """Normalize a single line."""
        if not text:
            return text
            
        # 1. Нормализуем Unicode к формату NFC (объединяет е+ударение, сохраняет ё)
        text = unicodedata.normalize("NFC", text)

        # 2. Убираем «нулевую ширину» и неразрывные пробелы (частая причина фильтрации)
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        text = text.replace('\xa0', ' ')

        # 3. Заменяем кавычки
        text = self.smart_quotes.sub('"', text)

        # 4. Заменяем ВСЕ виды дефисов и тире на обычный "-"
        text = self.special_hyphens.sub('-', text)

        # 5. Удаляем технический мусор (*, #, @, &)
        text = self.junk.sub(' ', text)

        # 6. Исправляем пробелы вокруг пунктуации (критично для сохранения фраз)
        text = self.space_before_punct.sub(r'\1', text)
        text = self.space_after_punct.sub(r'\1 ', text)

        # 7. Чистим множественные знаки препинания (!. -> !, !!! -> !)
        text = self.multi_punct.sub(r'\1', text)
        text = self.dup_punct.sub(r'\1', text)

        # 8. Финальная очистка от лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()

        return text

if __name__ == "__main__":
    normalizer = TextNormalizer()
    
    # Проверяем примеры 
    print(repr(normalizer.normalize("Спасибо , что дождались ответа.")))   # "Спасибо, что дождались ответа."
    print(repr(normalizer.normalize("Он приехал из Санкт-Петербурга вчера вечеро???. "))) # "Он приехал из Санкт-Петербурга вчера вечеро?. "
    print(normalizer.normalize("!.."))           # Ожидаем: "!"
    print(normalizer.normalize("«цитата»"))       # Ожидаем: '"цитата"'
    print(normalizer.normalize("текст * мусор"))  # Ожидаем: "текст мусор"
    print(normalizer.normalize("де‑факто"))      # Ожидаем: "де-факто" (с обычным дефисом)
    print(normalizer.normalize("Расстреливать надо таких писателей!.")) 