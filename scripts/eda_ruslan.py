# scripts/eda_ruslan.py
import pandas as pd
import os
import pathlib

def analyze_ruslan():
    # Автоматически определяем корень проекта (scripts/../)
    script_dir = pathlib.Path(__file__).parent.resolve()
    root_dir = script_dir.parent  # Корень проекта

    data_path = root_dir / "data" / "ruslan_dataset" / "metadata_RUSLAN_22200.csv"
    if not data_path.exists():
        print(f"Файл не найден: {data_path}")
        print("Убедитесь, что запустили скрипт из корня проекта или скачали датасет в папку data/")
        return

    df = pd.read_csv(data_path, sep="|", names=["id", "raw_text"])
    print(f"Всего фраз: {len(df)}")
    print(df.head())
    
    patterns = {
        "Цифры": r"\d+",
        "Некириллица": r"[^\w\s\u0400-\u04FF]",
        "Технические символы": r"[\*\@\+\-\/\<\>\(\)\[\]\{\}\&\#\%\$\^\~\`\|\:\\\"]",
        "Обозначения": r"[°%$€£¥₽]",
        "Сокращения": r"\b(?:г\.?|ул\.?|д\.?|м\.?|ит\.д\.?|и\.т\.п\.?)\b",
        "Аббревиатуры": r"\b[A-ZА-ЯЁ]{2,}(?!\w)",
        "Междометия/Эмодзи": r"(?:угу|хаха|ммм|ыых|😀|😂|👍)",
        "Длинные фразы (>1000 символов)": r".{1000,}"
    }
    
    stats = {}
    for name, pat in patterns.items():
        matches = df["raw_text"].str.contains(pat, regex=True, na=False).sum()
        stats[name] = matches
        
    # Сохраняем в папку data/scripts (автоматически создастся)
    output_dir = root_dir / "data" / "scripts"
    output_path = output_dir / "eda_stats.csv"
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(list(stats.items()), columns=["Тип", "Количество"]).to_csv(output_path, index=False)
    print(f"Статистика сохранена в: {output_path}")
    return df

if __name__ == "__main__":
    analyze_ruslan()