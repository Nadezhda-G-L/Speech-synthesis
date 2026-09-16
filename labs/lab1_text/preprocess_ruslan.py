"""Build the filtered RUSLAN metadata — entry point for lab 1.

Reads the corpus metadata (two columns), runs every utterance through the normalizer,
drops whatever the classifier rejects, and writes the result in LJSpeech format
(three columns)::

    000000_RUSLAN|С тревожным чувством берусь я за перо.|С тревожным чувством берусь я за перо.
                 ^ raw text                             ^ normalized text

The third column is your contribution: the original corpus does not have one. For most
rows it will equal the second, and that is expected — punctuation cleanup changes little.

Run from the lab directory::

    python preprocess_ruslan.py
"""

import csv

import pandas as pd

from text_filter import TextFilter
from text_normalizer import TextNormalizer

INPUT_PATH = "../../data/ruslan_dataset/metadata_RUSLAN_22200.csv"
OUTPUT_PATH = "../../data/metadata_RUSLAN_22200_normalized.csv"

# quoting=csv.QUOTE_NONE is required in both directions: the corpus text contains
# « » „ “ ” ' and pandas would otherwise read them as field delimiters.
CSV_KWARGS = {"sep": "|", "quoting": csv.QUOTE_NONE}


if __name__ == "__main__":
    text_filter = TextFilter()
    normalizer = TextNormalizer()

    raw = pd.read_csv(INPUT_PATH, names=["id", "raw"], **CSV_KWARGS)

    raw["nrm"] = raw["raw"].apply(normalizer.normalize)
    clean = raw[raw["nrm"].apply(text_filter.filter) == 1]

    print(f"Num rows before cleaning: {len(raw)}; after cleaning: {len(clean)}")

    clean[["id", "raw", "nrm"]].to_csv(
        OUTPUT_PATH, index=False, header=False, **CSV_KWARGS
    )
