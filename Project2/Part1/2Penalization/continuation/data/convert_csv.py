"""
convert_weird_csv.py

Convertit un CSV par blocs :
    Nom
    indices (ignorés)
    valeurs

en CSV tabulaire propre :
    index;Nom1;Nom2;...

- Gère des longueurs différentes
- Remplit avec NaN
- Lisible directement par pandas
"""

import csv
from pathlib import Path
import math


INPUT_FILE = "continuation_raw.csv"        # <-- adapter
OUTPUT_FILE = "continuation_clean.csv"


def read_block_csv(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    data = {}
    max_len = 0

    i = 0
    while i < len(lines):
        name = lines[i]
        val_line = lines[i + 2]

        values = [
            float(v.replace(",", "."))
            for v in val_line.split(";")
            if v
        ]

        data[name] = values
        max_len = max(max_len, len(values))

        i += 3

    indices = list(range(1, max_len + 1))
    return indices, data, max_len


def write_clean_csv(path, indices, data, max_len):
    headers = ["index"] + list(data.keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(headers)

        for k in range(max_len):
            row = [indices[k]]
            for name in data:
                if k < len(data[name]):
                    row.append(data[name][k])
                else:
                    row.append(math.nan)
            writer.writerow(row)


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    indices, data, max_len = read_block_csv(input_path)
    write_clean_csv(output_path, indices, data, max_len)

    print(f"OK → {output_path.resolve()}")
    print(f"{max_len} lignes")
    for k, v in data.items():
        print(f"{k}: {len(v)} valeurs")


if __name__ == "__main__":
    main()
