"""Загрузка Amazon Reviews 2018 и простая подготовка данных."""

import gzip
import html
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


SOURCE = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html"
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
FILES = {
    "Musical_Instruments_5.json.gz": f"{BASE_URL}/categoryFilesSmall/Musical_Instruments_5.json.gz",
    "meta_Musical_Instruments.json.gz": f"{BASE_URL}/metaFiles2/meta_Musical_Instruments.json.gz",
}


def download_data(directory="data"):
    """Скачать два архива. Уже скачанные файлы пропускаются."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = directory / name
        if path.exists():
            continue
        print(f"Скачивание {name}", flush=True)
        temporary = path.with_suffix(".part")
        with urlopen(url, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.rename(path)


def read_reviews(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            rows.append({
                "user_id": row.get("reviewerID"),
                "asin": row.get("asin"),
                "rating": row.get("overall"),
                "timestamp": row.get("unixReviewTime"),
            })
    return pd.DataFrame(rows)


def clean_reviews(raw):
    data = raw.copy()
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], unit="s", utc=True, errors="coerce"
    )
    data = data.dropna(subset=["user_id", "asin", "rating", "timestamp"])
    data = data[data.rating.between(1, 5)]
    # Первое взаимодействие сохраняет хронологию и исключает повторы в holdout.
    data = data.sort_values("timestamp", kind="stable")
    return data.drop_duplicates(["user_id", "asin"], keep="first").reset_index(drop=True)


def read_metadata(path, item_ids):
    """Читать только нужные товары, не загружая все поля метаданных в память."""
    wanted = set(item_ids)
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row.get("asin") not in wanted:
                continue
            categories = row.get("category", row.get("categories", []))
            if categories and isinstance(categories[0], list):
                categories = categories[0]
            rows.append({
                "asin": row["asin"],
                "title": html.unescape(row.get("title") or "").strip(),
                "brand": html.unescape(row.get("brand") or "").strip(),
                "category": html.unescape(categories[-1]).strip() if categories else "",
            })
    metadata = pd.DataFrame(rows, columns=["asin", "title", "brand", "category"])
    metadata = metadata.drop_duplicates("asin", keep="first")
    # Порядок строк совпадает с внутренними item ID, включая товары без metadata.
    items = pd.DataFrame({"asin": item_ids})
    return items.merge(metadata, on="asin", how="left").fillna("")


def encode_ids(data):
    """Числовые ID нужны для индексации матриц; сами по себе они не признаки."""
    data = data.copy()
    data["user"], users = pd.factorize(data.user_id, sort=True)
    data["item"], items = pd.factorize(data.asin, sort=True)
    return data, users, items


def temporal_split(data, train_fraction=0.8, valid_fraction=0.1):
    train_end, valid_end = data.timestamp.quantile(
        [train_fraction, train_fraction + valid_fraction]
    )
    train = data[data.timestamp <= train_end].copy()
    valid = data[(data.timestamp > train_end) & (data.timestamp <= valid_end)].copy()
    test = data[data.timestamp > valid_end].copy()
    return train, valid, test


def make_item_features(items, train, n_users):
    """ID товара, бренд и конечная категория как отдельные поля FM.

    Словари метаданных строятся только по товарам из обучающей истории.
    Пространства user, item, brand и category не пересекаются.
    """
    columns = [np.arange(len(items), dtype=np.int64) + n_users]
    offset = n_users + len(items)
    train_items = np.sort(train.item.unique())
    for field in ["brand", "category"]:
        values = sorted(set(items.iloc[train_items][field]) - {""})
        vocabulary = {value: index + 1 for index, value in enumerate(values)}
        codes = items[field].map(vocabulary).fillna(0).to_numpy(dtype=np.int64)
        columns.append(codes + offset)
        offset += len(vocabulary) + 1  # Код 0 означает неизвестное значение.
    return np.column_stack(columns)
