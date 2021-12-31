from typing import Callable

import pandas as pd
import plotly.express as px

Transform = Callable[[pd.DataFrame], pd.DataFrame]


def execute_pipeline(*transforms: Transform):
    def _pipeline(initial: pd.DataFrame):
        df = initial
        for transform in transforms:
            df = transform(df)
        return df

    return _pipeline


def categorize(series: pd.Series, age_min: int, age_max: int, step = 5):
    bins = range(age_min - 1, age_max + 1, step)
    return pd.cut(series, bins)


def encode(col: str):
    def _encode(df: pd.DataFrame):
        df[col] = encode_categorical(df[col])
        return df

    return _encode


def encode_categorical(series: pd.Series):
    encoded, _ = pd.factorize(series)
    return encoded


def find_na(df: pd.DataFrame):
    return df.isnull().any()


def min_max(series: pd.Series):
    return series.min(), series.max()


def normalize(col: str):
    def _normalize(df: pd.DataFrame):
        df[col] = normalize_min_max(df[col])
        return df

    return _normalize


def normalize_min_max(series: pd.Series):
    return (series - series.min()) / (series.max() - series.min())


def pie_plot(col: str, title: str = None):
    def _plot(df: pd.DataFrame):
        px.pie(df, col,
               title = title).show()
        return df

    return _plot


def histogram(col: str, title: str = None):
    def _plot(df: pd.DataFrame):
        px.histogram(df, x = col,
                     title = title).show()
        return df

    return _plot
