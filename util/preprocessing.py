import pandas as pd

from util.tools import execute_pipeline, encode, pie_plot, find_na, min_max, categorize, normalize, histogram


def preprocess(df: pd.DataFrame):
    return execute_pipeline(
        inspect_data_size,
        data_availbaility_check,
        histogram('age'),
        categorize_age,
        encode('age'),
        normalize('age'),
        inspect_gender_values,
        pie_plot('gender', title = 'Gender distribution'),
        encode('gender'),
        correlation_inspection,
    )(initial = df)


def inspect_data_size(df: pd.DataFrame):
    n_samples, n_features = df.shape
    print(f'Number of samples: {n_samples}')
    print(f'Number of features: {n_features}\n')
    return df


def data_availbaility_check(df: pd.DataFrame):
    has_na = find_na(df)
    print(f'Data availability check: \n{has_na}\n')
    return df


def categorize_age(df: pd.DataFrame):
    age_min, age_max = min_max(df['age'])
    print(f'Age range: {age_min}..{age_max}\n')

    df['age'] = categorize(df['age'], age_min, age_max)
    bin_count = df['age'].unique().size
    print(f'Number of age bins: {bin_count}\n')
    return df


def inspect_gender_values(df: pd.DataFrame):
    gender_values = df['gender'].unique()
    print(f'Unique genders: {gender_values}\n')
    return df


def correlation_inspection(df: pd.DataFrame):
    corr = df.corr().iloc[-1]
    print(f'Diabetes correlation:\n{corr}\n')
    return df
