#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apartment_price_tree.py

Skrypt do trenowania drzewa decyzyjnego na danych o wynajmie mieszkań.
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message="The least populated class in y has only 1 members"
)
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples"
)

import pandas as pd
import numpy as np
import sys

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

DATA_PATH = "data.csv"
DATA_SEP = ";"
DATA_ENCODING = "latin1"


def main():
    try:
        df = pd.read_csv(DATA_PATH, sep=DATA_SEP, encoding=DATA_ENCODING)
    except Exception as e:
        print(f"[ERROR] Nie udało się wczytać {DATA_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Wczytano dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
    print(df.head())

    # Usuwamy kolumny, których nie będziemy używać, a które będą tylko przeszkadzać
    cols_to_drop = ["id", "title", "body", "amenities", "source", "time"]
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # pokoje "Studio" rzutujemy jako 0 pokoi konwetując na float
    df_clean['bedrooms'] = (
        df_clean['bedrooms'].replace({'studio': 0, 'Studio': 0}).astype(float)
    )

    # Liczymy "udogodnienia" aka ile elementów na liście
    df_clean['amenity_count'] = (
        df['amenities'].fillna('').str.split(',').str.len()
    )

    # Konwersja kolumn numerycznych na liczby
    for col in ['bathrooms', 'square_feet', 'latitude', 'longitude', 'price']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Usuwamy wiersze z brakującymi kluczowymi wartościami
    df_clean = df_clean.dropna(subset=['bathrooms', 'bedrooms', 'square_feet', 'price'])

    print(f"[INFO] Po czyszczeniu: {df_clean.shape[0]} wierszy, {df_clean.shape[1]} kolumn")
    print(df_clean.head())

    # wybór kolumn numerycznych, które zostawimy "as is"
    numeric_feats = [
        'bathrooms',
        'bedrooms',
        'square_feet',
        'latitude',
        'longitude',
        'amenity_count'
    ]

    # wybór kolumn kategorycznych, które potem trzeba będzie zakodować
    categorical_feats = [
        'currency',
        'fee',
        'has_photo',
        'pets_allowed',
        'category',
        'state'
    ]

    # Tworzymy macierz cech X
    X = df_clean[numeric_feats + categorical_feats]

    # Tworzymy wektor odpowiedzi y
    y = df_clean['price'].astype(float)

    print(f"[INFO] Przygotowano X o kształcie {X.shape} oraz y o długości {len(y)}")

    # dzielimy dane na trening(80%) i test(20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[INFO] Zbiór treningowy: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[INFO] Zbiór testowy    : X_test ={X_test.shape}, y_test ={y_test.shape}")

    # Pipeline, czyli kodowanie kategorii i Decision Tree
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feats)
        ]
    )

    tree_model = DecisionTreeClassifier(random_state=42)

    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ('model', tree_model)
    ])

    # trenowanie
    param_grid = {
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_leaf': [1, 5, 20],
        'model__ccp_alpha': [0.0, 0.0005, 0.001]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    print("[INFO] Rozpoczynam strojenie hiperparametrów...")
    gs.fit(X_train, y_train)
    print(f"[RESULT] Najlepsze parametry: {gs.best_params_}")

    # podstawiamy zestrojony model jako najlepszy
    best_model = gs.best_estimator_

    # testujemy model na zbiorze test
    print("[INFO] Rozpoczynam ocenę modelu na zbiorze testowym...")
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"[RESULT] Test RMSE: {rmse:.2f}")
    print(f"[RESULT] Test R²  : {r2:.3f}")


if __name__ == "__main__":
    main()
