import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------------
# 1. Wczytywanie i przygotowanie danych
# -------------------------------------------------------------
# Wczytanie pliku CSV ze średnikiem jako separatorem kolumn
df = pd.read_csv('data.csv', sep=';')

# Przekształcenie kolumny 'location' na zmienne binarne (one-hot encoding)
# Dzięki temu model może lepiej rozróżnić kategorie: center, suburbs, outskirts
# drop_first = True usuwa jedną kolumnę, aby uniknąć problemu kolineacji
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# Konwersja kolumny 'has_elevator' na typ całkowity (0 lub 1)
# Pozwala modelowi traktować tę cechę jako liczbową
df['has_elevator'] = df['has_elevator'].astype(int)

# -------------------------------------------------------------
# 2. Definiowanie cech (X) i zmiennej docelowej (y)
# -------------------------------------------------------------
# X zawiera wszystkie kolumny wejściowe poza docelową 'price_eur'
X = df.drop('price_eur', axis=1)
# y zawiera ceny mieszkań, które model będzie przewidywać
y = df['price_eur']

# -------------------------------------------------------------
# 3. Podział na zbiory treningowy i testowy
# -------------------------------------------------------------
# Używamy 80% danych do treningu i 20% do oceny modelu
# random_state = 42 gwarantuje powtarzalność podziału
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------
# 4. Inicjalizacja i trening modelu drzewa decyzyjnego
# -------------------------------------------------------------
# Tworzymy obiekt DecisionTreeRegressor z ustalonym ziarnem losowości
model = DecisionTreeRegressor(random_state=42, max_depth=5)
# Trenujemy model na zbiorze treningowym
model.fit(X_train, y_train)

# -------------------------------------------------------------
# 5. Predykcja i ocena na zbiorze testowym
# -------------------------------------------------------------
# Generujemy przewidywane ceny dla danych testowych
y_pred = model.predict(X_test)

# Obliczamy błąd średniokwadratowy (MSE) — im niższy, tym lepiej
mse = mean_squared_error(y_test, y_pred)
# Obliczamy współczynnik determinacji R² — im bliżej 1, tym lepsze dopasowanie
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------------------------------------------------
# 6. Analiza ważności cech
# -------------------------------------------------------------
# Pobieramy ważności poszczególnych zmiennych wejściowych
# Sortujemy malejąco, aby zobaczyć najbardziej istotne cechy
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# -------------------------------------------------------------
# 7. Przykładowe porównanie wartości rzeczywistych i przewidywanych
# -------------------------------------------------------------
# Tworzymy DataFrame z cenami rzeczywistymi i przewidywanymi
pred_df = pd.DataFrame({
    'actual_price': y_test,
    'predicted_price': np.round(y_pred, 2)
})
print("\nSample Predictions:")
print(pred_df.head())

# -------------------------------------------------------------
# 8. Generowanie krzywej nauki
# -------------------------------------------------------------
# learning_curve zwraca liczbę próbek i wyniki metryk dla różnych rozmiarów zbioru treningowego
train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    cv=5,   # liczba fałd walidacji krzyżowej || 5-krotna walidacja krzyżowa
    train_sizes=np.linspace(0.1, 1.0, 10), # 10 punktów od 10% do 100% danych
    scoring='r2',   # oceniamy za pomocą współczynnika R²
    n_jobs=-1   # użycie wszystkich dostępnych rdzeni procesora (-1 oznacza użycie wszystkich)
)

# Obliczanie średnich i odchyleń standardowych wyników
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# -------------------------------------------------------------
# 9. Rysowanie krzywej nauki przy pomocy matplotlib
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
# Wykres średniej R² dla zbioru treningowego
plt.plot(train_sizes, train_mean, 'o-', label='Training score')
# Wykres średniej R² dla walidacji krzyżowej
plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score')

# Ustawienia osi, tytułu i legendy
plt.title('Krzywa nauki — Decision Tree Regressor')
plt.xlabel('Liczba próbek treningowych')
plt.ylabel('R² Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
# Wyświetlamy gotowy wykres
plt.show()
