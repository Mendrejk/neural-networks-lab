# Sprawozdanie – sieć wielowarstwowa uczona metodą propagacji wstecznej

## Sebastian Łakomy

#### Parametry początkowe:

- warstwy ukryte: 40, 60, 80 neuronów
- wagi początkowe: średnia 1, odchylenie 0.01
- funkcja aktywacji: Tangens Hiperboliczny
- współczynnik uczenia: 0.005
- wielkość paczki: 50
- nauka - 30 epok

1. Liczba neuronów w warstwach ukrytych
    - warstwa pierwsza (x, 60, 80)

| liczba neuronów | najlepszy wynik | numer epoki |
|:---------------:|:----------------|:-----------:|
|       20        | 0.8001          |     29      |
|       40        | 0.909           |     30      |
|       60        | 0.9138          |     30      |
|       80        | 0.9219          |     30      |
|       100       | 0.9218          |     29      |

-
    - warstwa druga (80, x, 80)

| liczba neuronów | najlepszy wynik | numer epoki |
|:---------------:|:----------------|:-----------:|
|       20        | 0.8264          |     30      |
|       40        | 0.9235          |     30      |
|       60        | 0.9228          |     30      |
|       80        | 0.9227          |     29      |
|       100       | 0.9143          |     29      |

-
    - warstwa trzecia (80, 40, x)

| liczba neuronów | najlepszy wynik | numer epoki |
|:---------------:|:----------------|:-----------:|
|       20        | 0.6154          |     30      |
|       40        | 0.743           |     30      |
|       60        | 0.8958          |     30      |
|       80        | 0.9254          |     30      |
|       100       | 0.9265          |     30      |

Wynik: 80, 40, 80
Co ciekawe, zwiększanie ilości neuronów w środkowej warstwie powoduje spadek wydajności uczenia. Największe zmiany
powoduje trzecia, ostatnia ukryta warstwa.

2. Wielkość paczki

| wielkość paczki | najlepszy wynik | numer epoki |
|:---------------:|:----------------|:-----------:|
|       10        | 0.9091          |      8      |
|       20        | 0.9392          |     29      |
|       50        | 0.8927          |     30      |
|       75        | 0.8628          |     30      |
|       100       | 0.7496          |     30      |
|       200       | 0.5082          |     30      |

Wynik: 20
O dziwo (poza paczką 10) im większa paczka, tym gorszy wynik. Prawdopodobnie wynika to z tego, że przy większych
paczkach poprawa wag jest pewniejsza ale mniej gwałtowna - uczenie jest wolniejsze i przy 30 epokach uczenie z dużymi
paczkami nie jest w stanie osiągnąć optimum.

3. Współczynnik uczenia

| współczynnik uczenia | najlepszy wynik | numer epoki |
|:--------------------:|:----------------|:-----------:|
|        0.0005        | 0.4772          |     10      |
|        0.001         | 0.8946          |     30      |
|        0.002         | 0.827           |     30      |
|        0.005         | 0.9391          |     30      |
|         0.01         | 0.9062          |      9      |
|         0.05         | 0.6068          |     10      |

Wynik: 0.005
Za małe współczynniki uczenia powodują wolną naukę i/lub osiągnięcie minimum lokalnego (dla 0.0005 stanęło przy +- 0.47). Za duże (0.05) - zbyt gwałtowne zmiany.

4. Inicjalizacja wag początkowych

| odchylenie standardowe | najlepszy wynik | numer epoki |
|:----------------------:|:----------------|:-----------:|
|         0.0001         | 0.098           |      1      |
|         0.001          | 0.1009          |      8      |
|          0.01          | 0.9354          |     30      |
|         0.015          | 0.9159          |     27      |
|          0.02          | 0.9891          |      8      |
|          0.05          | 0.7936          |      1      |

Wynik: 0.02

Za małe odchylenia powodują, że wagi startowe są zbyt bliskie 1 aby mogły się znacząco zmienić. Za duże odchylenia powodują zbyt wysoką losowość uczenia.
