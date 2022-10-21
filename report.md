z.1

| theta | avg epochs | x weight | y weight |
|-------|:----------:|----------|----------|
| 0.05  |     2      | 0.029    | 0.030    |
| 0.2   |     2      | 0.109    | 0.110    |
| 0.4   |     3      | 0.218    | 0.219    |
| 0.6   |     4      | 0.327    | 0.329    |
| 0.8   |     5      | 0.437    | 0.438    |
| 0.9   |     5      | 0.487    | 0.489    |
| 1.0   |     7      | 0.545    | 0.548    |
| 1.2   |     8      | 0.655    | 0.657    |

Uczenie jest tym szybsze im mniejsza jest wartość theta. Dla większych thet, model musi wypracować nadmiernie większe wagi, co powoduje wydłużenie uczenia.

z.2

| weight range | avg epochs | x weight | y weight | bias weight |
|--------------|:----------:|----------|----------|-------------|
| -1.0..1.0    |    11.0    | 0.260    | 0.260    | -0.423      |
| -0.8..0.8    |    11.4    | 0.139    | 0.191    | -0.289      |
| -0.5..0.5    |    7.7     | 0.072    | 0.102    | -0.132      |
| -0.2..0.2    |    4.9     | 0.044    | 0.053    | -0.080      |
| -0.1..0.1    |    3.6     | 0.027    | 0.025    | -0.044      |
| -0.05..0.05  |    2.8     | 0.015    | 0.016    | -0.024      |
| -0.01..0.01  |    3.0     | 0.013    | 0.019    | -0.025      |

Uczenie jest najszybsze dla wag w granicy -0.05..0.05. Ogólnie można powiedzieć, że im początkowe wagi są bliższe zeru, tym uczenie jest szybsze.

z.3

| learn factor | avg epochs | x weight | y weight | bias weight |
|--------------|:----------:|----------|----------|-------------|
| 0.001        |    14.9    | 0.023    | 0.014    | -0.034      |
| 0.01         |    3.4     | 0.022    | 0.037    | -0.049      |
| 0.1          |    2.5     | 0.123    | 0.154    | -0.213      |
| 0.2          |    3.1     | 0.268    | 0.396    | -0.516      |
| 0.5          |    3.1     | 0.708    | 0.994    | -1.260      |
| 0.8          |    3.3     | 1.247    | 1.692    | -2.261      |
| 1            |    3.6     | 1.805    | 2.334    | -3.216      |
| 2            |    4.8     | 4.198    | 5.379    | -7.586      |

Uczenie jest najszybsze przy alfie 0.1. Prawdopodobnie przy zbyt wysokiej alfie skoki są za duże i program nie może się wyuczyć, a przy zbyt małych alfach nauka jest zbyt wolna.

z.4

| type     | learn factor | avg epochs | x weight | y weight | bias weight |
|----------|--------------|:----------:|----------|----------|-------------|
| unipolar | 0.01         |    3.9     | 0.018    | 0.028    | -0.038      |
| bipolar  | 0.01         |    3.7     | 0.026    | 0.024    | -0.042      |

Nauka jest minimalnie szybsza dla funkcji aktywacji progowej bipolarnej. Prawdopodobnie ponieważ użycie bipolarnej funkcji progowej wywołuje większy kontrast w wynikach (i uczeniu) co przyspiesza ten proces.

