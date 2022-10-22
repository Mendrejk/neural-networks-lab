PERCEPTRON

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

ADALINE

z.1
weight range: -1.0..1.0, avg epochs: 7.4, reached error: 0.20106766, x weight: 0.72629744, y weight: 0.51416713, bias weight: -0.47946757
weight range: -0.8..0.8, avg epochs: 8.8, reached error: 0.20201369, x weight: 0.7125181, y weight: 0.48986974, bias weight: -0.46003342
weight range: -0.5..0.5, avg epochs: 8.7, reached error: 0.20354001, x weight: 0.72307736, y weight: 0.46323448, bias weight: -0.44109327
weight range: -0.2..0.2, avg epochs: 9, reached error: 0.20421803, x weight: 0.7250218, y weight: 0.45875636, bias weight: -0.43591675
weight range: -0.1..0.1, avg epochs: 8.8, reached error: 0.20411511, x weight: 0.72346365, y weight: 0.45929775, bias weight: -0.43623242
weight range: -0.05..0.05, avg epochs: 9, reached error: 0.20346801, x weight: 0.7255111, y weight: 0.46027002, bias weight: -0.43758535
weight range: -0.01..0.01, avg epochs: 9, reached error: 0.20274214, x weight: 0.7260271, y weight: 0.46221334, bias weight: -0.43973985

z.2
learn factor: 0.001, avg epochs: 70.6, reached error: 0.21903181, x weight: 0.6832651, y weight: 0.4077317, bias weight: -0.36545238
learn factor: 0.002, avg epochs: 36.1, reached error: 0.21850638, x weight: 0.6880817, y weight: 0.41016412, bias weight: -0.36995286
learn factor: 0.01, avg epochs: 8, reached error: 0.21374023, x weight: 0.71042204, y weight: 0.43703207, bias weight: -0.41179228
learn factor: 0.02, avg epochs: 4.7, reached error: 0.20559756, x weight: 0.73662454, y weight: 0.4793706, bias weight: -0.46894473
learn factor: 0.03, avg epochs: 3.4, reached error: 0.2083681, x weight: 0.7483674, y weight: 0.49738646, bias weight: -0.49769124
learn factor: 0.05, avg epochs: 2.9, reached error: 0.19779778, x weight: 0.7886459, y weight: 0.5772709, bias weight: -0.5934345
learn factor: 0.1, avg epochs: 2, reached error: 0.21271567, x weight: 0.8285847, y weight: 0.6725858, bias weight: -0.6847521

z.3
error boundary: 0.3, avg epochs: 3, reached error: 0.27687103, x weight: 0.6617611, y weight: 0.3709082, bias weight: -0.34708056
error boundary: 0.25, avg epochs: 4, reached error: 0.22530563, x weight: 0.7128776, y weight: 0.43886465, bias weight: -0.42337242
error boundary: 0.22, avg epochs: 4.8, reached error: 0.20540829, x weight: 0.7392472, y weight: 0.47851235, bias weight: -0.46889344
error boundary: 0.21, avg epochs: 5, reached error: 0.20182486, x weight: 0.7438632, y weight: 0.48682866, bias weight: -0.47803402
error boundary: 0.2, avg epochs: 5.8, reached error: 0.19220673, x weight: 0.75819576, y weight: 0.5169569, bias weight: -0.5109798
error boundary: 0.19, avg epochs: 6.6, reached error: 0.18683045, x weight: 0.768134, y weight: 0.53948736, bias weight: -0.5354692
error boundary: 0.18, avg epochs: 11, reached error: 0.1799027, x weight: 0.78959125, y weight: 0.60300183, bias weight: -0.6025791
