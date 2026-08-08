# Results

Backend: `lightgbm`. Seasons: 2021, 2022, 2023.

Panel: 7,289 games, 3 seasons, home win rate 0.5311.

## Shrinkage

Median fitted concentration in August and later: kappa = 43.6, implying a true-talent standard deviation of 0.0749. Real MLB team talent spreads near .045 to .050, so a fitted kappa around 100 is the expected neighbourhood.

| | raw win pct | shrunk talent |
|---|---|---|
| April spread (sd) | 0.1904 | 0.0324 |
| April range | 0.000 to 1.000 | 0.312 to 0.691 |

PyMC not installed, so the MCMC cross-check was skipped. Install pymc and rerun to populate this line.

## Leakage check

With the target shuffled, out-of-sample Brier is 0.2509 against an expected 0.2491. **PASS.** A score meaningfully below the expectation would mean a feature carries information about its own row.

## Feature ladder

### gbm

| model | features | Brier | vs base | t | vs previous |
|---|---:|---:|---:|---:|---:|
| constant base rate (0.5311) | 0 | 0.2490 | +0.00000 |  |  |
| raw team win pct | 2 | 0.2450 | +0.00401 | 5.01 | +0.00401 |
| + EB shrunk talent | 2 | 0.2445 | +0.00449 | 4.59 | +0.00048 |
| + form and rest | 6 | 0.2449 | +0.00406 | 4.17 | -0.00043 |
| + starting pitcher | 10 | 0.2445 | +0.00449 | 4.63 | +0.00043 |
| + park and prior season | 13 | 0.2443 | +0.00466 | 4.48 | +0.00017 |

### logistic

| model | features | Brier | vs base | t | vs previous |
|---|---:|---:|---:|---:|---:|
| constant base rate (0.5311) | 0 | 0.2490 | +0.00000 |  |  |
| raw team win pct | 2 | 0.2462 | +0.00276 | 5.09 | +0.00276 |
| + EB shrunk talent | 2 | 0.2434 | +0.00559 | 5.52 | +0.00282 |
| + form and rest | 6 | 0.2438 | +0.00519 | 5.14 | -0.00040 |
| + starting pitcher | 10 | 0.2443 | +0.00470 | 4.06 | -0.00049 |
| + park and prior season | 13 | 0.2444 | +0.00456 | 3.44 | -0.00014 |

Best rung: + EB shrunk talent (2 features).

## Headline

Best family: **logistic**, scored on 6,783 out-of-sample games.

| quantity | value |
|---|---:|
| base rate Brier | 0.2490 |
| model Brier | 0.2434 |
| improvement | 0.00559 |
| standard error | 0.00101 |
| t statistic | 5.52 |
| 95% CI | (0.00360, 0.00757) |
| bootstrap 95% CI | (0.00361, 0.00760) |
| log loss | 0.6798 |
| reliability (lower better) | 0.00030 |
| resolution (higher better) | 0.00551 |

## Calibration

Isotonic fit on the first 70% of out-of-sample rows and applied to the remaining 30%: Brier 0.2451 before, 0.2461 after.

|   bin |   n |   mean_forecast |   observed |         gap |
|------:|----:|----------------:|-----------:|------------:|
|     0 |   4 |        0.242773 |   0.75     | -0.507227   |
|     1 |  21 |        0.315489 |   0.333333 | -0.0178442  |
|     2 |  74 |        0.367321 |   0.418919 | -0.0515976  |
|     3 | 208 |        0.427881 |   0.447115 | -0.0192339  |
|     4 | 451 |        0.483094 |   0.476718 |  0.00637512 |
|     5 | 567 |        0.535406 |   0.500882 |  0.0345238  |
|     6 | 465 |        0.588115 |   0.572043 |  0.0160716  |
|     7 | 166 |        0.643828 |   0.698795 | -0.0549669  |
|     8 |  73 |        0.697086 |   0.671233 |  0.0258536  |
|     9 |   6 |        0.755533 |   0.5      |  0.255533   |

## Market benchmark

Matched a price for 6,919 of 7,289 games (94.9%). Mean closing margin 0.0415.

| forecaster | Brier | vs base | t |
|---|---:|---:|---:|
| this model | 0.2438 | +0.00508 | 4.88 |
| market opening line | 0.2388 | +0.01008 | 8.62 |
| market closing line | 0.2341 | +0.01478 | 11.88 |

The closing line beats a constant by 0.01478. This model captures 34% of that.

### Does the model know anything the market does not?

A worse Brier score is not the same as no information. Two forecasts can each be individually inferior and still combine into something better, if each sees what the other misses. Both go on the log-odds scale and the outcome is regressed on the pair; if the model adds nothing its coefficient is zero.

| benchmark | market coef | model coef | t on model |
|---|---:|---:|---:|
| opening line | +1.063 | -0.051 | -0.48 |
| closing line | +1.313 | -0.282 | -2.92 |

Standard errors are bootstrapped, because the two regressors are collinear by construction and asymptotic errors are optimistic there.

## Decision layer

Filled at the best opening price across books, marked against the consensus close. 4,920 bets on 76% of games, hit rate 0.4439.

| quantity | value |
|---|---:|
| ROI on bankroll | -89.53% |
| ROI per unit staked | -0.0105 |
| max drawdown | -95.46% |
| mean CLV | -0.00020 |
| CLV t statistic | -0.30 |
| share of bets with positive CLV | 45.9% |

Threshold sweep, so the result can be read against the choice rather than at one point:

|   threshold |   n_bets |   hit_rate |     roi |   roi_per_unit |   mean_clv |   clv_t |   max_drawdown |
|------------:|---------:|-----------:|--------:|---------------:|-----------:|--------:|---------------:|
|        0    |     5919 |     0.4486 | -0.9048 |        -0.0099 |    -0      | -0.0112 |        -0.9644 |
|        0.01 |     5554 |     0.4476 | -0.9034 |        -0.0101 |    -0.0006 | -1.012  |        -0.964  |
|        0.02 |     4920 |     0.4439 | -0.8953 |        -0.0105 |    -0.0002 | -0.2965 |        -0.9546 |
|        0.03 |     4143 |     0.4395 | -0.8832 |        -0.0114 |     0.0002 |  0.2581 |        -0.9501 |
|        0.04 |     3462 |     0.4359 | -0.8059 |        -0.0119 |     0.001  |  1.2924 |        -0.8937 |
|        0.05 |     2895 |     0.4342 | -0.6778 |        -0.0117 |     0.0013 |  1.5866 |        -0.8502 |
|        0.06 |     2308 |     0.4328 | -0.3972 |        -0.0086 |     0.0015 |  1.6148 |        -0.7662 |
|        0.07 |     1856 |     0.4213 | -0.6186 |        -0.0167 |     0.0019 |  1.7831 |        -0.8444 |
|        0.08 |     1473 |     0.4216 | -0.5048 |        -0.0171 |     0.0019 |  1.6152 |        -0.7817 |
|        0.09 |     1157 |     0.4114 | -0.6099 |        -0.0264 |     0.0025 |  1.8242 |        -0.7662 |
|        0.1  |      896 |     0.3973 | -0.7304 |        -0.0408 |     0.0023 |  1.6336 |        -0.7945 |

## Search budget

4 hyperparameter configurations across 6 feature sets (24 evaluations) were scored on the reported walk-forward split. The headline Brier is therefore mildly optimistic; a fully clean estimate would require a season held out and touched once.

