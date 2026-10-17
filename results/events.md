Events: 226 statements, 2000-02-02 to 2026-09-16.

### 2-year Treasury yield on statement day

| Tone surprise measure | Effect per 1 s.d. (bp) | t-stat | R² | R² decision only | OOS R² gain | OOS hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM document stance | 1.17 | 1.76 | 0.043 | 0.021 | 0.008 | 0.61 |
| LLM sentence net tone | 0.42 | 0.82 | 0.024 | 0.021 | -0.005 | 0.54 |
| TF-IDF net tone | 0.99 | 1.65 | 0.037 | 0.021 | 0.005 | 0.56 |
| Dictionary net tone | 0.84 | 1.22 | 0.032 | 0.021 | 0.001 | 0.50 |

### Trade-weighted dollar on statement day

| Tone surprise measure | Effect per 1 s.d. (%) | t-stat | R² | R² decision only | OOS R² gain | OOS hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM document stance | -0.01 | -0.74 | 0.021 | 0.019 | -0.007 | 0.59 |
| LLM sentence net tone | -0.02 | -1.07 | 0.022 | 0.019 | -0.002 | 0.58 |
| TF-IDF net tone | -0.01 | -0.41 | 0.020 | 0.019 | -0.007 | 0.57 |
| Dictionary net tone | 0.01 | 0.21 | 0.020 | 0.019 | -0.014 | 0.57 |

### 10-year Treasury yield on statement day

| Tone surprise measure | Effect per 1 s.d. (bp) | t-stat | R² | R² decision only | OOS R² gain | OOS hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM document stance | 1.10 | 2.35 | 0.021 | 0.001 | 0.017 | 0.51 |
| LLM sentence net tone | 0.30 | 0.64 | 0.002 | 0.001 | -0.004 | 0.51 |
| TF-IDF net tone | 0.64 | 1.50 | 0.008 | 0.001 | 0.003 | 0.50 |
| Dictionary net tone | 1.00 | 2.12 | 0.018 | 0.001 | 0.006 | 0.56 |

### Post-statement drift trade (2y, next 5 trading days)

| Measure | Events | Mean P&L (bp) | t-stat | Hit rate |
| --- | ---: | ---: | ---: | ---: |
| LLM document stance | 165 | -0.08 | -0.08 | 0.45 |
| LLM sentence net tone | 207 | 0.18 | 0.23 | 0.44 |
| TF-IDF net tone | 204 | -0.68 | -0.87 | 0.45 |
| Dictionary net tone | 191 | 0.42 | 0.50 | 0.44 |

### Sub-samples (2y yield)

| Measure | Period | Effect (bp) | t-stat | R² |
| --- | --- | ---: | ---: | ---: |
| LLM document stance | 2000-2012 | 0.73 | 0.67 | 0.028 |
| LLM sentence net tone | 2000-2012 | 0.06 | 0.10 | 0.020 |
| TF-IDF net tone | 2000-2012 | 0.54 | 0.82 | 0.027 |
| Dictionary net tone | 2000-2012 | 0.62 | 0.72 | 0.028 |
| LLM document stance | 2013-2026 | 1.69 | 2.52 | 0.093 |
| LLM sentence net tone | 2013-2026 | 1.38 | 1.65 | 0.062 |
| TF-IDF net tone | 2013-2026 | 2.57 | 2.84 | 0.099 |
| Dictionary net tone | 2013-2026 | 1.56 | 1.60 | 0.065 |

LLM consistency: average std of the stance score across repeated runs 0.061; supporting quotes verified verbatim in 100% of statements.
