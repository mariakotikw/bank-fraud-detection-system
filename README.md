# Bank Fraud Detection System

ML-проект для выявления мошеннических банковских транзакций на сильно несбалансированных данных.

## Описание задачи

Цель проекта — построить модель, которая по параметрам транзакции оценивает вероятность мошенничества и помогает принять решение:

- `approve` — транзакция выглядит безопасной;
- `review` — транзакцию стоит отправить на ручную проверку;
- `block` — транзакция выглядит подозрительной.

Особенность задачи — сильный дисбаланс классов: fraud-транзакции составляют около **0.17%** от всех наблюдений. Поэтому accuracy не используется как основная метрика: модель может почти всегда предсказывать `normal` и всё равно получать высокую accuracy.

## Dataset

Использован датасет **Credit Card Fraud Detection**.

Целевая переменная:

```text
Class = 0 — normal transaction
Class = 1 — fraud transaction
```

Основные признаки:

```text
Time, Amount, V1–V28
```

Признаки `V1–V28` анонимизированы и получены после PCA-преобразования.

## Project pipeline

1. Exploratory Data Analysis
2. Train/test split with stratification
3. Feature preprocessing
4. Baseline model: Logistic Regression
5. Model comparison: Logistic Regression, Random Forest, HistGradientBoosting
6. Threshold selection
7. Business cost analysis
8. Final decision logic: approve / review / block

## Class imbalance

Распределение классов:

```text
Normal transactions: 99.8273%
Fraud transactions:   0.1727%
```

Из-за такого дисбаланса основной фокус сделан на метриках:

- precision;
- recall;
- F1-score;
- ROC-AUC;
- PR-AUC;
- confusion matrix.

## Model comparison

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.50 | 0.972 | 0.716 | 0.061 | 0.918 | 0.114 |
| Random Forest | 0.50 | 0.982 | 0.800 | 0.648 | 0.847 | 0.735 |
| HistGradientBoosting | 0.50 | 0.961 | 0.716 | 0.290 | 0.878 | 0.435 |

Logistic Regression показала высокий recall, но слишком низкий precision: модель находила большинство fraud-транзакций, но создавала много ложных срабатываний.

HistGradientBoosting также показал высокий recall, но уступил Random Forest по precision, PR-AUC и F1-score.

Random Forest показал лучший баланс между precision и recall, а также лучшие значения PR-AUC и F1-score среди протестированных моделей, поэтому был выбран как финальная модель.

## Threshold selection

Стандартный threshold `0.5` не всегда оптимален для fraud detection.  
Были протестированы разные пороги принятия решения.

Лучший threshold по F1-score:

```text
threshold = 0.64
precision = 0.830
recall = 0.796
F1 = 0.813
false positives = 16
false negatives = 20
```

Также была добавлена простая бизнес-функция стоимости ошибок:

```text
business_cost = 5 * FP + 100 * FN
```

Где:

- `FP` — обычная транзакция ошибочно отправлена на проверку;
- `FN` — мошенническая транзакция пропущена.

Лучший threshold по бизнес-стоимости:

```text
threshold = 0.47
precision = 0.620
recall = 0.867
F1 = 0.723
false positives = 52
false negatives = 13
business cost = 1560
```

Для финальной модели выбран threshold `0.47`, так как в fraud detection пропустить мошенническую транзакцию обычно дороже, чем отправить дополнительную честную транзакцию на проверку.

## Final decision logic

```text
fraud_probability < 0.20         → approve
0.20 <= fraud_probability < 0.47 → review
fraud_probability >= 0.47        → block
```

## Results

Финальная модель — **Random Forest with selected threshold**

Модель позволяет:

- находить fraud-транзакции на сильно несбалансированных данных;
- управлять балансом между precision и recall;
- учитывать бизнес-стоимость ошибок;
- переводить вероятность fraud в понятное решение: approve / review / block.

## Project structure

```text
bank-fraud-detection-system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── models/
│
├── reports/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── threshold_metrics.png
│   │   ├── business_cost_by_threshold.png
│   │   └── confusion_matrix_rf.png
│   ├── model_comparison.csv
│   └── threshold_analysis.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Tech stack

```text
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Jupyter Notebook
```

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Put dataset into:

```text
data/raw/creditcard.csv
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/01_eda.ipynb
```

## Future improvements

- Add CatBoost / LightGBM model comparison;
- Add SHAP-based model interpretation;
- Add Streamlit demo for fraud probability prediction;
- Move training and inference logic from notebook to reusable Python scripts.