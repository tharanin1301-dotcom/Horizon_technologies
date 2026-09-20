# Unemployment Analysis Using Python

> **Horizon TechX Internship – Task 2**

---

## Overview

This project delivers a complete, evidence-based **Exploratory Data Analysis (EDA)** of unemployment trends across Indian states and union territories. Using the publicly available CMIE (Centre for Monitoring Indian Economy) Unemployment in India Dataset, we analyze unemployment patterns, regional disparities, labour participation trends, and evaluate measurable changes during the COVID-19 national lockdown period (March–June 2020).

All findings are derived from **actual calculated values**. No data was fabricated.

---

## Internship

| Field | Detail |
|:---|:---|
| Company | Horizon TechX |
| Task | Task 2 – Unemployment Analysis Using Python |
| Project Type | Data Analysis / Exploratory Data Analysis |

---

## Student

| Field | Detail |
|:---|:---|
| Name | Tharani Natarajan |
| College | IFET College of Engineering |
| Department | Artificial Intelligence and Data Science |

---

## Objective

1. **Unemployment Trend Analysis** – Understand how the national unemployment rate evolved month-by-month from May 2019 to June 2020.
2. **Regional Comparison** – Identify and rank 28 Indian states and union territories by their average unemployment rate.
3. **COVID-19 Period Analysis** – Quantify the measurable change in unemployment between the Pre-COVID baseline and the COVID-19 lockdown period.
4. **Employment & Labour Participation Analysis** – Track trends in estimated employed population and labour force participation rates across Rural and Urban areas.
5. **Visualization** – Create 8 publication-quality charts using Matplotlib and Seaborn.
6. **Evidence-Based Insights** – Generate findings grounded exclusively in actual dataset results.

---

## Dataset

| Attribute | Detail |
|:---|:---|
| **Dataset Name** | Unemployment in India Dataset |
| **Primary Source** | Centre for Monitoring Indian Economy (CMIE) |
| **Public Archive** | [GitHub – adduadnanali/Unemployment-Analysis-in-India](https://raw.githubusercontent.com/adduadnanali/Unemployment-Analysis-in-India/main/Unemployment%20in%20India.csv) |
| **File Name** | `data/Unemployment in India.csv` |
| **Time Period** | May 31, 2019 – June 30, 2020 (14 consecutive months) |
| **Geographic Coverage** | 28 Indian States & Union Territories |
| **Area Coverage** | Rural and Urban |
| **Raw Dimensions** | 768 rows × 7 columns |
| **Valid Observations** | 740 rows (after removing 28 trailing blank rows) |
| **Variables** | Region, Date, Frequency, Estimated Unemployment Rate (%), Estimated Employed, Estimated Labour Participation Rate (%), Area |

> **Disclaimer:** This dataset was sourced from a reputable public repository for academic and internship purposes. It has **not** been supplied by Horizon TechX.

---

## Technologies

| Library | Purpose |
|:---|:---|
| Python 3.14 | Core programming language |
| Pandas | Data loading, cleaning, manipulation, and grouping |
| NumPy | Numerical operations and COVID period classification |
| Matplotlib | Chart rendering, annotations, and 300 DPI export |
| Seaborn | Statistical visualizations (boxplot, heatmap, histplot) |
| Jupyter Notebook | Interactive analysis and narrative documentation |
| pathlib | OS-independent project path management |

---

## Project Structure

```
Task-2-Unemployment-Analysis/
│
├── README.md                           ← This file
├── requirements.txt                    ← Python dependencies
├── .gitignore                          ← Git ignore rules
│
├── data/
│   ├── README.md                       ← Dataset documentation
│   └── Unemployment in India.csv       ← CMIE source dataset (740 valid rows)
│
├── notebooks/
│   └── unemployment_analysis.ipynb     ← 19-section Jupyter Notebook
│
├── src/
│   └── unemployment_analysis.py        ← Production Python script (modular)
│
└── outputs/
    ├── figures/                        ← 8 visualizations (300 DPI PNG)
    │   ├── 01_unemployment_trend_over_time.png
    │   ├── 02_avg_unemployment_by_region.png
    │   ├── 03_unemployment_distribution.png
    │   ├── 04_unemployment_boxplot_by_area.png
    │   ├── 05_employment_trend_over_time.png
    │   ├── 06_labour_participation_trend.png
    │   ├── 07_covid19_period_comparison.png
    │   └── 08_correlation_heatmap.png
    │
    └── reports/                        ← Analytical reports
        ├── data_quality_report.txt
        ├── regional_unemployment_summary.csv
        ├── covid_comparison.csv
        ├── key_findings.txt
        └── viva_questions.md
```

---

## Data Cleaning

The following cleaning steps were applied to the raw CSV:

1. **Strip column header whitespace** – Raw column names contained leading spaces (e.g., `' Date'`, `' Frequency'`). Cleaned using `df.columns.str.strip()`.
2. **Remove blank trailing rows** – 28 completely empty rows at file end dropped using `dropna(how='all')`.
3. **Trim categorical strings** – `Region`, `Frequency`, and `Area` values trimmed with `.str.strip()`.
4. **Date parsing** – `Date` converted from `DD-MM-YYYY` strings to `datetime64[ns]` using `pd.to_datetime(format="%d-%m-%Y")`.
5. **Feature engineering** – Added `Year`, `Month`, `Month_Name`, `YearMonth`, and `Period` (Pre-COVID / COVID Period) columns.
6. **Type validation** – All numeric columns confirmed as `float64` using `pd.to_numeric()`.
7. **Sorted** – By `Date`, `Region`, `Area` for consistent sequential processing.

**Post-cleaning:** 740 valid observations, 0 missing values, 0 duplicates.

---

## EDA Performed

1. **Initial inspection** – Shape, column names, dtypes, `info()`, `describe()`, null counts, duplicates.
2. **Time-series analysis** – Monthly aggregated unemployment rate, employment count, and LPR trends.
3. **Regional analysis** – Mean, median, min, max, and std unemployment by 28 states/UTs with ranked table.
4. **Distribution analysis** – Histogram and KDE of all 740 unemployment rate observations.
5. **Area comparison** – Rural vs Urban boxplots and statistical comparison.
6. **COVID-19 period analysis** – Pre-COVID (May 2019–Feb 2020) vs COVID Period (Mar–Jun 2020).
7. **Correlation analysis** – Pearson correlation matrix and heatmap for 3 key numerical variables.

---

## Visualizations

| # | Filename | Description |
|:---|:---|:---|
| 1 | `01_unemployment_trend_over_time.png` | Line chart of monthly national mean unemployment with COVID lockdown shading and peak annotation |
| 2 | `02_avg_unemployment_by_region.png` | Color-coded ranked horizontal bar chart for all 28 states/UTs |
| 3 | `03_unemployment_distribution.png` | Histogram + KDE with mean, median, and Q3 vertical markers |
| 4 | `04_unemployment_boxplot_by_area.png` | Boxplot + strip plot comparing Rural vs Urban unemployment |
| 5 | `05_employment_trend_over_time.png` | Time series of total, rural, and urban employed population (millions) |
| 6 | `06_labour_participation_trend.png` | LPR time series for overall, rural, and urban sectors |
| 7 | `07_covid19_period_comparison.png` | Grouped horizontal bar chart: Pre-COVID vs COVID Period across top states |
| 8 | `08_correlation_heatmap.png` | Annotated Pearson correlation heatmap for 3 numerical indicators |

---

## COVID-19 Analysis

**Period Definitions:**
- **Pre-COVID Baseline**: May 31, 2019 – February 29, 2020 (10 months, 536 observations)
- **COVID Lockdown Period**: March 31, 2020 – June 30, 2020 (4 months, 204 observations)

**Rationale:** India's national lockdown was announced on March 24, 2020. The data from March 2020 onward captures the immediate labor market shock of the lockdown.

**Results (actual calculated values):**

| Metric | Pre-COVID | COVID Period | Change |
|:---|:---|:---|:---|
| Mean Unemployment Rate | 9.51% | 17.77% | +8.26 pp (+86.91%) |

> ⚠️ **Causal Disclaimer:** The observed increase in unemployment during the COVID period is correlational with the lockdown timing. Multiple economic factors influence unemployment simultaneously. This analysis describes observed changes and does not claim exclusive causation.

---

## Key Findings

All findings below are based on **actual calculated values** from the dataset.

1. **National mean unemployment rate**: **11.79%** across 740 observations (median: 8.35%).
2. **COVID-19 lockdown impact**: Unemployment increased from **9.51%** (Pre-COVID) to **17.77%** during the lockdown period — an absolute increase of +8.26 percentage points (+86.91%).
3. **Regional leaders (highest avg unemployment)**: Tripura (28.35%), Haryana (26.28%), Jharkhand (20.59%).
4. **Regional leaders (lowest avg unemployment)**: Meghalaya (4.80%), Odisha (5.66%), Assam (6.43%).
5. **Urban vs Rural**: Urban sectors recorded a slightly higher mean unemployment rate than rural sectors.
6. **Labour participation**: LPR declined during the lockdown period, indicating a discouraged-worker effect alongside rising unemployment.
7. **Distribution**: Unemployment rates follow a right-skewed distribution — most observations cluster below 10%, while lockdown-era spikes create a long right tail.

---

## Limitations

- Dataset ends June 2020 — does not capture second/third COVID waves or recovery.
- Monthly state-level aggregate estimates; individual-level microdata not available.
- Purely descriptive EDA — no causal inference capability without econometric controls.
- Seasonal agricultural and regional economic patterns are not controlled for.
- CMIE estimates are survey-based projections with inherent sampling variability.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Python script (generates all charts and reports)
```bash
python src/unemployment_analysis.py
```

### 3. Open the Jupyter Notebook
```bash
cd notebooks
jupyter notebook unemployment_analysis.ipynb
```

> **Note:** The dataset (`data/Unemployment in India.csv`) is already included in this repository — no manual download is required.

---

## Conclusion

This project provides a complete, professional, and reproducible exploratory analysis of unemployment in India during May 2019 – June 2020. The data reveals a relatively stable baseline unemployment rate in the pre-pandemic period, followed by a substantial measured increase coinciding with the COVID-19 national lockdown of March 2020. Significant regional variation exists across India's 28 states and union territories, and urban sectors show modestly higher average unemployment rates than rural ones. All findings are evidence-based, and appropriate caution is applied to avoid overstating causal conclusions from descriptive data.

---

*Task 2 – Horizon TechX Internship | Tharani Natarajan | IFET College of Engineering, Dept. of AI & Data Science*
