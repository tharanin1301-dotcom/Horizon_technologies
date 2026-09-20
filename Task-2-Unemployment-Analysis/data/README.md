# Dataset Documentation: Unemployment in India

## 1. Dataset Overview
- **Dataset Name**: Unemployment in India Dataset
- **Primary Source**: Centre for Monitoring Indian Economy (CMIE)
- **Public Archive**: [adduadnanali/Unemployment-Analysis-in-India (GitHub)](https://raw.githubusercontent.com/adduadnanali/Unemployment-Analysis-in-India/main/Unemployment%20in%20India.csv)
- **File Name**: `Unemployment in India.csv`
- **Data Collection Frequency**: Monthly
- **Time Period Covered**: May 31, 2019 to June 30, 2020 (14 consecutive months)
- **Geographic Coverage**: 28 Indian States & Union Territories
- **Area Coverage**: Rural and Urban areas
- **Raw Dimensions**: 768 rows × 7 columns (including 28 trailing blank lines)
- **Cleaned Dimensions**: 740 observations × 7 columns

---

## 2. Variables and Data Dictionary

| Column Name | Data Type (Raw) | Cleaned Type | Description |
| :--- | :--- | :--- | :--- |
| `Region` | `object` | `category` / `str` | Name of the Indian State or Union Territory |
| `Date` | `object` | `datetime64[ns]` | Reporting date (DD-MM-YYYY format, month-end observations) |
| `Frequency` | `object` | `category` / `str` | Measurement cadence (Monthly) |
| `Estimated Unemployment Rate (%)` | `float64` | `float64` | Estimated proportion of the labor force that is unemployed (%) |
| `Estimated Employed` | `float64` | `int64` / `float64` | Estimated aggregate count of employed individuals |
| `Estimated Labour Participation Rate (%)` | `float64` | `float64` | Ratio of labor force to the total working-age population (%) |
| `Area` | `object` | `category` / `str` | Sector classification (`Rural` or `Urban`) |

---

## 3. Data Lineage and Provenance
The dataset originates from the Consumer Pyramids Household Survey conducted by the **Centre for Monitoring Indian Economy (CMIE)**, widely regarded as India's premier independent economic research firm. During the COVID-19 pandemic, CMIE high-frequency indicators were extensively utilized by researchers, policy analysts, and journalists to track real-time labor market shocks in India.

This specific compiled dataset was released as a public research asset on Kaggle and mirrored on GitHub for data science education and exploratory labor economics analysis.

---

## 4. Download and Reproduction Instructions
The raw CSV file is stored locally at `data/Unemployment in India.csv`.

To independently fetch or refresh this dataset from the source repository using Python:
```python
import urllib.request
url = "https://raw.githubusercontent.com/adduadnanali/Unemployment-Analysis-in-India/main/Unemployment%20in%20India.csv"
urllib.request.urlretrieve(url, "data/Unemployment in India.csv")
```

---

## 5. Cleaning Assumptions and Data Hygiene Notes
1. **Header Whitespace**: The raw CSV contains leading spaces in column headers (`' Date'`, `' Frequency'`, `' Estimated Unemployment Rate (%)'`, etc.). Automated cleaning strips all column whitespace.
2. **Trailing Blank Rows**: 28 empty rows exist at the end of the raw CSV file (rows 740 to 767) containing only `NaN`. These rows are completely dropped (`dropna(how='all')`).
3. **Categorical Whitespace**: The `Frequency` column contains irregular string formatting (`' Monthly'` vs `'Monthly'`). All categorical strings (`Region`, `Frequency`, `Area`) are trimmed.
4. **Date Parsing**: Date strings are parsed using day-first format (`%d-%m-%Y`) into Pandas datetime timestamps.
5. **COVID-19 Period Definition**:
   - **Pre-COVID Period**: 31-05-2019 to 29-02-2020 (10 months, baseline conditions)
   - **COVID-19 Lockdown Period**: 31-03-2020 to 30-06-2020 (4 months, capturing the March 24 national lockdown and severe labor contraction)

---

## 6. Repository Policy
The raw CSV (`Unemployment in India.csv`) is preserved within this repository under `data/` to ensure full offline reproducibility.
