# Viva Questions & Answers
## Task 2 – Unemployment Analysis Using Python
### Horizon TechX Internship | Tharani Natarajan | IFET College of Engineering

---

## Q1. What is unemployment analysis?

**Answer:**
Unemployment analysis is the study of unemployment trends, patterns, and distributions within a population or geographic region over a period of time. It involves collecting, cleaning, and statistically analyzing data on the proportion of the labor force that is without work and actively seeking employment. In this project, we analyzed state-level monthly unemployment rates across 28 Indian states and union territories from May 2019 to June 2020.

---

## Q2. What is Exploratory Data Analysis (EDA)?

**Answer:**
EDA is the process of analyzing a dataset to summarize its main characteristics before applying formal modeling. It uses statistical summaries and visualizations to understand the data's structure, detect patterns, identify outliers, check assumptions, and discover relationships between variables. In this project, EDA included descriptive statistics, time-series analysis, regional comparisons, and correlation analysis.

---

## Q3. Why is data cleaning required?

**Answer:**
Raw data is rarely analysis-ready. It often contains missing values, formatting errors, inconsistent text, duplicate rows, and incorrect data types. In our dataset:
- Column headers had leading whitespace (e.g., `' Date'`).
- 28 entirely blank trailing rows existed.
- The `Date` column was stored as a string, not datetime.
- The `Frequency` column had mixed whitespace (`' Monthly'` vs `'Monthly'`).
Without cleaning, calculations and visualizations would produce incorrect or misleading results.

---

## Q4. What is the unemployment rate?

**Answer:**
The unemployment rate is the percentage of people in the labor force who do not have a job but are actively seeking employment.

**Formula:**  
`Unemployment Rate (%) = (Unemployed Persons / Labor Force) × 100`

In our dataset, it is labelled as **"Estimated Unemployment Rate (%)"** and computed at the state and area (Rural/Urban) level on a monthly basis.

---

## Q5. What is the labour participation rate (LPR)?

**Answer:**
The Labour Participation Rate (LPR) is the percentage of the working-age population that is either currently employed or actively looking for work (i.e., part of the labor force).

**Formula:**  
`LPR (%) = (Labor Force / Working-Age Population) × 100`

A declining LPR during the COVID-19 lockdown indicates that many workers exited the labor force entirely — a **discouraged-worker effect** — rather than remaining as actively unemployed.

---

## Q6. Why do we convert dates to datetime format?

**Answer:**
Date columns are often stored as plain text strings in CSV files. Converting them to Pandas `datetime64` format using `pd.to_datetime()` enables:
- Chronological sorting and filtering by date ranges.
- Extraction of temporal features (Year, Month, Quarter).
- Defining COVID-period boundaries using `pd.Timestamp` comparisons.
- Plotting time series with proper date axes in Matplotlib.

In our dataset, dates were formatted as `DD-MM-YYYY` and parsed using `format="%d-%m-%Y"`.

---

## Q7. Why do we use Pandas?

**Answer:**
Pandas is Python's primary data manipulation library. We use it because:
- It provides the `DataFrame` structure — a 2D labeled table ideal for tabular CSV data.
- It supports fast data loading (`read_csv()`), cleaning (`dropna()`, `str.strip()`), and grouping (`groupby()`).
- It integrates seamlessly with Matplotlib, Seaborn, and NumPy.
- It efficiently handles datetime indexing and period-based time series operations.

---

## Q8. Why do we use Matplotlib?

**Answer:**
Matplotlib is Python's foundational plotting library. We use it because:
- It provides fine-grained control over every chart element (axes, labels, annotations, colors, figure size).
- It supports saving high-resolution figures (300 DPI) using `savefig()`.
- All Seaborn charts are built on Matplotlib's backend, so both libraries work together.
- It enables professional, publication-quality chart customization suitable for internship reports.

---

## Q9. Why do we use Seaborn?

**Answer:**
Seaborn is a statistical visualization library built on Matplotlib. We use it because:
- It provides built-in support for statistical plots like boxplots, distribution plots, heatmaps, and violin plots with minimal code.
- It automatically handles grouping variables (`hue`, `x`, `y`) for comparative visualizations.
- It has attractive default themes (`set_theme(style="whitegrid")`) that produce clean, professional charts.
- `sns.histplot()` with `kde=True` draws both histogram bars and a smooth kernel density estimate simultaneously.

---

## Q10. What is a time-series trend?

**Answer:**
A time-series trend is the general direction of change in a variable measured over successive time periods. In this project, the time series is the monthly national mean unemployment rate from May 2019 to June 2020. By plotting it as a line chart, we can observe:
- A relatively stable baseline during May 2019 – February 2020.
- A sharp upward spike in unemployment from March 2020 onward, coinciding with the COVID-19 national lockdown.

---

## Q11. How do you calculate the average unemployment rate?

**Answer:**
Using Pandas `groupby()` and `.mean()`:

```python
# National overall mean
national_mean = df["Estimated Unemployment Rate (%)"].mean()

# Monthly mean over time
monthly = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

# By state/region
regional = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()
```

In our dataset, the national mean across all 740 observations is **11.79%**.

---

## Q12. How do you compare unemployment across regions?

**Answer:**
We use `groupby("Region").agg()` to compute multiple statistics (mean, median, min, max, std) for each of the 28 states and union territories. We then sort them by mean unemployment rate in descending order and display a ranked horizontal bar chart, with color encoding indicating relative severity. This approach is **descriptive** — it identifies statistical differences without implying causal factors.

---

## Q13. What is correlation?

**Answer:**
Correlation measures the statistical relationship (direction and strength) between two numerical variables. The **Pearson correlation coefficient (r)** ranges from -1 to +1:
- **r = +1**: Perfect positive relationship (both variables increase together)
- **r = -1**: Perfect negative relationship (one increases as the other decreases)
- **r ≈ 0**: No linear relationship

In our project, we computed correlations among: Unemployment Rate (%), Estimated Employed, and Labour Participation Rate (%).

---

## Q14. Does correlation prove causation?

**Answer:**
**No.** Correlation describes the statistical co-movement between two variables, not a cause-and-effect relationship. For example, if unemployment rate and labour participation rate are negatively correlated, it does not mean that unemployment *causes* people to leave the labor force. Both variables may be simultaneously influenced by a third factor — such as an economic shock like a national lockdown — without one causing the other directly. Establishing causation requires controlled experiments or formal econometric methods.

---

## Q15. How did you analyze the COVID-19 impact?

**Answer:**
We:
1. First inspected the actual date range of the dataset (May 2019 – June 2020) to confirm it spans both sides of the March 24, 2020 lockdown announcement.
2. Classified each observation as **Pre-COVID** (before March 2020) or **COVID Period** (March–June 2020) using `np.where()`.
3. Calculated descriptive statistics for each period using `groupby("Period")`.
4. Quantified the absolute change (+8.26 percentage points) and relative change (+86.91%) in mean unemployment.
5. Created a comparative visualization showing both periods side-by-side for major states.
6. Noted that multiple economic factors influence unemployment, so we avoided claiming exclusive causation.

---

## Q16. What is a histogram?

**Answer:**
A histogram is a bar chart that shows the frequency distribution of a continuous numerical variable by dividing the data into equal-width intervals (bins) and counting how many observations fall in each bin. In this project, we used a histogram with a KDE (Kernel Density Estimate) overlay to visualize the distribution of unemployment rates across all 740 observations. The chart revealed a **right-skewed distribution** — most rates are low, but a few extreme values (lockdown spikes) pull the distribution rightward.

---

## Q17. What is a boxplot?

**Answer:**
A boxplot (box-and-whisker plot) summarizes a variable's distribution through five statistics:
- **Minimum** (lower whisker)
- **Q1** — 25th percentile (lower box edge)
- **Median** — 50th percentile (center line)
- **Q3** — 75th percentile (upper box edge)
- **Maximum** (upper whisker)
- **Outliers** are shown as individual points beyond the whiskers.

We used boxplots to compare the unemployment rate distributions between Rural and Urban sectors, revealing differences in median rates and spread.

---

## Q18. What is a heatmap?

**Answer:**
A heatmap is a 2D grid visualization where each cell's color intensity represents a numerical value. In this project, we used a **correlation heatmap** to display the Pearson correlation coefficients between Unemployment Rate, Employment Count, and Labour Participation Rate. The `coolwarm` color map shows positive correlations in red and negative correlations in blue, with annotated values in each cell for precise reading.

---

## Q19. What are the limitations of this analysis?

**Answer:**
1. The dataset ends in June 2020 — the second and third COVID waves and post-pandemic recovery are not captured.
2. Data is aggregated at the state/monthly level; individual-level economic behavior is not observable.
3. All relationships are correlational — formal causal inference requires econometric controls.
4. Seasonal agricultural employment patterns, pre-existing state-level conditions, and policy interventions affect unemployment independently and are not controlled.
5. The CMIE estimates are survey-based projections; sampling errors are possible.

---

## Q20. What did you learn from this project?

**Answer:**
This project provided hands-on experience in:
1. **Real-world data cleaning** — handling whitespace, type conversions, and missing data professionally.
2. **Pandas proficiency** — `groupby()`, `agg()`, `pd.to_datetime()`, `pd.Period`, and `np.where()`.
3. **Matplotlib and Seaborn** — creating 8 publication-quality charts with proper labeling and annotations.
4. **Statistical thinking** — computing and interpreting means, medians, standard deviations, and percentiles.
5. **COVID-19 economic context** — understanding how labor market indicators respond to large economic shocks.
6. **Professional reporting** — generating structured data quality reports, regional summaries, and evidence-based insights without fabricating or overstating findings.
7. **Modular Python design** — organizing code into well-defined functions following software engineering best practices.
