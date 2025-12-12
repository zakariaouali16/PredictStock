# -*- coding: utf-8 -*-
"""
Google Stock Analysis using Apache Spark and Yahoo Finance.
Designed for Google Colab.

Updates in this version:
1.  **Bollinger Bands**: Added Upper and Lower bands for volatility analysis.
2.  **RSI (Relative Strength Index)**: Added momentum indicator calculation.
3.  **Advanced Visualizations**: Created a multi-panel financial dashboard (Price, Volume, RSI).
4.  **Trading Signals**: Logic to detect Overbought/Oversold conditions.
5.  **Fix**: Added handling for yfinance MultiIndex columns.
6.  **New Insights**: Added Seasonality (Month/Day), Intraday Volatility, and MA Crossover signals.
"""

import os
import sys
import subprocess

# --- 1. Environment Setup & Installation ---
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pyspark
    import yfinance as yf
    print("Libraries already installed.")
except ImportError:
    print("Installing pyspark and yfinance...")
    install("pyspark")
    install("yfinance")
    import pyspark
    import yfinance as yf

# --- Imports ---
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, lag, when, lit, abs as spark_abs, round as spark_round, month, dayofweek, date_format
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta

# --- 2. Initialize Spark Session ---
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("GoogleStockAnalysis") \
    .master("local[*]") \
    .config("spark.ui.port", "4050") \
    .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
print("Spark Session Created successfully.")

# --- 3. Data Extraction (Yahoo Finance) ---
ticker = "GOOGL"
# Fetch 5 years of data
start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
pdf = yf.download(ticker, start=start_date, end=end_date)
pdf.reset_index(inplace=True)

# Flatten columns if MultiIndex (newer yfinance versions return tuples)
if isinstance(pdf.columns, pd.MultiIndex):
    pdf.columns = pdf.columns.get_level_values(0)

# Clean up column names (Spark doesn't like spaces)
pdf.columns = [str(c).replace(' ', '_') for c in pdf.columns]

# Convert to Spark DataFrame
print("Converting to Spark DataFrame...")
sdf = spark.createDataFrame(pdf)
sdf = sdf.withColumnRenamed("Date", "date") \
         .withColumnRenamed("Open", "open") \
         .withColumnRenamed("High", "high") \
         .withColumnRenamed("Low", "low") \
         .withColumnRenamed("Close", "close") \
         .withColumnRenamed("Adj_Close", "adj_close") \
         .withColumnRenamed("Volume", "volume")

# --- 4. Technical Analysis (Feature Engineering) ---
print("\n--- Performing Advanced Technical Analysis ---")

# Define Windows
windowSpec = Window.orderBy("date")
windowSpec14 = Window.orderBy("date").rowsBetween(-13, 0) # For RSI
windowSpec20 = Window.orderBy("date").rowsBetween(-19, 0) # For Bollinger Bands
windowSpec50 = Window.orderBy("date").rowsBetween(-49, 0) # For SMA 50

# 1. Basic Calculations (Lag, Daily Change, Intraday Volatility)
df_calc = sdf.withColumn("prev_close", lag("close", 1).over(windowSpec)) \
             .withColumn("change", col("close") - col("prev_close")) \
             .withColumn("daily_return", (col("close") - col("prev_close")) / col("prev_close")) \
             .withColumn("intraday_range_pct", (col("high") - col("low")) / col("open") * 100) \
             .withColumn("month_num", month("date")) \
             .withColumn("day_of_week", date_format("date", "EEEE"))

# 2. RSI Components (Gain/Loss)
df_rsi_prep = df_calc.withColumn("gain", when(col("change") > 0, col("change")).otherwise(0)) \
                     .withColumn("loss", when(col("change") < 0, spark_abs(col("change"))).otherwise(0))

# 3. Aggregating Indicators
analysis_df = df_rsi_prep.withColumn("avg_gain", avg("gain").over(windowSpec14)) \
    .withColumn("avg_loss", avg("loss").over(windowSpec14)) \
    .withColumn("rs", col("avg_gain") / col("avg_loss")) \
    .withColumn("rsi", 100 - (100 / (1 + col("rs")))) \
    .withColumn("SMA_20", avg("close").over(windowSpec20)) \
    .withColumn("SMA_50", avg("close").over(windowSpec50)) \
    .withColumn("stddev_20", stddev("close").over(windowSpec20)) \
    .withColumn("upper_band", col("SMA_20") + (lit(2) * col("stddev_20"))) \
    .withColumn("lower_band", col("SMA_20") - (lit(2) * col("stddev_20")))

# Identify Crossovers (Signal Generation)
# We check if SMA20 was below SMA50 yesterday, but is above today (Golden Cross)
analysis_df = analysis_df.withColumn("prev_SMA_20", lag("SMA_20", 1).over(windowSpec)) \
                         .withColumn("prev_SMA_50", lag("SMA_50", 1).over(windowSpec)) \
                         .withColumn("crossover_signal",
                                     when((col("prev_SMA_20") < col("prev_SMA_50")) & (col("SMA_20") > col("SMA_50")), "Golden Cross (Buy)")
                                     .when((col("prev_SMA_20") > col("prev_SMA_50")) & (col("SMA_20") < col("SMA_50")), "Death Cross (Sell)")
                                     .otherwise("None"))

# Drop N/A (early rows without enough history for windows)
analysis_df = analysis_df.dropna()

# Register for SQL
analysis_df.createOrReplaceTempView("google_stock_advanced")

# --- 5. Spark SQL Insights ---

print("\n--- Market Insights (Spark SQL) ---")

# Insight 1: Yearly Summary
print("1. Yearly Summary:")
spark.sql("""
    SELECT
        year(date) as Year,
        ROUND(AVG(close), 2) as Avg_Price,
        ROUND(MAX(close), 2) as Max_Price,
        ROUND(AVG(rsi), 2) as Avg_RSI
    FROM google_stock_advanced
    GROUP BY year(date)
    ORDER BY Year DESC
""").show()

# Insight 2: Seasonality - Which months are best for Google?
print("2. Seasonality Analysis (Avg Return by Month):")
spark.sql("""
    SELECT
        month_num as Month,
        count(*) as Trading_Days,
        ROUND(AVG(daily_return) * 100, 3) as Avg_Daily_Return_Pct,
        ROUND(AVG(intraday_range_pct), 2) as Avg_Volatility_Pct
    FROM google_stock_advanced
    GROUP BY month_num
    ORDER BY Avg_Daily_Return_Pct DESC
""").show()

# Insight 3: Day of Week Analysis
print("3. Day of Week Patterns:")
spark.sql("""
    SELECT
        day_of_week as Day,
        ROUND(AVG(daily_return) * 100, 3) as Avg_Return_Pct,
        ROUND(AVG(volume)/1000000, 2) as Avg_Vol_Millions
    FROM google_stock_advanced
    GROUP BY day_of_week
    ORDER BY Avg_Return_Pct DESC
""").show()

# Insight 4: Recent Crossovers (Trend Changes)
print("4. Recent Trend Changes (Golden/Death Crosses):")
spark.sql("""
    SELECT date, close, crossover_signal
    FROM google_stock_advanced
    WHERE crossover_signal != 'None'
    ORDER BY date DESC
    LIMIT 5
""").show()

# Insight 5: Top 5 Highest Volatility Days (Intraday Swing)
print("5. Top 5 Most Volatile Days (Intraday High-Low Range):")
spark.sql("""
    SELECT date, round(open, 2) as open, round(high, 2) as high, round(low, 2) as low,
           round(intraday_range_pct, 2) as range_pct
    FROM google_stock_advanced
    ORDER BY intraday_range_pct DESC
    LIMIT 5
""").show()

# --- 6. Visualization (Professional Dashboard) ---
print("\n--- Generating Professional Dashboard ---")
plot_data = analysis_df.select("date", "close", "SMA_20", "SMA_50", "upper_band", "lower_band", "volume", "rsi").toPandas()

# Set visual style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['axes.titlesize'] = 14

# Create 3 subplots sharing x-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
fig.suptitle(f'{ticker} Technical Analysis Dashboard (Last 5 Years)', fontsize=18, fontweight='bold', y=0.95)

# --- Plot 1: Price & Bollinger Bands ---
ax1.plot(plot_data['date'], plot_data['close'], label='Close Price', color='#333333', linewidth=1.5)
ax1.plot(plot_data['date'], plot_data['upper_band'], label='Upper Bollinger Band', color='green', linestyle='--', alpha=0.5, linewidth=1)
ax1.plot(plot_data['date'], plot_data['lower_band'], label='Lower Bollinger Band', color='red', linestyle='--', alpha=0.5, linewidth=1)
ax1.fill_between(plot_data['date'], plot_data['upper_band'], plot_data['lower_band'], color='gray', alpha=0.1)
ax1.plot(plot_data['date'], plot_data['SMA_50'], label='SMA 50', color='blue', linewidth=1.5, alpha=0.8)
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.legend(loc='upper left', frameon=True)
ax1.set_title('Price Action & Bollinger Bands', loc='left')

# --- Plot 2: Volume ---
# Color volume bars based on price change (Green for up, Red for down - approximation)
colors = ['green' if r >= 0 else 'red' for r in plot_data['close'].diff().fillna(0)]
ax2.bar(plot_data['date'], plot_data['volume'], color=colors, alpha=0.7, width=1.0)
ax2.set_ylabel('Volume', fontsize=12)
ax2.set_title('Trading Volume', loc='left')

# --- Plot 3: RSI ---
ax3.plot(plot_data['date'], plot_data['rsi'], color='purple', linewidth=1.5)
ax3.axhline(70, linestyle='--', alpha=0.5, color='red')
ax3.axhline(30, linestyle='--', alpha=0.5, color='green')
ax3.fill_between(plot_data['date'], plot_data['rsi'], 70, where=(plot_data['rsi'] >= 70), facecolor='red', alpha=0.3)
ax3.fill_between(plot_data['date'], plot_data['rsi'], 30, where=(plot_data['rsi'] <= 30), facecolor='green', alpha=0.3)
ax3.set_ylabel('RSI (14)', fontsize=12)
ax3.set_title('Relative Strength Index', loc='left')
ax3.set_ylim(0, 100)

# Format X-Axis
ax3.set_xlabel('Date', fontsize=12)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.92) # Make room for title
plt.show()

# --- 7. Conclusion ---
spark.stop()
print("Analysis & Dashboard Generation Complete.")