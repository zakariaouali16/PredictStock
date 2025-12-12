import sys
import subprocess
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lead, lit, when
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Initialize Spark
spark = SparkSession.builder \
    .appName("StockPrediction_StrictSplit") \
    .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

def run_prediction(ticker, file_path, bucket_name):
    print(f"\n{'='*60}")
    print(f"Processing Stock (Strict Mode): {ticker}")
    print(f"{'='*60}")

    # 1. Load Data
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # 2. Clean & Cast
    df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd")) \
           .withColumn("Close", col("Close").cast(FloatType())) \
           .withColumn("Open", col("Open").cast(FloatType())) \
           .withColumn("High", col("High").cast(FloatType())) \
           .withColumn("Low", col("Low").cast(FloatType())) \
           .withColumn("Volume", col("Volume").cast(FloatType())) \
           .withColumn("Ticker_ID", lit(ticker))
    
    df = df.na.drop()

    # 3. Feature Engineering
    # Target: Next Day Close
    windowSpec = Window.partitionBy("Ticker_ID").orderBy("Date")
    df = df.withColumn("label", lead("Close", 1).over(windowSpec))
    data = df.na.drop()

    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data_vectorized = assembler.transform(data)
    
    # 4. STRICT Time-Series Split
    # We collect dates to find the exact cutoff
    dates_sorted = data_vectorized.select("Date").orderBy("Date").rdd.map(lambda r: r[0]).collect()
    split_idx = int(len(dates_sorted) * 0.8)
    split_date = dates_sorted[split_idx]

    print(f"Split Date: {split_date}")
    
    train_data = data_vectorized.filter(col("Date") < split_date)
    test_data = data_vectorized.filter(col("Date") >= split_date)

    # 5. Model Training (Only on Train Data)
    lr = LinearRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(train_data)

    # 6. Evaluation (Only on Test Data)
    predictions = lr_model.transform(test_data)
    
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Test RMSE: {rmse:.4f}")

    # 7. Visualization (The Fix)
    # We need Training data (for context) AND Test predictions
    # Note: We do NOT predict on training data for the plot.
    
    train_rows = train_data.select("Date", "label").orderBy("Date").collect()
    test_rows = predictions.select("Date", "label", "prediction").orderBy("Date").collect()

    # Unpack for plotting
    train_dates = [row['Date'] for row in train_rows]
    train_actuals = [row['label'] for row in train_rows]

    test_dates = [row['Date'] for row in test_rows]
    test_actuals = [row['label'] for row in test_rows]
    test_preds = [row['prediction'] for row in test_rows]

    plt.figure(figsize=(14, 7))

    # Plot Training Data (Context) - Grey/Green
    plt.plot(train_dates, train_actuals, label='Training Data (History)', color='grey', alpha=0.5)

    # Plot Test Data (Actual) - Blue
    plt.plot(test_dates, test_actuals, label='Test Data (Actual)', color='blue', linewidth=1.5)

    # Plot Predictions - Red Dashed
    plt.plot(test_dates, test_preds, label='Model Prediction', color='red', linestyle='--', linewidth=1.5)

    # Add Vertical Line at Split
    plt.axvline(x=split_date, color='black', linestyle=':', linewidth=2, label="Train/Test Split")

    plt.title(f"{ticker} Forecast (RMSE: {rmse:.2f})\nStrict Train/Test Separation")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    
    # Format Date Axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    # Save and Upload
    plot_filename = f"{ticker}_strict_prediction.png"
    plt.savefig(plot_filename)
    plt.close()

    if bucket_name:
        gcs_path = f"gs://{bucket_name}/output/{plot_filename}"
        print(f"Uploading to {gcs_path}...")
        subprocess.check_call(["gsutil", "cp", plot_filename, gcs_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", required=True)
    args = parser.parse_args()

    companies = {
        "GOOGLE": f"gs://{args.bucket_name}/data/GOOGL.csv"
    }

    for name, path in companies.items():
        run_prediction(name, path, args.bucket_name)

    spark.stop()