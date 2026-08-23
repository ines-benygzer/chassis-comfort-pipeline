import sys
import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window, expr, sqrt, avg, max as spark_max, abs as spark_abs, 
    current_timestamp, to_timestamp, year, month, day, 
    collect_list, udf
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, ArrayType
)

# --- CONFIGURATION ---
BASE_PATH = "/Users/inesbenyghzer/.gemini/antigravity/scratch/vehicle_comfort_pipeline"
DATA_PATH = os.path.join(BASE_PATH, "data")
CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints")

SAMPLING_RATE = 20  # Hz

# Path Setup
BRONZE_DIR = os.path.join(DATA_PATH, "bronze", "chassis_sensors")
SILVER_DIR = os.path.join(DATA_PATH, "silver", "chassis_sensors_clean")
GOLD_DIR = os.path.join(DATA_PATH, "gold", "chassis_comfort_metrics")

# --- FFT UDF (Simplified for Stability) ---
def compute_fft_simple(acc_z_list):
    """Compute FFT metrics from acceleration array"""
    if acc_z_list is None or len(acc_z_list) < 4:
        return (0.0, 0.0, 0.0)
    
    try:
        y = np.array(acc_z_list, dtype=float)
        n = len(y)
        
        # Real FFT
        yf = np.fft.rfft(y)
        xf = np.fft.rfftfreq(n, 1/SAMPLING_RATE)
        
        # Power spectrum
        powers = np.abs(yf)**2
        
        # Total energy
        total_energy = float(np.sum(powers))
        
        # Dominant frequency (skip DC)
        if len(powers) > 1:
            dom_idx = int(np.argmax(powers[1:])) + 1
            dominant_freq = float(xf[dom_idx])
        else:
            dominant_freq = 0.0
        
        # Band energy 4-8 Hz
        mask = (xf >= 4.0) & (xf <= 8.0)
        band_energy = float(np.sum(powers[mask]))
        
        return (dominant_freq, total_energy, band_energy)
    except Exception as e:
        return (0.0, 0.0, 0.0)

# Register UDF
fft_udf = udf(compute_fft_simple, StructType([
    StructField("dominant_freq", DoubleType()),
    StructField("total_energy", DoubleType()),
    StructField("band_energy_4_8", DoubleType())
]))

def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("IndustrialVehicleComfortPipeline") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.python.worker.memory", "512m") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Schema
    schema = StructType([
        StructField("vehicle_id", StringType()),
        StructField("test_id", StringType()),
        StructField("timestamp", DoubleType()),
        StructField("speed_kmh", DoubleType()),
        StructField("acc_z", DoubleType()),
        StructField("suspension_mm", DoubleType()),
        StructField("pitch_deg", DoubleType()),
        StructField("roll_deg", DoubleType())
    ])

    # --- BRONZE LAYER ---
    bronze_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "chassis_sensors") \
        .option("startingOffsets", "latest") \
        .load()

    bronze_df = bronze_raw \
        .withColumn("ingestion_timestamp", current_timestamp()) \
        .withColumn("year", year(col("ingestion_timestamp"))) \
        .withColumn("month", month(col("ingestion_timestamp"))) \
        .withColumn("day", day(col("ingestion_timestamp")))

    bronze_query = bronze_df.writeStream \
        .format("parquet") \
        .option("path", BRONZE_DIR) \
        .option("checkpointLocation", os.path.join(CHECKPOINT_PATH, "bronze")) \
        .partitionBy("year", "month", "day") \
        .outputMode("append") \
        .start()

    # --- SILVER LAYER ---
    silver_df = bronze_df \
        .selectExpr("CAST(value AS STRING) as json_payload", "ingestion_timestamp") \
        .select(from_json(col("json_payload"), schema).alias("data"), "ingestion_timestamp") \
        .select("data.*", "ingestion_timestamp") \
        .withColumn("event_time", to_timestamp(col("timestamp"))) \
        .withColumn("year", year(col("event_time"))) \
        .withColumn("month", month(col("event_time"))) \
        .withColumn("day", day(col("event_time"))) \
        .withWatermark("event_time", "10 seconds")

    silver_query = silver_df.writeStream \
        .format("parquet") \
        .option("path", SILVER_DIR) \
        .option("checkpointLocation", os.path.join(CHECKPOINT_PATH, "silver")) \
        .partitionBy("year", "month", "day") \
        .outputMode("append") \
        .start()

    # --- GOLD LAYER ---
    gold_windowed = silver_df.groupBy(
        window(col("event_time"), "10 seconds", "2 seconds"),
        col("vehicle_id"),
        col("test_id")
    ).agg(
        sqrt(avg(col("acc_z")**2)).alias("rms_acc_z"),
        spark_max(spark_abs(col("acc_z"))).alias("peak_acc_z"),
        avg("speed_kmh").alias("avg_speed"),
        collect_list("acc_z").alias("acc_z_series")
    )

    # Apply FFT UDF
    gold_with_fft = gold_windowed \
        .withColumn("fft_metrics", fft_udf(col("acc_z_series"))) \
        .select("*", "fft_metrics.*") \
        .drop("fft_metrics", "acc_z_series")

    # Comfort Score
    gold_final = gold_with_fft \
        .withColumn("rms_norm", expr("LEAST(rms_acc_z / 2.0, 1.0)")) \
        .withColumn("peak_norm", expr("LEAST(peak_acc_z / 3.0, 1.0)")) \
        .withColumn("band_norm", expr("LEAST(band_energy_4_8 / 50.0, 1.0)")) \
        .withColumn("comfort_score",
            expr("GREATEST(0.0, 100.0 - (0.4 * rms_norm + 0.2 * peak_norm + 0.4 * band_norm) * 100.0)")) \
        .select(
            col("window.start").alias("start_time"),
            col("window.end").alias("end_time"),
            "vehicle_id",
            "test_id",
            "rms_acc_z",
            "peak_acc_z",
            "avg_speed",
            "dominant_freq",
            "total_energy",
            "band_energy_4_8",
            "comfort_score"
        ) \
        .withColumn("year", year(col("start_time"))) \
        .withColumn("month", month(col("start_time"))) \
        .withColumn("day", day(col("start_time")))

    # Write Gold
    gold_query = gold_final.writeStream \
        .format("parquet") \
        .option("path", GOLD_DIR) \
        .option("checkpointLocation", os.path.join(CHECKPOINT_PATH, "gold")) \
        .partitionBy("year", "month", "day") \
        .outputMode("append") \
        .start()

    # Console output
    console_query = gold_final.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
