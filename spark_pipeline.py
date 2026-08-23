import sys
import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, window, expr, sqrt, avg, max as spark_max, abs as spark_abs, 
    current_timestamp, to_timestamp, year, month, day, 
    collect_list, pandas_udf, udf
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, ArrayType
)

# --- ENVIRONMENT SETUP ---
# Ensure Spark workers use the same Python version as the driver
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Java 21 compatibility flags for Arrow/Pandas UDFs
# These are strictly required when running on Java 17+
JAVA_OPTS = (
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.net.util=ALL-UNNAMED "
    "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED "
    "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED "
    "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED "
    "-Dio.netty.tryReflectionSetAccessible=true"
)

# Enforce packages and JVM options via submit args (required for Java 17+)
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    f"--packages io.delta:delta-spark_2.12:3.0.0,"
    f"org.apache.spark:spark-avro_2.12:3.5.0,"
    f"org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 "
    f"--conf 'spark.driver.extraJavaOptions={JAVA_OPTS}' "
    f"--conf 'spark.executor.extraJavaOptions={JAVA_OPTS}' "
    f"--conf 'spark.sql.execution.arrow.pyspark.enabled=true' "
    "pyspark-shell"
)
BASE_PATH = "/Users/inesbenyghzer/.gemini/antigravity/scratch/vehicle_comfort_pipeline"
DATA_PATH = os.path.join(BASE_PATH, "data")
CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints")

SAMPLING_RATE = 20  # Hz
SCHEMA_REGISTRY_URL = "http://127.0.0.1:8081"

# Path Setup (Migration to Delta)
BRONZE_DIR = os.path.join(DATA_PATH, "bronze", "chassis_sensors_delta")
SILVER_DIR = os.path.join(DATA_PATH, "silver", "chassis_sensors_delta")
GOLD_DIR = os.path.join(DATA_PATH, "gold", "chassis_comfort_delta")

def main():
    # Initialize Spark with Delta Lake and Avro support
    spark = SparkSession.builder \
        .appName("IndustrialVehicleComfortPipeline") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0,org.apache.spark:spark-avro_2.12:3.5.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED") \
        .config("spark.executor.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    # --- ISO 2631-1 WEIGHTED ACCELERATION (STANDARD UDF) ---
    @udf(DoubleType())
    def calculate_iso_weighted_acc(acc_z_list):
        """
        Computes ISO 2631-1 frequency-weighted acceleration (Wk curve).
        """
        try:
            if acc_z_list is None or len(acc_z_list) < 2:
                return 0.0
            
            # Convert to numpy and handle any potential NaNs in the input stream
            y = np.nan_to_num(np.array(acc_z_list), nan=0.0)
            n = len(y)
            
            # Remove DC component (detrend)
            y = y - np.mean(y)
            
            # FFT (rfft returns magnitudes for positive frequencies)
            # Normalize FFT by n to get amplitude-like values for RMS calculation
            # Use max(1, n/2.0) to avoid division by zero
            yf = np.abs(np.fft.rfft(y)) / max(1.0, n / 2.0)
            xf = np.fft.rfftfreq(n, 1/SAMPLING_RATE)
            
            # Wk Weights (ISO 2631-1 Vertical)
            freqs = np.array([0.5, 1.0, 2.0, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0])
            gains = np.array([0.062, 0.176, 0.643, 0.967, 1.000, 0.977, 0.892, 0.776, 0.648, 0.512, 0.409, 0.330, 0.266, 0.215, 0.176, 0.145, 0.119])
            
            # Interpolate gains for each FFT bin
            wk_gains = np.interp(xf, freqs, gains, left=0.0, right=0.0)
            
            # Apply weighting
            weighted_magnitudes = yf * wk_gains
            
            # RMS calculation
            # User formula: sqrt(mean(weighted_magnitudes**2))
            # Use nan_to_num again just in case of intermediate overflows
            weighted_rms = float(np.sqrt(np.mean(np.nan_to_num(weighted_magnitudes**2))))
            
            return weighted_rms
        except Exception:
            return 0.0

    spark.sparkContext.setLogLevel("WARN")

    # Avro Schema for deserialization (must match Producer)
    json_schema = """
    {
      "type": "record",
      "name": "ChassisSensor",
      "fields": [
        {"name": "vehicle_id", "type": "string"},
        {"name": "test_id", "type": "string"},
        {"name": "timestamp", "type": "double"},
        {"name": "speed_kmh", "type": "double"},
        {"name": "acc_z", "type": "double"},
        {"name": "suspension_mm", "type": "double"},
        {"name": "pitch_deg", "type": "double"},
        {"name": "roll_deg", "type": "double"}
      ]
    }
    """

    # --- BRONZE LAYER (Delta Sink) ---
    print("🔵 Starting Bronze Layer (Delta)...")
    bronze_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "chassis_sensors") \
        .option("startingOffsets", "latest") \
        .load()

    # Note: For production, we would use from_avro(col("value"), schema_registry_url=...)
    # Here we use the inline schema for simplicity in the local setup.
    from pyspark.sql.avro.functions import from_avro

    bronze_df = bronze_raw \
        .withColumn("ingestion_time", current_timestamp())

    # Write to Bronze Delta
    bronze_query = bronze_df.writeStream \
        .format("delta") \
        .option("checkpointLocation", os.path.join(CHECKPOINT_PATH, "bronze_delta")) \
        .outputMode("append") \
        .start(BRONZE_DIR)

    # --- SILVER LAYER (Cleaned & Parsed Delta) ---
    print("🔷 Starting Silver Layer (Data Quality + Delta)...")
    
    # Payload is 5 bytes offset if using Confluent Avro (Magic Byte + Schema ID)
    # We strip them to use standard Spark Avro functions if needed, 
    # but here we'll assume standard Avro for the prototype consistency.
    silver_parsed = bronze_df \
        .select(from_avro(expr("substring(value, 6, length(value)-5)"), json_schema).alias("data"), "ingestion_time") \
        .select("data.*", "ingestion_time")

    # DATA QUALITY: Filters and Constraints
    silver_cleaned = silver_parsed \
        .filter(col("vehicle_id").isNotNull()) \
        .filter(col("acc_z").isNotNull()) \
        .filter(col("acc_z").between(-5.0, 5.0)) \
        .filter(col("speed_kmh") >= 0) \
        .withColumn("event_time", to_timestamp(col("timestamp"))) \
        .withColumn("year", year(col("event_time"))) \
        .withColumn("month", month(col("event_time"))) \
        .withColumn("day", day(col("event_time")))

    silver_query = silver_cleaned.writeStream \
        .format("delta") \
        .partitionBy("year", "month", "day") \
        .option("checkpointLocation", os.path.join(CHECKPOINT_PATH, "silver_delta")) \
        .outputMode("append") \
        .start(SILVER_DIR)

    # --- GOLD LAYER (Optimized Metrics Delta) ---
    print("🟡 Starting Gold Layer (Pandas FFT + Delta)...")
    
    gold_windowed = silver_cleaned \
        .withWatermark("event_time", "10 seconds") \
        .groupBy(
            window(col("event_time"), "10 seconds", "2 seconds"),
            "vehicle_id", "test_id"
        ).agg(
            sqrt(avg(col("acc_z")**2)).alias("rms_acc_z"),
            spark_max(spark_abs(col("acc_z"))).alias("peak_acc_z"),
            avg("speed_kmh").alias("avg_speed"),
            collect_list("acc_z").alias("acc_z_series")
        )

    # Apply ISO Weighted Acceleration UDF
    gold_with_iso = gold_windowed \
        .withColumn("weighted_acc_z", calculate_iso_weighted_acc(col("acc_z_series"))) \
        .drop("acc_z_series")

    # Industrial Comfort Score Logic (ISO 2631-1 Based)
    # Scale: 0-100, where 100 is perfect comfort.
    # User formula: 100 - weighted_acc_z * 50, clamped 0-100.
    gold_metrics = gold_with_iso \
        .withColumn("comfort_score", 
            expr("GREATEST(0.0, LEAST(100.0, 100.0 - (weighted_acc_z * 50.0)))"))

    gold_final = gold_metrics.select(
        col("window.start").alias("start_time"),
        col("window.end").alias("end_time"),
        "vehicle_id", "test_id", "rms_acc_z", "peak_acc_z", 
        "avg_speed", "weighted_acc_z", "comfort_score"
    ).withColumn("year", year(col("start_time"))) \
     .withColumn("month", month(col("start_time"))) \
     .withColumn("day", day(col("start_time")))

    # Write Gold to Delta with Vehicle Partitioning
    gold_query = gold_final.writeStream \
        .format("delta") \
        .partitionBy("year", "month", "day", "vehicle_id") \
        .option("checkpointLocation", os.path.join(CHECKPOINT_PATH, "gold_delta")) \
        .option("mergeSchema", "true") \
        .option("overwriteSchema", "true") \
        .outputMode("append") \
        .trigger(processingTime="5 seconds") \
        .start(GOLD_DIR)

    print("\n🚀 Industrial Pipeline Active (Delta + Avro + Pandas UDF)")
    print(f"📍 Gold Layer: {GOLD_DIR}")
    
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
