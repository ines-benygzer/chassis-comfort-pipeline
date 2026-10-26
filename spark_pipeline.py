import sys
import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, window, expr, sqrt, avg, max as spark_max, min as spark_min, abs as spark_abs, 
    current_timestamp, to_timestamp, to_date, year, month, day, 
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
BASE_PATH = os.getenv("BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_PATH, "data"))
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", os.path.join(BASE_PATH, "checkpoints"))

SAMPLING_RATE = 20  # Hz
SCHEMA_REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL", "http://127.0.0.1:8081")

# Path Setup (Migration to Delta)
BRONZE_DIR = os.getenv("BRONZE_DIR", os.path.join(DATA_PATH, "bronze", "chassis_sensors_delta"))
SILVER_DIR = os.getenv("SILVER_DIR", os.path.join(DATA_PATH, "silver", "chassis_sensors_delta"))
GOLD_DIR = os.getenv("GOLD_DIR", os.path.join(DATA_PATH, "gold", "chassis_comfort_delta"))

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

    # --- FULL MULTI-AXIAL ISO 2631-1:1997 COMFORT & SHOCK EVALUATION (UDF) ---
    iso_multi_axial_schema = StructType([
        StructField("weighted_acc_z", DoubleType(), False),  # a_wz (vertical, Wk filter, m/s^2)
        StructField("weighted_acc_x", DoubleType(), False),  # a_wx (longitudinal, Wd filter, m/s^2)
        StructField("weighted_acc_y", DoubleType(), False),  # a_wy (lateral, Wd filter, m/s^2)
        StructField("total_acc_v", DoubleType(), False),     # a_v = sqrt(kx^2*awx^2 + ky^2*awy^2 + kz^2*awz^2)
        StructField("vdv_total", DoubleType(), False),       # Multi-axial VDV = (VDVx^4 + VDVy^4 + VDVz^4)^0.25 (m/s^1.75)
        StructField("vdv_acc_z", DoubleType(), False),       # Vertical shock dosage VDV_z
        StructField("crest_factor_z", DoubleType(), False),  # Crest Factor on vertical
        StructField("dominant_freq_z", DoubleType(), False), # Dominant frequency on vertical (Hz)
        StructField("shock_event", StringType(), False),     # Shock alert flag
        StructField("iso_comfort_tier", StringType(), False),# ISO 2631-1 comfort classification from a_v
    ])

    @udf(iso_multi_axial_schema)
    def calculate_triaxial_iso_metrics(acc_x_list, acc_y_list, acc_z_list):
        """
        Full Multi-Axial ISO 2631-1:1997 human vibration comfort evaluation:
          1. Vertical (z-axis): Wk filter (peak sensitivity at 4-8 Hz for spine compression).
          2. Longitudinal (x-axis) & Lateral (y-axis): Wd filter (peak sensitivity at 0.5-2 Hz for vestibular/sway).
          3. Point Vibration Total Value (a_v):
             a_v = sqrt(kx^2 * a_wx^2 + ky^2 * a_wy^2 + kz^2 * a_wz^2), with kx=1.0, ky=1.0, kz=1.0 (seated).
          4. Tri-Axial Vibration Dose Value (VDV_total):
             VDV_total = (VDV_x^4 + VDV_y^4 + VDV_z^4)^(1/4)
          5. Official ISO 2631-1 Annex C Scale mapped to total vibration a_v.
        """
        default_res = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, "NOMINAL", "Comfortable")
        try:
            if acc_z_list is None or len(acc_z_list) < 4:
                return default_res
            
            n = len(acc_z_list)
            dt = 1.0 / SAMPLING_RATE
            
            # Wk Weights (ISO 2631-1 Vertical, peak 4-8 Hz)
            wk_freqs = np.array([0.5, 1.0, 2.0, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0])
            wk_gains = np.array([0.062, 0.176, 0.643, 0.967, 1.000, 0.977, 0.892, 0.776, 0.648, 0.512, 0.409, 0.330, 0.266, 0.215, 0.176, 0.145, 0.119])
            
            # Wd Weights (ISO 2631-1 Horizontal, peak 0.5-2 Hz)
            wd_freqs = np.array([0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0])
            wd_gains = np.array([0.214, 0.316, 0.468, 0.638, 0.794, 0.912, 1.000, 0.912, 0.785, 0.631, 0.501, 0.398, 0.313, 0.248, 0.197, 0.155, 0.123, 0.098, 0.078, 0.062, 0.049, 0.039, 0.031])

            def filter_axis(arr, freqs, gains):
                y = np.nan_to_num(np.array(arr, dtype=float), nan=0.0)
                y_detrend = y - np.mean(y)
                yf = np.fft.rfft(y_detrend)
                xf = np.fft.rfftfreq(len(y), dt)
                w_interp = np.interp(xf, freqs, gains, left=0.0, right=0.0)
                sig_w = np.fft.irfft(yf * w_interp, len(y))
                rms = float(np.sqrt(np.mean(sig_w**2)))
                vdv = float((dt * np.sum(sig_w**4))**0.25)
                return sig_w, rms, vdv, yf, xf

            # 1. Vertical Filtering (Wk)
            sig_wz, a_wz, vdv_z, yf_z, xf_z = filter_axis(acc_z_list, wk_freqs, wk_gains)
            
            # 2. Longitudinal Filtering (Wd)
            acc_x = acc_x_list if (acc_x_list and len(acc_x_list) == n) else [0.0]*n
            _, a_wx, vdv_x, _, _ = filter_axis(acc_x, wd_freqs, wd_gains)
            
            # 3. Lateral Filtering (Wd)
            acc_y = acc_y_list if (acc_y_list and len(acc_y_list) == n) else [0.0]*n
            _, a_wy, vdv_y, _, _ = filter_axis(acc_y, wd_freqs, wd_gains)

            # 4. Point Vibration Total Value (a_v) - ISO 2631-1 Section 6.5:
            # kx = 1.0, ky = 1.0, kz = 1.0 for seated passengers
            a_v = float(np.sqrt(1.0 * (a_wx**2) + 1.0 * (a_wy**2) + 1.0 * (a_wz**2)))
            
            # 5. Tri-Axial Vibration Dose Value (VDV_total)
            vdv_total = float((vdv_x**4 + vdv_y**4 + vdv_z**4)**0.25)
            
            # 6. Crest Factor on Vertical (ISO 2631-1: CF > 9 requires VDV)
            peak_wz = float(np.max(np.abs(sig_wz)))
            crest_factor_z = float(peak_wz / max(0.001, a_wz))
            shock_event = "SHOCK_DETECTED (CF>9)" if crest_factor_z > 9.0 else "NOMINAL"
            
            # 7. Dominant Frequency on Vertical
            powers_z = np.abs(yf_z)**2
            dom_freq_z = float(xf_z[np.argmax(powers_z[1:]) + 1]) if len(powers_z) > 1 else 0.0

            # 8. Official ISO 2631-1:1997 Annex C Comfort Classification (based on a_v)
            if a_v < 0.315:
                comfort_tier = "Comfortable"
            elif a_v < 0.63:
                comfort_tier = "A little uncomfortable"
            elif a_v < 1.0:
                comfort_tier = "Fairly uncomfortable"
            elif a_v < 1.6:
                comfort_tier = "Uncomfortable"
            elif a_v < 2.5:
                comfort_tier = "Very uncomfortable"
            else:
                comfort_tier = "Extremely uncomfortable"

            return (a_wz, a_wx, a_wy, a_v, vdv_total, vdv_z, crest_factor_z, dom_freq_z, shock_event, comfort_tier)
        except Exception:
            return default_res

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
        {"name": "acc_x", "type": "double"},
        {"name": "acc_y", "type": "double"},
        {"name": "acc_z", "type": "double"},
        {"name": "suspension_mm", "type": "double"},
        {"name": "pitch_deg", "type": "double"},
        {"name": "roll_deg", "type": "double"},
        {"name": "tire_load_ratio", "type": "double"}
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
        .filter(col("acc_z").between(-15.0, 15.0)) \
        .filter(col("speed_kmh") >= 0) \
        .withColumn("acc_x", expr("COALESCE(acc_x, 0.0)")) \
        .withColumn("acc_y", expr("COALESCE(acc_y, 0.0)")) \
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
            spark_min(expr("COALESCE(tire_load_ratio, 1.0)")).alias("min_tire_load"),
            avg(expr("COALESCE(tire_load_ratio, 1.0)")).alias("avg_tire_load"),
            collect_list("acc_x").alias("acc_x_series"),
            collect_list("acc_y").alias("acc_y_series"),
            collect_list("acc_z").alias("acc_z_series")
        )

    # Apply Full Multi-Axial ISO 2631-1:1997 Comfort Evaluation (Wk + Wd + Total Vibration av)
    gold_with_iso = gold_windowed \
        .withColumn("iso_metrics", calculate_triaxial_iso_metrics(col("acc_x_series"), col("acc_y_series"), col("acc_z_series"))) \
        .select("*", "iso_metrics.*") \
        .drop("iso_metrics", "acc_x_series", "acc_y_series", "acc_z_series")

    # Dynamic Multi-Axial Comfort Score (0-100%):
    # Evaluates total vibration av (vector sum of weighted vertical, lateral, longitudinal acceleration) and multi-axial VDV
    gold_metrics = gold_with_iso \
        .withColumn("comfort_score", 
            expr("GREATEST(0.0, LEAST(100.0, 100.0 - (total_acc_v * 35.0 + vdv_total * 10.0)))")) \
        .withColumn("road_holding_tier",
            expr("CASE WHEN min_tire_load < 0.15 THEN 'CRITICAL (Wheel Lift Hazard)' " +
                 "WHEN min_tire_load < 0.40 THEN 'MARGINAL (Tire Unloading)' " +
                 "ELSE 'OPTIMAL (Full Grip)' END"))

    gold_final = gold_metrics.select(
        col("window.start").alias("start_time"),
        col("window.end").alias("end_time"),
        "vehicle_id", "test_id", "rms_acc_z", "peak_acc_z", 
        "avg_speed", "weighted_acc_z", "weighted_acc_x", "weighted_acc_y",
        "total_acc_v", "vdv_total", "vdv_acc_z", "crest_factor_z",
        "dominant_freq_z", "shock_event", "iso_comfort_tier", "comfort_score",
        "min_tire_load", "avg_tire_load", "road_holding_tier"
    ).withColumn("date", to_date(col("start_time"))) \
     .withColumn("year", year(col("start_time"))) \
     .withColumn("month", month(col("start_time"))) \
     .withColumn("day", day(col("start_time")))

    # Delta Lake Small File Optimization:
    # Use coarse date partitioning to eliminate the small file problem created by high-frequency microbatches
    gold_partition_cols = ["date"]
    try:
        from deltalake import DeltaTable as PyDeltaTable
        if os.path.exists(GOLD_DIR):
            existing_parts = PyDeltaTable(GOLD_DIR).metadata().partition_columns
            if existing_parts:
                gold_partition_cols = existing_parts
    except Exception:
        pass

    # Write Gold to Delta with Coarse Partitioning & Compaction Flags
    gold_query = gold_final.writeStream \
        .format("delta") \
        .partitionBy(*gold_partition_cols) \
        .option("checkpointLocation", os.path.join(CHECKPOINT_PATH, "gold_delta")) \
        .option("mergeSchema", "true") \
        .option("delta.autoOptimize.optimizeWrite", "true") \
        .option("delta.autoOptimize.autoCompact", "true") \
        .outputMode("append") \
        .trigger(processingTime="5 seconds") \
        .start(GOLD_DIR)

    print("\n🚀 Industrial Pipeline Active (Delta + Avro + Pandas UDF)")
    print(f"📍 Gold Layer: {GOLD_DIR}")
    
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
