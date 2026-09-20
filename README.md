# Vehicle Comfort Pipeline

### Motivation & Context
This project is motivated by the Porsche Engineering article *"Objectively Comfortable"* (Porsche Newsroom), which describes using automated evaluation systems to replace subjective human test-driver assessments with reproducible measurement data—specifically targeting chassis vibrations below 35 Hz. 

In vehicle validation, individual driver preferences lead to inconsistent comfort ratings. This project addresses that discrepancy by streaming raw physical telemetry (vertical acceleration, suspension travel, pitch, and roll) and translating those physical measurements into objective comfort metrics. By applying ISO 2631-1 frequency-weighting curves ($W_k$) and spectral analysis, the pipeline quantifies vibration severity based on human biodynamic sensitivity rather than qualitative impressions.

---

### Architecture & Processing Pipeline

* **Ingestion:** Telemetry producer generating 20 Hz vehicle dynamic states across standard test scenarios (highway, urban, pothole). Messages are serialized with Apache Avro and registered against Confluent Schema Registry.
* **Storage & Medallion Layers (Delta Lake):**
  * **Bronze:** Raw streaming Kafka records stored with ingestion timestamps.
  * **Silver:** Parsed, validated, and structured records partitioned by vehicle and date.
  * **Gold:** Windowed aggregations calculating statistical metrics (RMS, peak acceleration), dominant vibration frequencies, and ISO 2631-1 weighted acceleration values.
* **Signal Processing:** Implements an ISO 2631-1 vertical weighting filter ($W_k$) and Fast Fourier Transform (FFT) power spectrum analysis focusing on the 4–8 Hz range where the human spine is most sensitive to vertical vibration.

---

### Running the Pipeline

**1. Start Kafka and Schema Registry:**
```bash
docker compose up -d
```

**2. Start the telemetry producer:**
```bash
python3 chassis_sensors.py --mode kafka --vehicle_count 3 --frequency 20
```

**3. Run the Spark streaming engine:**
```bash
python3 spark_pipeline.py
```

**4. Launch the monitoring dashboard:**
```bash
streamlit run dashboard.py
```

**5. (Optional) Run pipeline throughput and latency benchmarks:**
```bash
python3 measure_reliability.py
```

---

### Technology Stack

* **Message Broker & Schema:** Apache Kafka, Confluent Schema Registry, Apache Avro
* **Stream Processing:** Apache Spark (Structured Streaming), PySpark
* **Storage Layer:** Delta Lake (ACID transactions, Parquet format)
* **Signal Processing:** NumPy (FFT, ISO 2631-1 filter curves)
* **UI & Metrics:** Streamlit, Plotly, Pandas
