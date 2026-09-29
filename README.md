# Vehicle Comfort Pipeline

### Motivation & Context
This project is motivated by the Porsche Engineering article *"Objectively Comfortable"* (Porsche Newsroom), which describes using automated evaluation systems to replace subjective human test-driver assessments with reproducible measurement data—specifically targeting chassis vibrations below 35 Hz. 

In vehicle validation, individual driver preferences lead to inconsistent comfort ratings. This project addresses that discrepancy by streaming raw physical telemetry (vertical acceleration, suspension travel, pitch, and roll) and translating those physical measurements into objective comfort metrics. By applying ISO 2631-1 frequency-weighting curves ($W_k$) and spectral analysis, the pipeline quantifies vibration severity based on human biodynamic sensitivity rather than qualitative impressions.

---

### Architecture & Processing Pipeline

* **Ingestion (2-DOF Quarter-Car & ISO 8608 Road Profiles):** 
  * **2-DOF Dynamics:** Models sprung mass ($m_s = 320\text{ kg}$), unsprung mass ($m_u = 42\text{ kg}$), asymmetric damping (rebound $1800\text{ N}\cdot\text{s/m}$, compression $1100\text{ N}\cdot\text{s/m}$), tire compliance ($190\text{ kN/m}$), and progressive bump stops.
  * **ISO 8608 Standard Roughness:** Implements formal road roughness Classes A through E via a 1st-order temporal differential shaping filter ($\dot{z}_r = -2\pi f_0 z_r + 2\pi n_0 \sqrt{G_d(n_0) v} \cdot w(t)$). Supported via CLI flag `--road_class [A|B|C|D|E]`.
  * **Telemetry Serialization:** Solved via a 400 Hz numerical integrator emitting 20 Hz Avro telemetry registered with Confluent Schema Registry.
* **Storage & Medallion Layers (Delta Lake):**
  * **Bronze:** Raw streaming Kafka records stored with ingestion timestamps.
  * **Silver:** Parsed, schema-validated records with dynamic filter $[-15, 15]\text{ m/s}^2$ capturing pothole impacts.
  * **Gold:** Windowed aggregations calculating statistical metrics, dominant frequencies, ISO 2631-1 weighted acceleration ($a_w$), and Vibration Dose Values (VDV). Optimized with coarse date partitioning and auto-compaction to eliminate the Delta Lake Small File Problem.
* **Signal Processing (ISO 2631-1 Standards):** 
  * **Continuous Vibration (FFT $W_k$ Filter):** Decomposes vertical acceleration into the frequency domain, applying the $W_k$ sensitivity curve centered on the $4–8\text{ Hz}$ spinal resonance band.
  * **Transient Shock (Vibration Dose Value - VDV):** Reconstructs the weighted time history $a_w(t)$ via inverse FFT and calculates 4th-power accumulated shock dosage: $\text{VDV} = (\Delta t \sum a_w^4)^{1/4}$.
  * **Crest Factor ($\text{Peak}/\text{RMS}$):** Automatically flags when Crest Factor $> 9.0$, where standard RMS fails and VDV governs.
  * **Qualitative Scales:** Classifies comfort into official ISO 2631-1 tiers (*Comfortable*, *A little uncomfortable*, *Fairly uncomfortable*, *Uncomfortable*, *Very uncomfortable*, *Extremely uncomfortable*).

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
