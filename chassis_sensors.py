import time
import json
import random
import math
import argparse
import sys
from datetime import datetime

# --- CONFIG & CONSTANTS ---
GRAVITY = 9.81  # m/s^2

# Physics Constraints
MIN_SPEED = 20.0   # km/h
MAX_SPEED = 140.0  # km/h
MAX_ACC_Z = 8.0    # m/s^2 (Higher limit for bumps)
MAX_SUSP = 80.0    # mm
MIN_PITCH = -8.0   # deg
MAX_PITCH = 8.0    # deg
MIN_ROLL = -6.0    # deg
MAX_ROLL = 6.0     # deg

# Scenarios with enhanced variability
SCENARIOS = {
    "Run_Smooth_Highway": {"roughness": 0.1, "spike_prob": 0.001, "spike_mag": 0.3},
    "Run_Urban_Road":     {"roughness": 0.8, "spike_prob": 0.03,  "spike_mag": 2.0},
    "Run_Pothole_Alley":  {"roughness": 2.0, "spike_prob": 0.15,  "spike_mag": 5.0},
}

# Avro Schema for real-time streaming
AVRO_SCHEMA = """
{
  "type": "record",
  "name": "ChassisSensor",
  "namespace": "com.automotive.telemetry",
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

class VehiclePhysics:
    """Simulates realistic vehicle chassis dynamics including bumps and road noise."""
    def __init__(self, vehicle_id, scenario_name=None):
        self.vehicle_id = vehicle_id
        if scenario_name and scenario_name in SCENARIOS:
            self.scenario_name = scenario_name
        else:
            self.scenario_name = random.choice(list(SCENARIOS.keys()))
        
        self.test_id = f"{self.scenario_name}_{datetime.now().strftime('%H%M%S')}"
        self.scenario_params = SCENARIOS[self.scenario_name]
        
        # State Variables
        self.speed_kmh = random.uniform(30.0, 60.0)
        self.target_speed = random.uniform(MIN_SPEED, MAX_SPEED)
        self.acc_z = 0.0
        self.suspension_mm = 30.0 
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        
        self.speed_change_timer = 0
        self.roll_phase = random.uniform(0, 2 * math.pi)
        
    def set_scenario(self, scenario_name):
        if scenario_name in SCENARIOS:
            self.scenario_name = scenario_name
            self.scenario_params = SCENARIOS[scenario_name]

    def update(self, dt):
        """Calculates the next physics state based on dt (delta time)."""
        
        # 1. Speed Dynamics (Slowly drifting target)
        self.speed_change_timer += dt
        if self.speed_change_timer > 8.0:
            self.target_speed = random.uniform(MIN_SPEED, MAX_SPEED)
            self.speed_change_timer = 0
        
        speed_delta = (self.target_speed - self.speed_kmh) * dt * 0.2
        self.speed_kmh += speed_delta
        self.speed_kmh = max(MIN_SPEED, min(self.speed_kmh, MAX_SPEED))

        # 2. Road Excitation (Vibration)
        speed_factor = (self.speed_kmh / 100.0) # Higher speed = more vibration
        roughness = self.scenario_params["roughness"]
        
        # Base Road Noise (Gaussian)
        base_noise = random.gauss(0, 0.4) * roughness * speed_factor
        
        # ⚠️ BUMP GENERATOR (Spikes)
        bump_noise = 0.0
        if random.random() < (self.scenario_params["spike_prob"]):
            # Generate a "Shock" event
            impact_mag = self.scenario_params["spike_mag"] * random.uniform(0.7, 1.5)
            # Randomly either a pothole (negative) or a bump (positive)
            bump_noise = random.choice([-1, 1]) * impact_mag
        
        # Combine noise and bumps
        self.acc_z = base_noise + bump_noise
        # Limit to physical maximums
        self.acc_z = max(-MAX_ACC_Z, min(self.acc_z, MAX_ACC_Z))
        
        # 3. Suspension Response
        # Suspension dampens the acceleration
        target_susp = 35.0 + (self.acc_z * 8.0) + random.gauss(0, 0.5)
        self.suspension_mm += (target_susp - self.suspension_mm) * dt * 12.0
        self.suspension_mm = max(0.0, min(self.suspension_mm, MAX_SUSP))

        # 4. Chassis Orientation (Pitch/Roll)
        # Pitch reacts to speed changes
        target_pitch = speed_delta * 10.0 + (self.acc_z * 0.5)
        self.pitch_deg += (target_pitch - self.pitch_deg) * dt * 4.0
        
        # Roll simulates highway curves
        self.roll_phase += dt * 0.1
        self.roll_deg = math.sin(self.roll_phase) * 2.0 * speed_factor + random.gauss(0, 0.1)

    def generate_message(self):
        """Formats the current state into an Avro-ready dictionary."""
        return {
            "vehicle_id": self.vehicle_id,
            "test_id": self.test_id,
            "timestamp": time.time(),
            "speed_kmh": float(round(self.speed_kmh, 2)),
            "acc_z": float(round(self.acc_z, 4)),
            "suspension_mm": float(round(self.suspension_mm, 2)),
            "pitch_deg": float(round(self.pitch_deg, 3)),
            "roll_deg": float(round(self.roll_deg, 3))
        }

def run_simulation(args):
    producer = None
    avro_serializer = None
    
    if args.mode == "kafka":
        try:
            from confluent_kafka import Producer
            from confluent_kafka.schema_registry import SchemaRegistryClient
            from confluent_kafka.schema_registry.avro import AvroSerializer
            from confluent_kafka.serialization import SerializationContext, MessageField
            
            # Setup Registry
            sr_client = SchemaRegistryClient({'url': args.schema_registry})
            avro_serializer = AvroSerializer(sr_client, AVRO_SCHEMA)
            
            # Setup Producer
            producer = Producer({'bootstrap.servers': args.bootstrap_servers})
            print(f"✅ Real-time pipeline connected: {args.bootstrap_servers}")
        except Exception as e:
            print(f"❌ Error setting up Kafka: {e}")
            sys.exit(1)

    # Initialize vehicles with different behaviors
    vehicles = [VehiclePhysics(f"Vehicle_{i+1:02d}", args.scenario) for i in range(args.vehicle_count)]
    
    print(f"🚀 Launching simulation: {args.vehicle_count} vehicles at {args.frequency}Hz")
    print("💡 Dynamic bumps are enabled for visual verification on the dashboard.")
    
    period = 1.0 / args.frequency
    try:
        while True:
            start_loop = time.time()
            
            for v in vehicles:
                v.update(period)
                msg = v.generate_message()
                
                if args.mode == "console":
                    print(json.dumps(msg))
                elif args.mode == "kafka":
                    producer.produce(
                        topic=args.topic,
                        key=v.vehicle_id,
                        value=avro_serializer(msg, SerializationContext(args.topic, MessageField.VALUE))
                    )
            
            if args.mode == "kafka":
                producer.poll(0) # Non-blocking poll
                
            elapsed = time.time() - start_loop
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        if producer:
            producer.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["console", "kafka"], default="kafka")
    parser.add_argument("--frequency", type=int, default=20)
    parser.add_argument("--vehicle_count", type=int, default=3)
    parser.add_argument("--topic", default="chassis_sensors")
    parser.add_argument("--bootstrap_servers", default="localhost:9092")
    parser.add_argument("--schema_registry", default="http://127.0.0.1:8081")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Force a specific road scenario")
    
    args = parser.parse_args()
    run_simulation(args)
