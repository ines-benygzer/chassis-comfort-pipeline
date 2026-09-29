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
MAX_ACC_Z = 12.0   # m/s^2 (Accommodates severe dynamic shocks and potholes)
MAX_SUSP = 80.0    # mm
MIN_PITCH = -8.0   # deg
MAX_PITCH = 8.0    # deg
MIN_ROLL = -6.0    # deg
MAX_ROLL = 6.0     # deg

# --- ISO 8608 ROAD ROUGHNESS CLASSIFICATION ---
# Power Spectral Density (PSD): Gd(n) = Gd(n0) * (n / n0)^(-w)
# Reference spatial frequency n0 = 0.1 cycles/m (0.1 m^-1), waviness w = 2
# Values correspond to ISO 8608 geometric mean displacement PSD Gd(n0) in m^3
ISO_8608_CLASSES = {
    "A": {
        "class": "A",
        "name": "Class A (Very Good / Motorway)",
        "Gd_n0": 2e-6,
        "target_speed_min": 90.0, "target_speed_max": 130.0,
        "bump_prob": 0.001, "bump_height": 0.006,  # Tiny expansion seams
    },
    "B": {
        "class": "B",
        "name": "Class B (Good / Smooth Pavement)",
        "Gd_n0": 8e-6,
        "target_speed_min": 65.0, "target_speed_max": 105.0,
        "bump_prob": 0.008, "bump_height": 0.015,
    },
    "C": {
        "class": "C",
        "name": "Class C (Average / Urban & Secondary Road)",
        "Gd_n0": 32e-6,
        "target_speed_min": 35.0, "target_speed_max": 65.0,
        "bump_prob": 0.030, "bump_height": 0.030,  # Speed bumps / manholes
    },
    "D": {
        "class": "D",
        "name": "Class D (Poor / Cobblestone & Pothole Alley)",
        "Gd_n0": 128e-6,
        "target_speed_min": 20.0, "target_speed_max": 45.0,
        "bump_prob": 0.100, "bump_height": -0.050, # Severe potholes
    },
    "E": {
        "class": "E",
        "name": "Class E (Very Poor / Extreme Deformation)",
        "Gd_n0": 512e-6,
        "target_speed_min": 15.0, "target_speed_max": 30.0,
        "bump_prob": 0.180, "bump_height": -0.070, # Broken road craters
    },
}

# Standard test scenarios mapped directly to ISO 8608 road roughness classes
SCENARIOS = {
    "Run_Smooth_Highway": ISO_8608_CLASSES["A"],
    "Run_Urban_Road":     ISO_8608_CLASSES["C"],
    "Run_Pothole_Alley":  ISO_8608_CLASSES["D"],
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
    {"name": "roll_deg", "type": "double"},
    {"name": "tire_load_ratio", "type": "double", "default": 1.0}
  ]
}
"""

class VehiclePhysics:
    """
    Simulates realistic 2-DOF Quarter-Car chassis dynamics.
    
    Differential Equations:
      m_s * d2(z_s)/dt2 = -F_susp
      m_u * d2(z_u)/dt2 =  F_susp - F_tire
      
    Where:
      - m_s: Sprung mass (chassis body quarter) ~ 320 kg
      - m_u: Unsprung mass (wheel assembly) ~ 42 kg
      - F_susp: Spring + Asymmetric Damper + Non-linear bump-stops
      - F_tire: Tire vertical stiffness + contact damping
      - Natural frequencies: Body bounce ~ 1.4 Hz, Wheel hop ~ 11.4 Hz
    """
    def __init__(self, vehicle_id, scenario_name=None, road_class=None):
        self.vehicle_id = vehicle_id
        if road_class and road_class.upper() in ISO_8608_CLASSES:
            rc = road_class.upper()
            self.scenario_name = f"ISO_Class_{rc}"
            self.params = ISO_8608_CLASSES[rc]
        elif scenario_name and scenario_name in SCENARIOS:
            self.scenario_name = scenario_name
            self.params = SCENARIOS[scenario_name]
        else:
            self.scenario_name = random.choice(list(SCENARIOS.keys()))
            self.params = SCENARIOS[self.scenario_name]
        
        iso_tag = self.params.get("class", "C")
        self.test_id = f"{self.scenario_name}_ISO-{iso_tag}_{datetime.now().strftime('%H%M%S')}"
        
        # Quarter-Car 2-DOF Physical Parameters
        self.m_s = 320.0       # Sprung mass (kg)
        self.m_u = 42.0        # Unsprung mass (kg)
        self.k_s = 26000.0     # Suspension spring stiffness (N/m)
        self.c_s_comp = 1100.0 # Compression damping (N*s/m)
        self.c_s_reb  = 1800.0 # Rebound damping (N*s/m)
        self.k_t = 190000.0    # Tire vertical stiffness (N/m)
        self.c_t = 150.0       # Tire damping (N*s/m)
        
        # Progressive elastomeric bump-stops
        self.bump_stop_clearance = 0.035 # 35 mm stroke before progressive stop engagement
        self.k_bump_stop = 80000.0       # Progressive spring rate
        
        # State variables (relative to static equilibrium)
        self.z_s = 0.0         # Sprung mass vertical displacement (m)
        self.v_s = 0.0         # Sprung mass vertical velocity (m/s)
        self.acc_z = 0.0       # Sprung mass vertical acceleration (m/s^2)
        
        self.z_u = 0.0         # Unsprung mass vertical displacement (m)
        self.v_u = 0.0         # Unsprung mass vertical velocity (m/s)
        
        self.z_r = 0.0         # Road elevation profile (m)
        self.road_profile_x = 0.0 # Spatial distance along test route (m)
        
        # Speed dynamics
        self.speed_kmh = random.uniform(self.params["target_speed_min"], self.params["target_speed_max"])
        self.target_speed = self.speed_kmh
        self.speed_timer = 0.0
        self.long_acc = 0.0    # Longitudinal acceleration (m/s^2)
        
        # Suspension sensor reading
        self.suspension_mm = 35.0 # Nominal resting height
        
        # Tire Normal Force & Road-Holding
        self.f_static = (self.m_s + self.m_u) * GRAVITY
        self.tire_load_ratio = 1.0
        
        # Chassis posture (degrees)
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.roll_phase = random.uniform(0, 2 * math.pi)
        
        # Active transient obstacle (bump / pothole / seam)
        self.active_obstacle = None
        
    def set_scenario(self, scenario_name):
        if scenario_name in SCENARIOS:
            self.scenario_name = scenario_name
            self.params = SCENARIOS[scenario_name]
        elif scenario_name in ISO_8608_CLASSES:
            self.scenario_name = f"ISO_Class_{scenario_name}"
            self.params = ISO_8608_CLASSES[scenario_name]

    def update(self, dt):
        """Advances vehicle dynamics using high-frequency numerical sub-stepping."""
        # 1. Longitudinal speed dynamics
        self.speed_timer += dt
        if self.speed_timer > 6.0:
            self.target_speed = random.uniform(self.params["target_speed_min"], self.params["target_speed_max"])
            self.speed_timer = 0.0
            
        speed_error = (self.target_speed - self.speed_kmh)
        speed_delta = speed_error * dt * 0.25
        self.long_acc = (speed_delta / 3.6) / dt # m/s^2
        self.speed_kmh += speed_delta
        self.speed_kmh = max(MIN_SPEED, min(self.speed_kmh, MAX_SPEED))
        v_ms = self.speed_kmh / 3.6
        
        # 2. Transient obstacle trigger (Speed bumps, potholes, expansion seams)
        if self.active_obstacle is None:
            if random.random() < (self.params["bump_prob"] * (dt * 20.0)):
                h = self.params["bump_height"] * random.uniform(0.8, 1.4)
                length = random.uniform(0.6, 1.2) if h > 0 else random.uniform(0.4, 0.8)
                self.active_obstacle = {
                    "start_x": self.road_profile_x,
                    "length": length,
                    "height": h
                }
                
        # 3. High-frequency Sub-stepping for 2-DOF numerical integration (400 Hz solver)
        sub_steps = 20
        h_step = dt / sub_steps
        
        # ISO 8608 Filter Parameters: n0 = 0.1 cycles/m, f0 = 0.05 Hz cutoff
        n0 = 0.1
        f0 = 0.05
        gd_n0 = self.params["Gd_n0"]
        
        for _ in range(sub_steps):
            dx = v_ms * h_step
            self.road_profile_x += dx
            
            # Exact ISO 8608 1st-order temporal shaping filter:
            # dz_r/dt = -2*pi*f0 * z_r + 2*pi*n0 * sqrt(Gd(n0) * v) * w(t)
            white_noise = random.gauss(0.0, 1.0)
            dz_noise = 2.0 * math.pi * n0 * math.sqrt(gd_n0 * max(1.0, v_ms)) * math.sqrt(h_step) * white_noise
            self.z_r = self.z_r * (1.0 - 2.0 * math.pi * f0 * h_step) + dz_noise
            
            # Add active obstacle profile (Half-sine wave)
            obstacle_z = 0.0
            if self.active_obstacle:
                rel_x = self.road_profile_x - self.active_obstacle["start_x"]
                l_obs = self.active_obstacle["length"]
                if 0 <= rel_x <= l_obs:
                    obstacle_z = self.active_obstacle["height"] * math.sin(math.pi * rel_x / l_obs)
                else:
                    self.active_obstacle = None
                    
            z_road_total = self.z_r + obstacle_z
            
            # Quarter-Car forces
            delta_z = self.z_s - self.z_u
            v_rel = self.v_s - self.v_u
            
            # Asymmetric damping (rebound stiffer than compression)
            c_damper = self.c_s_reb if v_rel > 0 else self.c_s_comp
            f_damper = c_damper * v_rel
            f_spring = self.k_s * delta_z
            
            # Non-linear bump-stop force
            f_bump_stop = 0.0
            excess_jounce = abs(delta_z) - self.bump_stop_clearance
            if excess_jounce > 0:
                f_bump_stop = math.copysign(self.k_bump_stop * (excess_jounce ** 2), delta_z)
                
            f_suspension = f_spring + f_damper + f_bump_stop
            
            # Tire dynamic vertical force
            tire_deflection = self.z_u - z_road_total
            f_tire = max(0.0, self.k_t * (tire_deflection + 0.02) + self.c_t * self.v_u) - (self.k_t * 0.02)
            
            # Accelerations (Newton's 2nd Law)
            a_sprung = -f_suspension / self.m_s
            a_unsprung = (f_suspension - f_tire) / self.m_u
            
            # Symplectic Euler integration
            self.v_s += a_sprung * h_step
            self.z_s += self.v_s * h_step
            self.v_u += a_unsprung * h_step
            self.z_u += self.v_u * h_step
            
        self.acc_z = max(-MAX_ACC_Z, min(MAX_ACC_Z, a_sprung))
        
        # 4. Suspension Stroke (Nominal 35mm at equilibrium)
        susp_disp = (self.z_u - self.z_s) * 1000.0 # mm
        self.suspension_mm = max(5.0, min(MAX_SUSP, 35.0 + susp_disp))
        
        # Dynamic Tire Normal Force & Road-Holding Ratio:
        # F_normal = max(0, F_static - f_tire)
        # eta = F_normal / F_static. eta < 0.15 indicates imminent wheel lift-off hazard!
        f_normal = max(0.0, self.f_static - f_tire)
        self.tire_load_ratio = float(max(0.0, min(3.5, f_normal / self.f_static)))
        
        # 5. Chassis Posture: Pitch & Roll Dynamics
        target_pitch = (self.long_acc * 1.8) + (self.acc_z * 0.25)
        self.pitch_deg += (target_pitch - self.pitch_deg) * dt * 5.0
        self.pitch_deg = max(MIN_PITCH, min(MAX_PITCH, self.pitch_deg))
        
        self.roll_phase += dt * (v_ms / 30.0)
        target_roll = math.sin(self.roll_phase) * 1.5 * (self.speed_kmh / 100.0) + (self.z_s * 15.0)
        self.roll_deg += (target_roll - self.roll_deg) * dt * 3.0
        self.roll_deg = max(MIN_ROLL, min(MAX_ROLL, self.roll_deg))

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
            "roll_deg": float(round(self.roll_deg, 3)),
            "tire_load_ratio": float(round(self.tire_load_ratio, 4))
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
    vehicles = [VehiclePhysics(f"Vehicle_{i+1:02d}", scenario_name=args.scenario, road_class=args.road_class) for i in range(args.vehicle_count)]
    
    print(f"🚀 Launching simulation: {args.vehicle_count} vehicles at {args.frequency}Hz")
    if args.road_class:
        print(f"🛣️  Enforcing ISO 8608 Road Class: {args.road_class} ({ISO_8608_CLASSES[args.road_class]['name']})")
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
    parser = argparse.ArgumentParser(description="Vehicle Telemetry Simulator (2-DOF Quarter-Car + ISO 8608)")
    parser.add_argument("--mode", choices=["console", "kafka"], default="kafka")
    parser.add_argument("--frequency", type=int, default=20)
    parser.add_argument("--vehicle_count", type=int, default=3)
    parser.add_argument("--topic", default="chassis_sensors")
    parser.add_argument("--bootstrap_servers", default="localhost:9092")
    parser.add_argument("--schema_registry", default="http://127.0.0.1:8081")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Force a specific road scenario")
    parser.add_argument("--road_class", choices=["A", "B", "C", "D", "E"], help="Force a specific ISO 8608 road roughness class (A=Very Good to E=Very Poor)")
    
    args = parser.parse_args()
    run_simulation(args)
