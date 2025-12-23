import random
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import time

class WarehouseAnalytics:
    def __init__(self):
        self.history = []
        self.corrections = 0

    def log(self, temp, humidity, heat, cool, humidify, dehumidify):
        self.history.append({
            "temperature": temp,
            "humidity": humidity,
            "heat": heat,
            "cool": cool,
            "humidify": humidify,
            "dehumidify": dehumidify
        })
        self.corrections += 1

    def report(self):
        print("\n===== ОТЧЕТ СИСТЕМЫ УПРАВЛЕНИЯ СКЛАДОМ =====")
        print(f"Всего управляющих воздействий: {self.corrections}")

def create_warehouse_ontology():
    return {
        "temperature_range": (0, 40),
        "humidity_range": (40, 100),
        "optimal_temperature": 20,
        "optimal_humidity": 60
    }

def setup_fuzzy_systems(warehouse):
    temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
    heat = ctrl.Consequent(np.arange(0, 101, 1), 'heat')
    cool = ctrl.Consequent(np.arange(0, 101, 1), 'cool')

    temperature['low']  = fuzz.trapmf(temperature.universe,[0, 0, 16, 20])
    temperature['normal'] = fuzz.trimf(temperature.universe, [16, warehouse['optimal_temperature'], 24])
    temperature['high'] = fuzz.trapmf(temperature.universe, [22, 26, 40, 40])

    heat['low'] = fuzz.trapmf(heat.universe, [0, 0, 20, 40])
    heat['high'] = fuzz.trapmf(heat.universe, [30, 60, 100, 100])

    cool['low'] = fuzz.trapmf(cool.universe, [0, 0, 20, 40])
    cool['high'] = fuzz.trapmf(cool.universe, [30, 60, 100, 100])

    temp_rules = [
        ctrl.Rule(temperature['low'], heat['high']),
        ctrl.Rule(temperature['high'], cool['high']),
        ctrl.Rule(temperature['normal'], heat['low']),
        ctrl.Rule(temperature['normal'], cool['low']),
    ]
    temp_system = ctrl.ControlSystem(temp_rules)

    # -----------------------
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    humidify = ctrl.Consequent(np.arange(0, 101, 1), 'humidify')
    dehumidify = ctrl.Consequent(np.arange(0, 101, 1), 'dehumidify')

    humidity['low'] = fuzz.trapmf(humidity.universe, [0, 0, 45, 55])
    humidity['normal'] = fuzz.trimf(humidity.universe, [50, warehouse['optimal_humidity'], 70])
    humidity['high'] = fuzz.trapmf(humidity.universe, [65, 75, 100, 100])

    humidify['low'] = fuzz.trapmf(humidify.universe, [0, 0, 20, 40])
    humidify['high'] = fuzz.trapmf(humidify.universe, [30, 60, 100, 100])

    dehumidify['low'] = fuzz.trapmf(dehumidify.universe, [0, 0, 20, 40])
    dehumidify['high'] = fuzz.trapmf(dehumidify.universe, [30, 60, 100, 100])

    hum_rules = [
        ctrl.Rule(humidity['low'], humidify['high']),
        ctrl.Rule(humidity['high'], dehumidify['high']),
        ctrl.Rule(humidity['normal'], humidify['low']),
        ctrl.Rule(humidity['normal'], dehumidify['low'])
    ]
    hum_system = ctrl.ControlSystem(hum_rules)

    return temp_system, hum_system


def safe_get(output, key):
    return output[key] if key in output else 0.0

def apply_control(temp, humidity, analytics, TEMP_CTRL, HUM_CTRL):
    # Система температуры
    temp_sim = ctrl.ControlSystemSimulation(TEMP_CTRL)
    temp_sim.input['temperature'] = temp
    temp_sim.compute()

    heat = safe_get(temp_sim.output, 'heat')
    cool = safe_get(temp_sim.output, 'cool')

    # Система влажности
    hum_sim = ctrl.ControlSystemSimulation(HUM_CTRL)
    hum_sim.input['humidity'] = humidity
    hum_sim.compute()

    humidify = safe_get(hum_sim.output, 'humidify')
    dehumidify = safe_get(hum_sim.output, 'dehumidify')

    analytics.log(temp, humidity, heat, cool, humidify, dehumidify)

    return heat, cool, humidify, dehumidify


def run_simulation(steps=100):
    warehouse = create_warehouse_ontology()
    analytics = WarehouseAnalytics()

    TEMP_CTRL, HUM_CTRL = setup_fuzzy_systems(warehouse)

    temperature = random.randint(warehouse['temperature_range'][0], warehouse['temperature_range'][1])
    humidity = random.randint(warehouse['humidity_range'][0], warehouse['humidity_range'][1])

    print("Запуск симуляции склада\n")

    for step in range(1, steps + 1):
        heat, cool, humidify, dehumidify = apply_control(
            temperature, humidity, analytics, TEMP_CTRL, HUM_CTRL
        )

        temperature += (heat - cool) * 0.05

        humidity += (humidify - dehumidify) * 0.02

        print(
            f"Шаг {step:02d}: "
            f"T={temperature:.1f}°C, H={humidity:.1f}% | "
            f"Heat={heat:.0f}, Cool={cool:.0f}, "
            f"Hum={humidify:.0f}, Dehum={dehumidify:.0f}"
        )

        time.sleep(0.1)

    analytics.report()
    visualize(analytics.history)

def visualize(history):
    temps = [h['temperature'] for h in history]
    hums = [h['humidity'] for h in history]

    plt.figure(figsize=(10, 4))
    plt.plot(temps, label="Температура (°C)")
    plt.plot(hums, label="Влажность (%)")
    plt.axhline(20, linestyle='--', color='gray')
    plt.axhline(60, linestyle='--', color='gray')
    plt.legend()
    plt.grid()
    plt.show()

run_simulation()
