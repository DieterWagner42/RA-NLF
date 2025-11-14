# Implizite Protection Functions (Korrekt)

## Konzept

**Safety/Hygiene Requirements → Protection Functions**

### ❌ FALSCH (Lösungen):
- `depressurize()` - WIE (Ventil öffnen)
- `cool()` - WIE (Kühlung aktivieren)
- `monitor()` - WIE (Sensor auslesen)
- `track()` - WIE (Daten verfolgen)

### ✅ RICHTIG (Protection Functions):
- `OverheatProtection` - WAS (Schutz vor Überhitzung gewährleisten)
- `OverpressureProtection` - WAS (Schutz vor Überdruck gewährleisten)
- `HygieneProtection` - WAS (Hygienische Bedingungen gewährleisten)

---

## 1. WaterLiquidManager

### Safety Functions

**OverheatProtection**
```
Constraint: Max temperature 100°C (safety_requirements.thermal_safety)
Trigger: Heating operations (UC1 B2a, UC2 B2a "activates water heater")
Function: protectOverheat() oder OverheatProtection
Implizit: Ja (nicht im UC, aber aus Safety Constraint)
```

**OverpressureProtection**
```
Constraint: Max pressure 15 bar (safety_requirements.pressure_safety)
Trigger: Pressurization operations (UC2 B3 "starts water compressor")
Function: protectOverpressure() oder OverpressureProtection
Implizit: Ja (nicht im UC, aber aus Safety Constraint)
```

**DryRunProtection**
```
Constraint: Heater darf nicht ohne Wasser laufen
Trigger: Heating operations + Low water level (UC1 A1 "too little water")
Function: protectDryRun() oder DryRunProtection
Implizit: Ja (nicht im UC, aber aus Safety Constraint)
```

### Hygiene Functions

**WaterQualityProtection**
```
Constraint: Wasser älter als 24h muss ausgetauscht werden (hygiene_requirements)
Trigger: Water precondition "Water is available"
Function: protectWaterQuality() oder WaterQualityProtection
Implizit: Ja (nicht im UC, aber aus Hygiene Constraint)
```

---

## 2. MilkLiquidManager

### Safety Functions

**TemperatureProtection** (Hygiene-kritisch)
```
Constraint: Milk storage < 4°C (hygiene_requirements.milk_hygiene)
Trigger: Precondition "Milk is available"
Function: protectTemperature() oder TemperatureProtection
Implizit: Ja (nicht im UC, aber aus Hygiene Constraint)
```

**FreshnessProtection**
```
Constraint: Milk expiration monitoring (hygiene_requirements.milk_hygiene)
Trigger: Precondition "Milk is available"
Function: protectFreshness() oder FreshnessProtection
Implizit: Ja (nicht im UC, aber aus Hygiene Constraint)
```

### Hygiene Functions

**LineHygieneProtection**
```
Constraint: Automatic milk line purging after use (hygiene_requirements.milk_hygiene)
Trigger: Nach Milch-Verwendung (UC1 B3b "adds milk")
Function: protectLineHygiene() oder LineHygieneProtection
Implizit: Ja (nicht im UC, aber aus Hygiene Constraint)
```

**SystemHygieneProtection**
```
Constraint: Daily cleaning cycle (hygiene_requirements.cleaning_cycles)
Trigger: Täglich (nicht im UC)
Function: protectSystemHygiene() oder SystemHygieneProtection
Implizit: Ja (nicht im UC, aber aus Hygiene Constraint)
```

---

## 3. SugarSolidManager

### Hygiene Functions

**MoistureProtection**
```
Constraint: Dry environment, sealed containers (operational_materials.sugar.storage_conditions)
Trigger: Precondition "Sugar is available"
Function: protectFromMoisture() oder MoistureProtection
Implizit: Ja (nicht im UC, aber aus Hygiene Constraint)
```

**DosingPrecisionProtection** (Quality)
```
Constraint: Präzise Dosierung erforderlich (quality_parameters)
Trigger: Sugar dispensing (E1.1 "adds defined amount of sugar")
Function: protectDosingPrecision() oder DosingPrecisionProtection
Implizit: Teilweise (Menge ist im UC, aber Präzision nicht)
```

---

## 4. CoffeeSolidManager

### Quality Functions

**FreshnessProtection**
```
Constraint: Dry, cool, dark environment (operational_materials.coffee_beans.storage_requirements)
Trigger: Precondition "Coffee beans are available"
Function: protectFreshness() oder FreshnessProtection
Implizit: Ja (nicht im UC, aber aus Quality Constraint)
```

---

## 5. FilterManager

### Hygiene Functions

**SingleUseProtection**
```
Constraint: Filter ist Einmalgebrauch (hygiene_requirements.contamination_prevention)
Trigger: Nach Verwendung (UC1 B6 "End UC")
Function: protectSingleUse() oder SingleUseProtection
Implizit: Ja (nicht im UC, aber aus Hygiene Constraint)
```

---

## 6. SystemControlManager

### Safety Functions

**OverheatProtection** (System-wide)
```
Constraint: System max temperature 90°C (safety_requirements.thermal_safety)
Trigger: UC1 A3 "Overheat detected"
Function: protectSystemOverheat() oder SystemOverheatProtection
Implizit: Teilweise (Overheat detection im UC, aber Protection nicht)
```

**EmergencyProtection**
```
Constraint: Emergency stop functionality required (safety_requirements.electrical_safety)
Trigger: UC1 A3.1 "stops all actions", A3.2 "switch of itself"
Function: protectEmergency() oder EmergencyProtection
Implizit: Nein (ist explizit im UC als A3)
```

---

## Finale Liste: Implizite Protection Functions

### 🔴 Safety Functions (5)

| Controller | Function | Constraint | Trigger |
|------------|----------|------------|---------|
| WaterLiquidManager | **OverheatProtection** | Max 100°C | Heating operations |
| WaterLiquidManager | **OverpressureProtection** | Max 15 bar | Pressurization operations |
| WaterLiquidManager | **DryRunProtection** | No water + heating | Low water level |
| MilkLiquidManager | **TemperatureProtection** | < 4°C | Milk storage |
| MilkLiquidManager | **FreshnessProtection** | Expiration | Milk storage |

### 🟡 Hygiene Functions (5)

| Controller | Function | Constraint | Trigger |
|------------|----------|------------|---------|
| WaterLiquidManager | **WaterQualityProtection** | < 24h age | Water storage |
| MilkLiquidManager | **LineHygieneProtection** | Purge after use | After milk dispensing |
| MilkLiquidManager | **SystemHygieneProtection** | Daily cleaning | Daily cycle |
| SugarSolidManager | **MoistureProtection** | Dry storage | Sugar storage |
| FilterManager | **SingleUseProtection** | One-time use | After brewing |

### 🟢 Quality Functions (2)

| Controller | Function | Constraint | Trigger |
|------------|----------|------------|---------|
| SugarSolidManager | **DosingPrecisionProtection** | ±0.5g | Sugar dispensing |
| CoffeeSolidManager | **FreshnessProtection** | Cool, dark, dry | Bean storage |

---

## JSON Struktur für beverage_preparation.json

```json
{
  "implicit_protection_functions": {
    "description": "Protection functions implicitly required by safety/hygiene constraints but not explicitly mentioned in Use Cases",

    "water": {
      "safety_functions": [
        {
          "name": "OverheatProtection",
          "constraint": "Max temperature 100°C",
          "constraint_source": "safety_requirements.thermal_safety.max_temperature",
          "trigger": "heating operations detected",
          "trigger_patterns": ["heat", "activate heater", "warm", "boil"],
          "controller": "WaterLiquidManager",
          "implicit": true,
          "criticality": "high"
        },
        {
          "name": "OverpressureProtection",
          "constraint": "Max pressure 15 bar",
          "constraint_source": "safety_requirements.pressure_safety.max_pressure",
          "trigger": "pressurization operations detected",
          "trigger_patterns": ["pressurize", "compress", "compressor", "pressure"],
          "controller": "WaterLiquidManager",
          "implicit": true,
          "criticality": "high"
        },
        {
          "name": "DryRunProtection",
          "constraint": "No heating without sufficient water",
          "constraint_source": "safety_requirements.thermal_safety.overheat_protection",
          "trigger": "heating operations AND low water level",
          "trigger_patterns": ["too little water", "water level low"],
          "controller": "WaterLiquidManager",
          "implicit": true,
          "criticality": "high"
        }
      ],
      "hygiene_functions": [
        {
          "name": "WaterQualityProtection",
          "constraint": "Water age < 24 hours",
          "constraint_source": "hygiene_requirements.contamination_prevention.bacterial_growth",
          "trigger": "water storage detected",
          "trigger_patterns": ["water is available", "precondition"],
          "controller": "WaterLiquidManager",
          "implicit": true,
          "criticality": "medium"
        }
      ]
    },

    "milk": {
      "safety_functions": [
        {
          "name": "TemperatureProtection",
          "constraint": "Storage temperature < 4°C",
          "constraint_source": "hygiene_requirements.milk_hygiene.temperature_control",
          "trigger": "milk storage detected",
          "trigger_patterns": ["milk is available", "precondition"],
          "controller": "MilkLiquidManager",
          "implicit": true,
          "criticality": "high"
        },
        {
          "name": "FreshnessProtection",
          "constraint": "Milk expiration monitoring",
          "constraint_source": "hygiene_requirements.milk_hygiene.freshness_monitoring",
          "trigger": "milk storage detected",
          "trigger_patterns": ["milk is available", "precondition"],
          "controller": "MilkLiquidManager",
          "implicit": true,
          "criticality": "high"
        }
      ],
      "hygiene_functions": [
        {
          "name": "LineHygieneProtection",
          "constraint": "Automatic purging after milk use",
          "constraint_source": "hygiene_requirements.milk_hygiene.automatic_purging",
          "trigger": "milk dispensing completed",
          "trigger_patterns": ["adds milk", "pour milk", "steam milk"],
          "controller": "MilkLiquidManager",
          "implicit": true,
          "criticality": "high"
        },
        {
          "name": "SystemHygieneProtection",
          "constraint": "Daily cleaning cycle",
          "constraint_source": "hygiene_requirements.cleaning_cycles.daily_cleaning",
          "trigger": "daily schedule",
          "trigger_patterns": ["daily", "cleaning cycle"],
          "controller": "MilkLiquidManager",
          "implicit": true,
          "criticality": "medium"
        }
      ]
    },

    "sugar": {
      "hygiene_functions": [
        {
          "name": "MoistureProtection",
          "constraint": "Dry storage environment",
          "constraint_source": "operational_materials_addressing.material_types.sugar.storage_conditions",
          "trigger": "sugar storage detected",
          "trigger_patterns": ["sugar is available", "precondition"],
          "controller": "SugarSolidManager",
          "implicit": true,
          "criticality": "medium"
        }
      ],
      "quality_functions": [
        {
          "name": "DosingPrecisionProtection",
          "constraint": "Precise dosing ±0.5g",
          "constraint_source": "operational_materials_addressing.material_types.sugar.quality_parameters",
          "trigger": "sugar dispensing operations",
          "trigger_patterns": ["adds sugar", "defined amount of sugar"],
          "controller": "SugarSolidManager",
          "implicit": true,
          "criticality": "low"
        }
      ]
    },

    "coffee_beans": {
      "quality_functions": [
        {
          "name": "FreshnessProtection",
          "constraint": "Cool, dark, dry storage",
          "constraint_source": "operational_materials_addressing.material_types.coffee_beans.storage_requirements",
          "trigger": "coffee beans storage detected",
          "trigger_patterns": ["coffee beans are available", "precondition"],
          "controller": "CoffeeSolidManager",
          "implicit": true,
          "criticality": "low"
        }
      ]
    },

    "filter": {
      "hygiene_functions": [
        {
          "name": "SingleUseProtection",
          "constraint": "Filter single-use only",
          "constraint_source": "hygiene_requirements.contamination_prevention.residue_removal",
          "trigger": "brewing completed",
          "trigger_patterns": ["End UC", "brewing completed"],
          "controller": "FilterManager",
          "implicit": true,
          "criticality": "medium"
        }
      ]
    }
  }
}
```

---

## Zusammenfassung

**12 implizite Protection Functions identifiziert**:
- 5 Safety Functions (High Criticality)
- 5 Hygiene Functions (High/Medium Criticality)
- 2 Quality Functions (Low Criticality)

Alle sind **funktionale Anforderungen** (WAS), nicht Lösungen (WIE)!
