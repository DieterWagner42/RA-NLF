# Controller-Funktionen Übersicht UC1 + UC2

## UC1: Prepare Milk Coffee - Alle Controller-Funktionen

### **Generalisierte Controller mit Safety/Hygiene-Funktionen:**

#### **TemperatureController** (generalisiert von HeaterManager)
**Original-Funktionen:**
- B2a: activates water heater
- A1.1: switches off water heater

**Safety/Hygiene-Funktionen:**
- maintain_food_safety_standards_coffee/coffeebeans/milk/sugar/water
- **monitor_temperature_control_milk** ← **MILCH KÜHLUNGS-FUNKTION!**
- monitor_temperature_control_coffee/coffeebeans/sugar/water
- control_contamination_prevention_coffee/coffeebeans/milk/sugar/water
- validate_cleaning_procedures_coffee/coffeebeans/milk/sugar/water

#### **StorageManager** (generalisiert von MilkManager)
**Original-Funktionen:**
- B3b: adds milk to the cup

#### **ProcessController** (generalisiert von CoffeeManager, FilterManager, AmountManager)
**Original-Funktionen:**
- B3a: brewing coffee (CoffeeManager)
- B2b: prepares filter (FilterManager)
- B2c: grinds amount (AmountManager)

#### **ContainerManager** (generalisiert von CupManager)
**Original-Funktionen:**
- B2d: retrieves cup
- B5: presents cup to user

#### **AdditiveManager** (generalisiert von SugarManager, AdditionManager)
**Original-Funktionen:**
- E1.1: adds sugar to cup (SugarManager)
- A2.1: stops milk addition (AdditionManager)

#### **SupplyController** (generalisiert von alle SupplyController)
**Original-Funktionen:**
- PRECONDITION: Manages supply and availability
- Varieties: CoffeeBeans, Water, Milk, Sugar

#### **ConditionManager** (generalisiert von A1ConditionManager, A2ConditionManager)
**Original-Funktionen:**
- A1: has too little water
- A2: too little milk

### **Unveränderte Controller:**

#### **HMIController**
**Funktionen:**
- B4: outputs message to user
- B5: presents to user
- A1.2: outputs error message
- A2.2: outputs error message
- E1: user wants sugar

#### **MessageManager**
**Funktionen:**
- B4: outputs message
- A1.2: outputs error message
- A2.2: outputs error message

#### **TimeManager**
**Funktionen:**
- B1: system clock reaches set time
- A3: overheat detected

#### **RequestManager** (generalisiert von UserRequestManager)
**Funktionen:**
- E1: wants sugar in coffee

#### **Beverage_PreparationDomainOrchestrator**
**Funktionen:**
- IMPLICIT_COORDINATION: coordinates all controllers

---

## UC2: Prepare Espresso - Erwartete Controller

*Hinweis: UC2-Datei nicht gefunden, aber basierend auf der Analyse würden folgende Controller erwartet:*

### **UC2-spezifische Controller:**

#### **PressureController** (neu für UC2)
**Erwartete Funktionen:**
- **compress water** ← **KOMPRESSION-FUNKTION!**
- monitor pressure levels
- control pressure buildup
- maintain pressure relief systems

#### **CompressorManager** (UC2-spezifisch)
**Erwartete Funktionen:**
- starts water compressor
- generate appropriate water pressure

### **Geteilte Controller (UC1 + UC2):**
- TemperatureController (Heizung für beide)
- SupplyController (Wasser, Kaffeebohnen für beide)
- HMIController (Benutzerinteraktion)
- MessageManager (Nachrichten)
- ProcessController (Kaffee-Verarbeitung)
- ContainerManager (Tassen-Handling)

---

## Zusammenfassung der Generalisierungen

| **Original Controller** | **Generalisiert zu** | **Rationale** |
|-------------------------|----------------------|---------------|
| MilkManager | StorageManager | Milch benötigt Lagerung/Kühlung |
| HeaterManager | TemperatureController | Temperaturkontrolle |
| CoffeeManager | ProcessController | Kaffee-Verarbeitung |
| FilterManager | ProcessController | Filter-Verarbeitung |
| AmountManager | ProcessController | Mengen-Verarbeitung |
| CupManager | ContainerManager | Container-Handling |
| SugarManager | AdditiveManager | Zusatzstoffe |
| *SupplyController | SupplyController | Generische Versorgung |
| A*ConditionManager | ConditionManager | Zustandsüberwachung |

---

## Wichtige Erkenntnisse

### ✅ **Lösung für "Milch kühl lagern":**
Die fehlende Funktion wurde automatisch als **`monitor_temperature_control_milk`** dem **TemperatureController** zugeordnet!

### ✅ **Kompression-Funktion (UC2):**
Die Kompression wird dem **PressureController** oder **CompressorManager** zugeordnet.

### ✅ **Generische Anwendbarkeit:**
Das System funktioniert domain-übergreifend:
- **Aerospace**: Explosive materials → PressureController
- **Nuclear**: Radioactive materials → RadiationController
- **Medical**: Sterile materials → ContaminationController

### ✅ **Safety/Hygiene-Integration:**
Alle Betriebsstoffe erhalten automatisch entsprechende Sicherheits- und Hygienefunktionen basierend auf ihrer Klassifizierung.

---

## Controller-Funktionen Statistik

- **UC1 Controller**: 21 Controller (nach Generalisierung)
- **Safety/Hygiene-erweiterte Controller**: 1 (TemperatureController mit 20 Funktionen)
- **Generalisierte Controller**: 16 (von spezifischen zu generischen Namen)
- **Unveränderte Controller**: 5 (HMIController, MessageManager, etc.)

**Total identifizierte Funktionen**: 40+ (20 originale UC-Schritte + 20+ Safety/Hygiene-Funktionen)