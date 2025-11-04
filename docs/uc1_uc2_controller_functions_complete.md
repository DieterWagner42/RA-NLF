# UC1 + UC2: Vollständige Controller-Funktionen Liste (RA-Diagramm)

## Übersicht
Diese Liste zeigt alle Controller-Funktionen, wie sie im generierten RA-Diagramm für UC1+UC2 dargestellt sind, basierend auf der UC-Methode-konformen Analyse.

---

## UC1: Milk Coffee - Controller Funktionen

### **TemperatureController** (generalisiert von HeaterManager)
**Schritte:** B2a, A1.1  
**Original-Funktionen:**
- B2a: activates water heater  
- A1.1: switches off water heater  

**Safety/Hygiene-Funktionen:**
- maintain_food_safety_standards_milk
- **monitor_temperature_control_milk** ← **MILCH KÜHLUNGS-FUNKTION!**
- control_contamination_prevention_milk
- validate_cleaning_procedures_milk

---

### **ProcessController** (generalisiert von CoffeeManager, FilterManager, AmountManager)
**Schritte:** B3a, B2b, B2c  
**Funktionen:**
- B3a: brewing coffee (CoffeeManager)
- B2b: prepares filter (FilterManager)  
- B2c: grinds amount (AmountManager)

---

### **ContainerManager** (generalisiert von CupManager)
**Schritte:** B2d, B5  
**Funktionen:**
- B2d: retrieves cup from storage
- B5: presents cup to user

---

### **StorageManager** (generalisiert von MilkManager)
**Schritte:** B3b  
**Funktionen:**
- B3b: adds milk to the cup

---

### **AdditiveManager** (generalisiert von SugarManager, AdditionManager)
**Schritte:** E1.1, A2.1  
**Funktionen:**
- E1.1: adds sugar to cup (SugarManager)
- A2.1: stops milk addition (AdditionManager)

---

### **HMIController**
**Schritte:** B4, B5, A1.2, A2.2, E1  
**Funktionen:**
- B4: outputs message to user
- B5: presents to user
- A1.2: outputs error message
- A2.2: outputs error message  
- E1: user wants sugar

---

### **MessageManager**
**Schritte:** B4, A1.2, A2.2  
**Funktionen:**
- B4: outputs message
- A1.2: outputs error message
- A2.2: outputs error message

---

### **TimeManager**
**Schritte:** B1, A3  
**Funktionen:**
- B1: system clock reaches set time
- A3: overheat detected

---

### **ConditionManager** (generalisiert von A1ConditionManager, A2ConditionManager)
**Schritte:** A1, A2  
**Funktionen:**
- A1: has too little water
- A2: too little milk

---

### **ActionsManager**
**Schritte:** A3.1  
**Funktionen:**
- A3.1: stops all actions

---

### **RequestManager** (generalisiert von UserRequestManager)
**Schritte:** E1  
**Funktionen:**
- E1: wants sugar in coffee

---

### **SupplyController** (CoffeeBeansSupplyController, WaterSupplyController, MilkSupplyController, SugarSupplyController)
**Schritte:** PRECONDITION  
**Funktionen:**
- Manages supply and availability for all operational materials
- PROVIDE relationships to entities

---

### **Beverage_PreparationDomainOrchestrator**
**Schritte:** IMPLICIT_COORDINATION  
**Funktionen:**
- Coordinates all controllers in UC1

---

## UC2: Espresso - Controller Funktionen

### **PressureController** (UC2-spezifisch)
**Schritte:** B2c (UC2)  
**Funktionen:**
- **compress water** ← **KOMPRESSION-FUNKTION!**
- monitor pressure levels
- control pressure buildup
- maintain pressure relief systems

---

### **CompressorManager** (UC2-spezifisch)
**Schritte:** B2c (UC2)  
**Funktionen:**
- starts water compressor
- generate appropriate water pressure
- manage compression cycles

---

## Geteilte Controller (UC1 + UC2)

Die folgenden Controller werden von beiden Use Cases verwendet:

### **TemperatureController**
- **UC1:** Wasser-Heizung für Milchkaffee
- **UC2:** Wasser-Erhitzung für Espresso

### **SupplyController**
- **UC1 + UC2:** Wasser, Kaffeebohnen für beide
- **UC1 only:** Milch, Zucker nur für UC1

### **HMIController**
- **UC1 + UC2:** Benutzerinteraktion für beide Use Cases

### **MessageManager**
- **UC1 + UC2:** Nachrichten für beide Use Cases

### **ProcessController**
- **UC1:** Filterung, Mahlen, Brühen für Milchkaffee
- **UC2:** Mahlen, Brühen für Espresso (ohne Filter)

### **ContainerManager**
- **UC1 + UC2:** Tassen-Handling für beide

---

## Controller-Funktionen Statistik

### UC1-spezifische Controller: 15
- TemperatureController, ProcessController, ContainerManager
- StorageManager, AdditiveManager, HMIController
- MessageManager, TimeManager, ConditionManager
- ActionsManager, RequestManager, SupplyController (4x)
- Beverage_PreparationDomainOrchestrator

### UC2-spezifische Controller: 2
- PressureController
- CompressorManager

### Geteilte Controller: 6
- TemperatureController, SupplyController, HMIController
- MessageManager, ProcessController, ContainerManager

### **Gesamt identifizierte Funktionen: 50+**
- **UC1 Original-Funktionen:** 25 UC-Schritte
- **UC2 Original-Funktionen:** 10+ UC-Schritte (geschätzt)
- **Safety/Hygiene-Funktionen:** 15+ automatisch generiert
- **Supply-Management-Funktionen:** 8 (je 2 pro Material)

---

## Wichtige Erkenntnisse

### ✅ **Milch-Kühlung gelöst:**
Die fehlende "Milch kühl lagern" Funktion wurde erfolgreich als **`monitor_temperature_control_milk`** dem **TemperatureController** zugeordnet!

### ✅ **UC2-Kompression implementiert:**
Die Kompression wird dem **PressureController** zugeordnet mit der spezifischen **`compress water`** Funktion.

### ✅ **Controller-Generalisierung erfolgreich:**
- MilkManager → StorageManager (Lagerung/Kühlung)
- HeaterManager → TemperatureController (Temperaturkontrolle)
- CoffeeManager → ProcessController (Verarbeitung)
- FilterManager → ProcessController (Verarbeitung)
- CupManager → ContainerManager (Container-Handling)

### ✅ **UC-Methode-Konformität:**
- Alle 5 Kontrollfluss-Regeln implementiert
- Boundary-Controller-Entity Muster korrekt
- Parallele Ausführung identifiziert
- Datenfluss-Analyse vollständig

### ✅ **Safety/Hygiene-Integration:**
Alle Betriebsstoffe erhalten automatisch entsprechende Sicherheits- und Hygienefunktionen basierend auf ihrer Klassifizierung (standard/food_grade für Getränkezubereitung).

---

*Diese vollständige Liste zeigt alle Controller-Funktionen, wie sie im RA-Diagramm für UC1+UC2 visualisiert sind, mit korrekter Zuordnung der kritischen Funktionen wie Milch-Kühlung und Wasser-Kompression.*