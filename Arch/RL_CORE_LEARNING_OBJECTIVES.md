# RL Core Learning Objectives - Was soll eigentlich gelernt werden?

## Die zentrale Frage

**Was ist der Kern des Problems, das RL lösen soll?**

---

## 1. Das aktuelle Problem

### Was der Analyzer HEUTE kann (sehr gut):

```
✅ UC-Methode Regeln 1-5 anwenden (regelbasiert, perfekt)
✅ RUP Stereotypes korrekt zuordnen (formal definiert)
✅ Parallel Flow Nodes generieren (Regel 2-5, deterministisch)
✅ Control Flows erstellen (sequenziell → parallel, eindeutig)
```

### Was der Analyzer HEUTE nicht kann (die LÜCKE):

```
❌ Mehrdeutige UC-Texte korrekt interpretieren
❌ Domänen-spezifische implizite Annahmen erkennen
❌ User-Intention bei unklaren Formulierungen verstehen
❌ Von User-Korrekturen systematisch lernen
❌ Sich an neue Domänen/Patterns adaptieren
```

---

## 2. Die Interpretations-Lücke

### Beispiel 1: "System activates the water heater"

**UC-Text (natürlichsprachig)**: "System activates the water heater"

**NLP-Analyse (spaCy)**:
- Verb: "activate"
- Direct Object: "water heater"
- Verb Type: Action verb (nicht transformation!)

**Regelbasierte Interpretation (aktuell)**:
```python
# 1. Verb "activate" → nicht in transformation_verbs
# 2. Direct Object "water heater" enthält "water"
# 3. Implementation element "heater" erkannt
# 4. Funktion: heat() (aus implementation_elements mapping)
# 5. Material: water
# 6. Controller: WaterLiquidManager
```

**Problem: Diese Interpretation ist HEURISTISCH, nicht gelernt!**

Was wenn:
- User meint eigentlich "WaterHeatingController"? (user preference)
- In dieser Domäne ist "heater" NICHT implementation element? (domain-specific)
- "activate" hat besondere Bedeutung? (domain verb)

### Was RL lernen soll:

**Lerne: Welcher Controller ist in diesem KONTEXT korrekt?**

```python
# Input (State):
state = {
    "verb": "activate",
    "object": "water heater",
    "domain": "beverage_preparation",
    "step_id": "B2a",
    "previous_steps": ["B1: system clock reaches 7am"],
    "preconditions": ["Water is available"],
    "similar_steps_in_other_UCs": [...],
    "user_history": [...]  # Hat dieser User vorher ähnliches korrigiert?
}

# Output (Action):
# Option A: WaterLiquidManager.heat() (regelbasiert)
# Option B: WaterHeatingController.activate() (user preference?)
# Option C: HeatingSystemManager.control() (domain-specific?)

# RL lernt: Welche Option führt zu User-Akzeptanz?
```

---

## 3. Was NICHT gelernt werden soll

### ❌ Formale Regeln (DÜRFEN NICHT geändert werden):

1. **UC-Methode Regeln 1-5**
   - Rule 1: Serial → Serial (direkte Verbindung)
   - Rule 2: Serial → Parallel (PX_START einfügen)
   - Rule 3-5: Parallel Flow Logik
   - **Grund**: Diese sind FORMALE DEFINITIONEN, keine Interpretationen

2. **RUP Stereotypes**
   - Actor, Boundary, Controller, Entity
   - <<boundary>>, <<controller>>, <<entity>>
   - **Grund**: RUP-Standard, nicht verhandelbar

3. **Grundlegende Architekturprinzipien**
   - Material-basierte Controller (nicht Verben!)
   - Actor → Boundary → Controller (nie direkt!)
   - Aggregation States (solid/liquid/gas)
   - **Grund**: Kern-Design-Entscheidungen

### ✅ Was gelernt werden SOLL:

**Die "Soft Decisions" - wo Interpretation nötig ist:**

---

## 4. Learning Objective 1: Controller Selection Heuristic

### Problem:
Bei mehrdeutigen Texten gibt es mehrere plausible Controller.

### Beispiele:

**Beispiel A: "System heats water"**
```
Option 1: WaterLiquidManager.heat()     ← Regelbasiert (aktuell)
Option 2: WaterHeatingController.heat() ← User könnte das bevorzugen
Option 3: ThermalManager.controlHeating() ← Domain-spezifisch?

RL-Frage: Welche Option wählt dieser User in dieser Domäne normalerweise?
```

**Beispiel B: "System adds milk"**
```
Option 1: MilkLiquidManager.add()       ← Regelbasiert
Option 2: CoffeeLiquidManager.add()     ← Coffee ist das Ziel (alternative Sicht)
Option 3: MixingController.addIngredient() ← Prozess-orientiert

RL-Frage: Wie abstrakt/konkret will dieser User die Controller?
```

### Was RL lernen soll:

**Lerne: Material-Controller vs. Process-Controller vs. Abstract-Controller**

```python
class ControllerAbstractionPreference:
    """
    Jeder User/Projekt hat eine Präferenz für Abstraktionslevel:

    Level 1: Material-basiert (WaterLiquidManager, MilkLiquidManager)
    Level 2: Prozess-basiert (HeatingController, MixingController)
    Level 3: Funktional-abstrakt (ThermalManager, IngredientManager)

    RL lernt: Welches Level bevorzugt dieser User?
    """
    pass

# Training:
# - User akzeptiert WaterLiquidManager 10 mal → bevorzugt Level 1
# - User korrigiert zu HeatingController 3 mal → bevorzugt Level 2
# - RL passt zukünftige Vorschläge an
```

---

## 5. Learning Objective 2: Implicit Context Detection

### Problem:
Implizite Annahmen im UC-Text sind nicht explizit genannt.

### Beispiele:

**Beispiel A: "Coffee beans are available" (Precondition)**
```
Explizit im Text: Coffee beans exist
Implizit (nicht genannt):
- Coffee beans müssen frisch sein (FreshnessProtection)
- Coffee beans müssen trocken gelagert werden
- Coffee beans müssen gemahlen werden (vor Verwendung)

RL-Frage: Welche impliziten Requirements sind in DIESER Domäne üblich?
```

**Beispiel B: "System grinds coffee beans with grinding degree"**
```
Explizit: grind, coffee beans, grinding degree (Entity)
Implizit (KÖNNTE relevant sein):
- Mahlgrad-Einstellung muss validiert werden (im Bereich?)
- Geräuschpegel-Limit (in manchen Domänen)
- Energieverbrauch-Monitoring (in Smart-Home?)

RL-Frage: Welche impliziten Aspekte sind wichtig in DIESEM Kontext?
```

### Was RL lernen soll:

**Lerne: Domänen-typische implizite Requirements**

```python
class ImplicitRequirementDetector:
    """
    Verschiedene Domänen haben verschiedene implizite Requirements:

    Beverage Preparation:
    - Hygiene (immer wichtig)
    - Freshness (Material-abhängig)
    - Temperature control (für Flüssigkeiten)

    Aerospace:
    - Safety (kritisch)
    - Redundancy (immer)
    - Real-time constraints (immer)

    Automotive:
    - Safety (kritisch)
    - Emissions (immer)
    - User comfort (wichtig)

    RL lernt: Welche impliziten Requirements sind typisch für diese Domäne?
    """
    pass

# Training:
# - In 100 Beverage UCs wird immer Hygiene erwähnt → hohe Wahrscheinlichkeit
# - In 100 Aerospace UCs wird immer Redundancy erwähnt → hohe Wahrscheinlichkeit
# - RL schlägt vor: "In Aerospace, soll ich RedundantSensorProtection hinzufügen?"
```

---

## 6. Learning Objective 3: NLP Error Correction

### Problem:
spaCy macht manchmal Fehler bei der grammatischen Analyse.

### Beispiele:

**Beispiel A: "System begins brewing coffee"**
```
spaCy-Analyse:
- Verb: "begin" (auxiliary)
- Main Verb: "brewing" (gerund/noun?)
- Direct Object: "coffee"

Richtige Interpretation:
- Verb: "brew" (transformation verb)
- Direct Object: "coffee"
- Transformation: "GroundCoffee + HotWater -> Coffee"

Problem: spaCy erkennt "brewing" nicht als Hauptverb
```

**Beispiel B: "The system grinds the user defined amount of coffee beans"**
```
spaCy-Analyse:
- Direct Object: "amount" (falsch!)
- Prep Phrase: "of coffee beans"

Richtige Interpretation:
- Direct Object: "coffee beans"
- Modifier: "user defined amount"

Problem: spaCy denkt "amount" ist Direct Object
```

### Was RL lernen soll:

**Lerne: Korrigiere typische NLP-Fehler basierend auf Kontext**

```python
class NLPErrorCorrection:
    """
    Typische spaCy-Fehler in UC-Texten:

    Fehler-Typ 1: Auxiliary verb als main verb erkannt
    - "begins brewing" → main verb sollte "brew" sein
    - "starts heating" → main verb sollte "heat" sein

    Fehler-Typ 2: Falsche Direct Object Erkennung
    - "grinds amount of beans" → DO sollte "beans" sein
    - "adds quantity of sugar" → DO sollte "sugar" sein

    Fehler-Typ 3: Compound nouns nicht erkannt
    - "coffee beans" → ein Konzept, nicht zwei
    - "grinding degree" → ein Konzept

    RL lernt: Korrigiere diese Fehler basierend auf Domain-Wissen
    """
    pass

# Training:
# - User korrigiert "begin" → "brew" 5 mal
# - RL lernt: Bei "begin/start + gerund" → nutze gerund als Hauptverb
# - Zukünftig: Automatische Korrektur
```

---

## 7. Learning Objective 4: Domain Pattern Recognition

### Problem:
Jede Domäne hat typische Patterns, die sich wiederholen.

### Beispiele:

**Pattern 1: Temperature Control (überall in Beverage)**
```
Wenn Verb = "heat" DANN:
- Controller: XxxLiquidManager (material-based)
- Implizite Function: OverheatProtection
- Typische Entity: HotWater, HotMilk, etc.

RL lernt: In beverage_preparation ist "heat" immer mit Protection Function
```

**Pattern 2: Inventory Check (überall in Manufacturing)**
```
Wenn Text enthält "check inventory" ODER "available" DANN:
- Boundary: XxxSupplyBoundary
- Controller: XxxDataManager (für inventory_data)
- Protection Functions: DataQualityProtection, DataFreshnessProtection

RL lernt: Inventory = immer Data Source + Quality Protection
```

**Pattern 3: User Interaction (überall)**
```
Wenn Text enthält "User wants" ODER "User requests" DANN:
- Flow: User → XxxRequestBoundary → HMIManager → XxxController
- NIEMALS: User → Controller (direkt)

RL lernt: User interaction = immer HMI Pattern
```

### Was RL lernen soll:

**Lerne: Domänen-spezifische Muster und Best Practices**

```python
class DomainPatternLearner:
    """
    Lerne Patterns aus User-Akzeptanz:

    Pattern: (Trigger, Action, Outcome)

    Beispiel:
    Trigger: verb="heat" AND domain="beverage"
    Action: Add OverheatProtection
    Outcome: User accepts 95%
    → Pattern learned: heat → OverheatProtection (high confidence)

    Trigger: verb="check" AND object contains "inventory"
    Action: Use ERP data source
    Outcome: User accepts 90%
    → Pattern learned: inventory check → ERP (high confidence)
    """

    def learn_pattern(self, trigger, action, outcome):
        if outcome.user_accepted:
            self.pattern_db.increment_confidence(trigger, action)
        else:
            self.pattern_db.decrement_confidence(trigger, action)

    def suggest_action(self, trigger):
        patterns = self.pattern_db.get_patterns(trigger)
        # Return action with highest confidence
        return max(patterns, key=lambda p: p.confidence)
```

---

## 8. Learning Objective 5: User Preference Learning

### Problem:
Verschiedene Users/Teams haben verschiedene Präferenzen.

### Beispiele:

**Präferenz A: Abstraktionslevel**
```
User A bevorzugt: WaterLiquidManager, MilkLiquidManager (konkret)
User B bevorzugt: LiquidManager (abstrakt)
User C bevorzugt: HeatingController, CoolingController (prozess-orientiert)

RL lernt: Welcher User bevorzugt welchen Stil?
```

**Präferenz B: Protection Function Granularität**
```
User A bevorzugt: Alle Protection Functions (sehr vorsichtig)
User B bevorzugt: Nur high-criticality (pragmatisch)
User C bevorzugt: Nur Safety (minimal)

RL lernt: Welches Criticality-Level pro User?
```

**Präferenz C: Entity-Detailgrad**
```
User A bevorzugt: Viele Entities (detailliert)
User B bevorzugt: Wenige Entities (high-level)

RL lernt: Abstraktionsgrad für Entities pro User
```

### Was RL lernen soll:

**Lerne: User-spezifische Präferenzen und adaptiere**

```python
class UserPreferenceLearner:
    """
    Track User-Präferenzen über Zeit:
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.preferences = {
            "controller_abstraction_level": None,  # 1-3
            "protection_function_criticality": None,  # "all", "high", "critical"
            "entity_detail_level": None,  # "detailed", "moderate", "minimal"
            "naming_convention": None,  # "material-based", "process-based"
        }

    def update_from_feedback(self, decision_type, chosen_option, rejected_options):
        """
        Beispiel:
        decision_type = "controller_selection"
        chosen_option = "WaterLiquidManager" (material-based)
        rejected_options = ["HeatingController", "ThermalManager"]

        → Update: User bevorzugt material-based naming
        """
        if decision_type == "controller_selection":
            if self._is_material_based(chosen_option):
                self.preferences["naming_convention"] = "material-based"
            elif self._is_process_based(chosen_option):
                self.preferences["naming_convention"] = "process-based"

    def predict_preference(self, decision_type, options):
        """
        Basierend auf gelernten Präferenzen, welche Option würde User wählen?
        """
        if decision_type == "controller_selection":
            pref = self.preferences["naming_convention"]
            if pref == "material-based":
                return [opt for opt in options if self._is_material_based(opt)][0]

        return options[0]  # Default: erste Option
```

---

## 9. Mein konkreter Plan - Phasen-Ansatz

### Phase 0: Fundamentals (JETZT - vor RL)

**Ziel**: Logging & Feedback Infrastructure

```python
# 1. Decision Logging
class DecisionLogger:
    """Log JEDE Entscheidung mit vollem Kontext"""
    def log_decision(self, decision_type, state, action, alternatives):
        # Speichere in SQLite
        pass

# 2. User Feedback Collection
class FeedbackCollector:
    """Sammle User Feedback"""
    def collect_feedback(self, analysis_id):
        # UI: Accept/Reject Buttons
        # UI: Star Rating
        # UI: Correction Interface
        pass

# 3. Experience Database
class ExperienceDB:
    """Persistente Speicherung"""
    # SQLite schema für experiences
    pass
```

**Deliverables**:
- [ ] Decision logging in analyzer integriert
- [ ] Feedback UI (einfach)
- [ ] SQLite database mit schema
- [ ] 100 UCs analysiert mit Logging

---

### Phase 1: Single Agent - Controller Selection (3 Monate)

**Ziel**: BEWEISE dass RL funktioniert mit EINEM Agent

**Scope**:
- Nur 1 Agent: Controller Selection
- Nur 1 Domain: beverage_preparation
- Nur 1 Mode: Assistant (User entscheidet)
- Nur 1 Metrik: User Acceptance Rate

**Approach**:
```python
class ControllerSelectionAgent:
    """
    DQN Agent für Controller Selection

    State: (verb, object, domain, step_id, context)
    Actions: [WaterLiquidManager, HeatingController, ThermalManager, ...]
    Reward: +1 (accept), -1 (reject/correct)
    """

    def select_controller(self, state):
        # Epsilon-greedy
        if random.random() < epsilon:
            return random.choice(self.controllers)
        else:
            return self.q_network.predict(state).argmax()
```

**Training Data**:
- Synthetic: 500 UCs mit ground truth
- Real: 500 UCs mit User feedback
- Total: 1000 UCs × ~5 decisions = 5000 experiences

**Success Criteria**:
- RL Agent Accuracy ≥ 80% (vs. 70% baseline)
- User Acceptance Rate ≥ 85%
- Inference Time < 100ms

**Deliverables**:
- [ ] ControllerSelectionAgent implementiert
- [ ] Training pipeline (offline)
- [ ] Assistant Mode UI
- [ ] Evaluation metrics
- [ ] A/B Test: RL vs. Rule-based

---

### Phase 2: Expand Agents (3 Monate)

**Nach erfolgreicher Phase 1**:

**Ziel**: Füge 2 weitere Agents hinzu

**Agents**:
1. ControllerSelectionAgent (bereits fertig)
2. ProtectionFunctionAgent (neu)
3. DataFlowAgent (neu)

**Success Criteria**:
- 3 Agents operational
- Combined accuracy ≥ 85%
- User satisfaction ≥ 4/5 stars

---

### Phase 3: Multi-Domain Support (3 Monate)

**Ziel**: Transfer Learning zwischen Domains

**Domains**:
- beverage_preparation (bereits trainiert)
- smart_manufacturing (neu)
- aerospace (neu)

**Approach**:
- Pre-train auf beverage
- Fine-tune auf smart_manufacturing
- Measure transfer learning effectiveness

---

### Phase 4: Auto Mode (3 Monate)

**Ziel**: RL macht Entscheidungen automatisch

**Requirements**:
- RL Confidence ≥ 90% für Auto-Entscheidung
- User kann jederzeit überschreiben
- Fallback zu Rule-based bei low confidence

---

## 10. Messbarer Erfolg - KPIs

### Kurzfristig (3 Monate - Phase 1):

```
Metrik 1: Controller Selection Accuracy
Baseline: 70% (regelbasiert)
Target: 80% (RL Agent)
Measurement: % korrekte Vorhersagen auf Test Set

Metrik 2: User Acceptance Rate
Baseline: 75% (User akzeptiert ohne Korrektur)
Target: 85%
Measurement: % accepted decisions

Metrik 3: Manual Correction Rate
Baseline: 25% (User muss korrigieren)
Target: 15%
Measurement: % decisions corrected

Metrik 4: Time to Analysis
Baseline: 10 minutes (mit manuellen Korrekturen)
Target: 7 minutes (weniger Korrekturen)
Measurement: Durchschnittliche Zeit pro UC
```

### Mittelfristig (6 Monate - Phase 2):

```
Metrik 5: Multi-Agent Accuracy
Target: 85% (combined 3 agents)

Metrik 6: User Satisfaction
Target: 4.2/5 stars

Metrik 7: Learning Efficiency
Target: 90% accuracy nach 1000 experiences
```

### Langfristig (12 Monate - Phase 4):

```
Metrik 8: Auto Mode Success Rate
Target: 90% (decisions made automatically, accepted by user)

Metrik 9: Domain Transfer Effectiveness
Target: 80% accuracy on new domain with 100 examples (vs. 1000 from scratch)

Metrik 10: Business Impact
Target: 50% reduction in manual effort
```

---

## 11. Risiko-Mitigation

### Risiko 1: Zu wenig Trainingsdaten

**Mitigation**:
- Start with synthetic data (500 UCs)
- Data augmentation (paraphrase UCs)
- Active learning (ask user for unclear cases)

### Risiko 2: RL degradiert Qualität

**Mitigation**:
- Assistant Mode first (user decides)
- Confidence thresholds (only suggest if confident)
- Always allow fallback to rule-based

### Risiko 3: User vertraut RL nicht

**Mitigation**:
- Transparency (show confidence scores)
- Explainability (why this suggestion?)
- Gradual rollout (A/B testing)

### Risiko 4: Overfitting zu spezifischem User

**Mitigation**:
- Regularization
- Cross-validation
- Test auf verschiedenen Users

---

## 12. Zusammenfassung - Was soll gelernt werden?

### KERN-ANTWORT:

**Lerne die INTERPRETATIONS-LÜCKE zwischen UC-Text und UC-Methode-Analyse:**

1. **Controller Selection Heuristic**
   - Material-basiert vs. Prozess-basiert vs. Abstrakt
   - User-Präferenz für Naming Conventions

2. **Implicit Context Detection**
   - Domänen-typische implizite Requirements
   - Protection Functions die NICHT im UC stehen

3. **NLP Error Correction**
   - Typische spaCy-Fehler korrigieren
   - Domänen-spezifisches Vokabular

4. **Domain Pattern Recognition**
   - Wiederkehrende Muster in Domäne
   - Best Practices aus User-Feedback

5. **User Preference Learning**
   - Abstraktionslevel
   - Granularität
   - Naming Conventions

### NICHT lernen:

- ❌ UC-Methode Regeln 1-5
- ❌ RUP Stereotypes
- ❌ Grundlegende Architektur

### START:

**Phase 1 (3 Monate)**:
- 1 Agent (Controller Selection)
- 1 Domain (beverage_preparation)
- 1 Mode (Assistant)
- 1 Metrik (Accuracy ≥ 80%)

**Dann skalieren basierend auf Erfolg.**

---

## Das ist mein Plan. 🎯
