# RL Knowledge Integration Architecture

## Core Question: Wie fließt das Erlernte zurück in das System?

Diese Datei beschreibt, WIE und WO das durch Reinforcement Learning erlernte Wissen im System gespeichert und angewendet wird.

---

## 1. Integration Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Learning System                        │
└──────────────┬──────────────────────────────────────────────┘
               │ Learned Knowledge
               ├──────────┬──────────┬──────────┬─────────────┐
               ▼          ▼          ▼          ▼             ▼
         Domain JSON   RL Models   Pattern DB  User Prefs  Analyzer
         (erweitert)   (.pth/.pkl) (.json)     (.json)     (unchanged)
```

### Wichtige Prinzipien:

✅ **JA - Erweitere bestehende Strukturen**
- Domain JSON wird erweitert (nicht ersetzt)
- Neue Sektionen für gelernte Patterns
- Abwärtskompatibel

✅ **JA - Separate RL Model Files**
- PyTorch Models (.pth) für neuronale Netze
- Pickle Files (.pkl) für einfachere Modelle
- Versioniert und austauschbar

✅ **JA - Pattern Databases**
- JSON/SQLite für gelernte NLP-Korrekturen
- Strukturierte Speicherung von Mustern

❌ **NEIN - Python Code NICHT generieren**
- Zu riskant, schwer wartbar
- Code bleibt regelbasiert

❌ **NEIN - SpaCy NICHT modifizieren**
- SpaCy ist externes pre-trained Model
- Stattdessen: Post-Processing Layer

---

## 2. Integration Layers (5 Ebenen)

### Layer 1: Domain JSON Extensions (Gelernte Domain-Patterns)

**Datei**: `domains/beverage_preparation.json`

**Was wird erweitert**:
```json
{
  "domain_name": "beverage_preparation",

  // EXISTING: Manuelle Definition
  "verb_classification": {
    "transaction_verbs": {...},
    "transformation_verbs": {...}
  },

  // NEW: RL-gelernte Ergänzungen
  "learned_patterns": {
    "version": "1.2.5",
    "last_updated": "2025-11-15T14:30:00Z",
    "training_samples": 1247,

    "implicit_requirements": [
      {
        "pattern": "milk.*foam",
        "implied_requirement": "TemperatureControl",
        "confidence": 0.89,
        "learned_from": 45,
        "user_approved": true
      },
      {
        "pattern": "espresso.*pressure",
        "implied_requirement": "PressureMonitoring",
        "confidence": 0.95,
        "learned_from": 78,
        "user_approved": true
      }
    ],

    "controller_preferences": {
      "heating_water": {
        "preferred_controller": "WaterLiquidManager",
        "abstraction_level": "material_based",
        "confidence": 0.92,
        "alternative": "HeatingController",
        "user_corrections": 3
      }
    },

    "nlp_corrections": [
      {
        "original_parse": "been heated",
        "correction": "is heated",
        "pattern": "been + past_participle",
        "context": "passive_voice_misparse",
        "frequency": 23
      }
    ]
  }
}
```

**Wie wird es angewendet**:
```python
# In DomainVerbLoader
def get_learned_patterns(self, domain_name: str) -> Dict:
    """Load learned patterns from domain JSON"""
    domain_config = self.domain_configs.get(domain_name, {})
    return domain_config.get('learned_patterns', {})

def get_implicit_requirements_for_text(self, domain: str, text: str) -> List[str]:
    """Check if text matches learned implicit requirement patterns"""
    learned = self.get_learned_patterns(domain)
    requirements = []

    for pattern_entry in learned.get('implicit_requirements', []):
        if pattern_entry['confidence'] > 0.8:  # Nur hohe Konfidenz
            if re.search(pattern_entry['pattern'], text, re.IGNORECASE):
                requirements.append(pattern_entry['implied_requirement'])

    return requirements
```

---

### Layer 2: RL Model Files (Neuronale Netze)

**Speicherort**: `models/rl/`

**Struktur**:
```
models/
├── rl/
│   ├── controller_selection/
│   │   ├── dqn_model_v1.2.5.pth          # PyTorch Model
│   │   ├── dqn_config.json                # Hyperparameter
│   │   └── training_metadata.json         # Training history
│   ├── protection_function/
│   │   ├── bandit_model_v1.1.0.pkl       # Scikit-learn Model
│   │   └── arm_statistics.json            # Bandit statistics
│   └── data_flow/
│       ├── policy_gradient_v0.9.3.pth
│       └── baseline_network.pth
```

**Model Loading**:
```python
# In StructuredUCAnalyzer.__init__()
class StructuredUCAnalyzer:
    def __init__(self, domain_name: str, rl_mode: str = "assistant"):
        # ... existing init ...

        # Load RL models if enabled
        self.rl_mode = rl_mode
        self.rl_agents = {}

        if rl_mode in ["assistant", "auto", "learning"]:
            self._load_rl_models()

    def _load_rl_models(self):
        """Load pre-trained RL models"""
        model_dir = Path("models/rl")

        # Controller Selection Agent
        controller_model_path = model_dir / "controller_selection" / "dqn_model_latest.pth"
        if controller_model_path.exists():
            self.rl_agents['controller'] = ControllerSelectionAgent.load(controller_model_path)
            print(f"[RL] Loaded controller selection agent (confidence: {self.rl_agents['controller'].avg_confidence})")

        # Protection Function Agent
        protection_model_path = model_dir / "protection_function" / "bandit_model_latest.pkl"
        if protection_model_path.exists():
            self.rl_agents['protection'] = ProtectionFunctionAgent.load(protection_model_path)
```

**Anwendung im Analyzer**:
```python
def _select_controller_for_material(self, material: str, action: str, context: str) -> str:
    """Select controller with RL assistance"""

    # Rule-based fallback (immer verfügbar)
    rule_based_controller = f"{material}LiquidManager"

    # RL suggestion (wenn model geladen)
    if 'controller' in self.rl_agents and self.rl_mode != "rules_only":
        state = AnalysisState(
            material=material,
            action=action,
            context=context,
            domain=self.domain_name
        )

        rl_suggestion = self.rl_agents['controller'].suggest(state)

        if self.rl_mode == "assistant":
            # Show both options to user
            print(f"[RL] Rule-based: {rule_based_controller}")
            print(f"[RL] RL suggests: {rl_suggestion.controller} (confidence: {rl_suggestion.confidence:.2f})")
            # User chooses via UI/CLI

        elif self.rl_mode == "auto" and rl_suggestion.confidence > 0.85:
            # Auto-apply if high confidence
            print(f"[RL] Using {rl_suggestion.controller} (confidence: {rl_suggestion.confidence:.2f})")
            return rl_suggestion.controller

    return rule_based_controller
```

---

### Layer 3: Learned Pattern Database (NLP-Korrekturen & Patterns)

**Datei**: `learned_knowledge/nlp_corrections.json`

**Struktur**:
```json
{
  "version": "1.0.3",
  "last_updated": "2025-11-15T14:30:00Z",
  "total_corrections": 127,

  "spacy_error_patterns": [
    {
      "id": "passive_been_error",
      "pattern": {
        "original_dep": "ROOT",
        "original_pos": "VBN",
        "context": "been + VERB",
        "spacy_output": "been heated",
        "expected_output": "is heated"
      },
      "correction_rule": {
        "replace": "been",
        "with": "is",
        "condition": "followed_by_past_participle"
      },
      "frequency": 47,
      "accuracy": 0.96,
      "examples": [
        {"uc_text": "Water been heated", "corrected": "Water is heated"},
        {"uc_text": "Milk been added", "corrected": "Milk is added"}
      ]
    },
    {
      "id": "compound_noun_split",
      "pattern": {
        "original_parse": "coffee + beans",
        "spacy_output": ["coffee", "beans"],
        "expected_output": "CoffeeBeans"
      },
      "correction_rule": {
        "merge_compounds": true,
        "patterns": ["coffee beans", "milk powder", "water tank"]
      },
      "frequency": 89,
      "accuracy": 0.94
    }
  ],

  "domain_specific_corrections": {
    "beverage_preparation": [
      {
        "text_pattern": "prepare.*espresso",
        "typical_spacy_error": "espresso as generic noun",
        "correction": "espresso as domain entity",
        "frequency": 34
      }
    ]
  }
}
```

**Anwendung**:
```python
# New: NLPCorrectionLayer
class NLPCorrectionLayer:
    """Post-processing layer to correct typical spaCy errors"""

    def __init__(self):
        self.corrections = self._load_corrections()

    def _load_corrections(self) -> Dict:
        path = Path("learned_knowledge/nlp_corrections.json")
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def correct_parse(self, doc: spacy.tokens.Doc, domain: str) -> spacy.tokens.Doc:
        """Apply learned corrections to spaCy parse"""

        # Apply known error corrections
        for error_pattern in self.corrections.get('spacy_error_patterns', []):
            if error_pattern['accuracy'] > 0.9:  # Nur zuverlässige
                doc = self._apply_correction(doc, error_pattern)

        # Apply domain-specific corrections
        domain_corrections = self.corrections.get('domain_specific_corrections', {}).get(domain, [])
        for correction in domain_corrections:
            doc = self._apply_domain_correction(doc, correction)

        return doc

# Integration in StructuredUCAnalyzer
class StructuredUCAnalyzer:
    def __init__(self, ...):
        # ...
        self.nlp_corrector = NLPCorrectionLayer()

    def _parse_uc_step(self, step_text: str) -> spacy.tokens.Doc:
        """Parse UC step with correction layer"""
        doc = self.nlp(step_text)

        # Apply learned corrections
        doc = self.nlp_corrector.correct_parse(doc, self.domain_name)

        return doc
```

---

### Layer 4: User Preference Profiles (Pro User/Projekt)

**Datei**: `user_profiles/{user_id}/preferences.json`

**Struktur**:
```json
{
  "user_id": "dieter_wagner",
  "created": "2025-10-01T10:00:00Z",
  "last_updated": "2025-11-15T14:30:00Z",
  "total_use_cases_analyzed": 47,

  "controller_preferences": {
    "abstraction_level": "material_based",  // vs. "process_based" or "functional_abstract"
    "confidence": 0.91,
    "learned_from_corrections": 12,

    "specific_mappings": {
      "heating_water": {
        "preferred": "WaterLiquidManager",
        "rejected": ["HeatingController", "ThermalManager"],
        "correction_count": 3
      },
      "grinding_beans": {
        "preferred": "CoffeeBeansLiquidManager",  // Note: should be SolidManager
        "rejected": ["GrindingController"],
        "correction_count": 1
      }
    }
  },

  "naming_conventions": {
    "controller_suffix": "Manager",  // vs. "Controller"
    "entity_style": "CamelCase",     // vs. "snake_case"
    "boundary_prefix": "HMI",        // User bevorzugt "HMI" prefix
    "confidence": 0.88
  },

  "implicit_requirement_sensitivity": {
    "protection_functions": "high",   // User will sehen alle Protection Functions
    "safety_requirements": "high",
    "hygiene_requirements": "medium",
    "performance_requirements": "low"
  },

  "domain_expertise": {
    "beverage_preparation": 0.95,  // Expert level
    "aerospace": 0.15,             // Beginner level
    "automotive": 0.60             // Intermediate level
  }
}
```

**Anwendung**:
```python
class UserPreferenceManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.prefs = self._load_preferences()

    def _load_preferences(self) -> Dict:
        path = Path(f"user_profiles/{self.user_id}/preferences.json")
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._create_default_profile()

    def get_controller_preference(self, context: str) -> Optional[str]:
        """Get user's preferred controller for a specific context"""
        mappings = self.prefs.get('controller_preferences', {}).get('specific_mappings', {})
        return mappings.get(context, {}).get('preferred')

    def record_correction(self, context: str, rejected: str, accepted: str):
        """Learn from user correction"""
        # Update preferences
        # Increment correction count
        # Recalculate confidence
        self._save_preferences()

# Integration in analyzer
class StructuredUCAnalyzer:
    def __init__(self, domain_name: str, user_id: str = "default"):
        # ...
        self.user_prefs = UserPreferenceManager(user_id)

    def _select_controller(self, material: str, action: str) -> str:
        # Check user preference first
        context = f"{action}_{material}"
        user_pref = self.user_prefs.get_controller_preference(context)

        if user_pref and self.user_prefs.get_confidence(context) > 0.8:
            print(f"[USER_PREF] Using {user_pref} based on your past preferences")
            return user_pref

        # Fall back to RL or rule-based
        # ...
```

---

### Layer 5: Python Analyzer Integration (Read-Only)

**Wichtig**: Der Python Analyzer Code wird NICHT generiert oder modifiziert!

**Stattdessen**: Der Analyzer lädt und nutzt die gelernten Informationen:

```python
# src/structured_uc_analyzer.py (MODIFIED, not generated)
class StructuredUCAnalyzer:
    """
    Enhanced with RL integration layers:
    - Loads learned patterns from domain JSON
    - Uses RL models for suggestions
    - Applies NLP correction layer
    - Respects user preferences
    """

    def __init__(self,
                 domain_name: str,
                 rl_mode: str = "assistant",
                 user_id: str = "default"):

        # EXISTING: Rule-based components
        self.nlp = spacy.load("en_core_web_md")
        self.verb_loader = DomainVerbLoader()
        self.controller_registry = MaterialControllerRegistry()

        # NEW: RL integration layers
        self.rl_mode = rl_mode
        self.user_prefs = UserPreferenceManager(user_id)
        self.nlp_corrector = NLPCorrectionLayer()
        self.rl_agents = {}

        if rl_mode in ["assistant", "auto", "learning"]:
            self._load_rl_models()

        # NEW: Load learned patterns from domain
        self.learned_patterns = self.verb_loader.get_learned_patterns(domain_name)

        print(f"[INIT] RL Mode: {rl_mode}")
        print(f"[INIT] User: {user_id}")
        print(f"[INIT] Learned patterns: {len(self.learned_patterns.get('implicit_requirements', []))}")

    def analyze_uc_file(self, uc_file_path: str) -> Tuple[List, List]:
        """
        Enhanced analysis with RL integration.
        The flow:
        1. Parse with SpaCy
        2. Apply NLP correction layer (learned)
        3. Apply rule-based analysis
        4. Get RL suggestions (if enabled)
        5. Apply user preferences
        6. Return results
        """
        # ... implementation uses all 4 layers above
```

**Wichtige Punkte**:
- Analyzer Code ist handgeschrieben und wartbar
- Analyzer LÄDT gelernte Informationen
- Analyzer NUTZT RL Models für Vorschläge
- Analyzer bleibt funktionsfähig OHNE RL (Fallback)

---

## 3. Data Flow: Vom Feedback zum integrierten Wissen

### Schritt-für-Schritt Beispiel:

**Ausgangssituation**: User korrigiert Controller-Auswahl

```
1. User Input UC Step: "Heat water to 95°C"

2. Analyzer Suggestion (Rule-based):
   Controller: "WaterLiquidManager"

3. RL Agent Suggestion:
   Controller: "HeatingController" (confidence: 0.72)

4. User Correction:
   User wählt: "WaterLiquidManager"
   Feedback: "I prefer material-based controllers"

5. Feedback Recording:
   ├─> Update user_profiles/dieter/preferences.json
   │   └─> controller_preferences.specific_mappings.heating_water.preferred = "WaterLiquidManager"
   │
   ├─> Update RL training buffer
   │   └─> Record: (state, action="HeatingController", reward=-1.0)
   │   └─> Record: (state, action="WaterLiquidManager", reward=+1.0)
   │
   └─> Trigger online learning (if enabled)
       └─> Update DQN weights

6. Next Time (Same UC Step):
   ├─> User Preference Layer: "WaterLiquidManager" (confidence: 0.85)
   ├─> RL Agent re-trained: Now suggests "WaterLiquidManager" (confidence: 0.88)
   └─> Result: Automatisch richtig vorgeschlagen
```

### Learning Pipeline:

```
User Feedback
    │
    ├──> Immediate Update (User Preferences)
    │    └─> user_profiles/{user}/preferences.json
    │
    ├──> Experience Buffer (RL Training)
    │    └─> training_data/experiences_{date}.json
    │
    ├──> Online Learning (Optional)
    │    └─> Update RL model weights
    │    └─> Save to models/rl/{agent}/model_v{version}.pth
    │
    └──> Batch Training (Weekly)
         └─> Aggregate all experiences
         └─> Train RL agents
         └─> Update domain JSON learned_patterns
         └─> Update NLP correction database
         └─> Validate & deploy new models
```

---

## 4. Konkrete Antworten auf die Fragen

### Frage 1: "Verbessert Domain JSON?"

**Antwort**: JA, aber nur erweitert, nicht ersetzt.

**Was wird hinzugefügt**:
```json
{
  "learned_patterns": {
    "implicit_requirements": [...],      // Gelernte implizite Anforderungen
    "controller_preferences": {...},     // Gelernte Controller-Auswahl
    "nlp_corrections": [...]             // Gelernte NLP-Korrekturen
  }
}
```

**Wie**:
- Wöchentlicher Batch-Update Prozess
- Nur Patterns mit Confidence > 0.8
- User-approved Patterns
- Versioniert (kann zurückgerollt werden)

**Beispiel**:
```python
# tools/update_domain_learned_patterns.py
def update_domain_from_training(domain_name: str, training_results: Dict):
    """Update domain JSON with newly learned patterns"""

    domain_path = Path(f"domains/{domain_name}.json")

    # Load existing
    with open(domain_path, 'r', encoding='utf-8') as f:
        domain_config = json.load(f)

    # Backup old version
    backup_path = domain_path.with_suffix(f'.backup_{datetime.now():%Y%m%d}.json')
    shutil.copy(domain_path, backup_path)

    # Update learned patterns
    if 'learned_patterns' not in domain_config:
        domain_config['learned_patterns'] = {}

    # Add high-confidence patterns only
    for pattern in training_results['new_implicit_requirements']:
        if pattern['confidence'] > 0.8 and pattern['user_approved']:
            domain_config['learned_patterns']['implicit_requirements'].append(pattern)

    # Increment version
    domain_config['learned_patterns']['version'] = increment_version(...)
    domain_config['learned_patterns']['last_updated'] = datetime.now().isoformat()

    # Save
    with open(domain_path, 'w', encoding='utf-8') as f:
        json.dump(domain_config, f, indent=2, ensure_ascii=False)

    print(f"[UPDATE] Domain {domain_name} updated with {len(training_results['new_implicit_requirements'])} new patterns")
```

---

### Frage 2: "Verbesserter Python Analyzer?"

**Antwort**: NEIN, Code wird NICHT generiert. Aber: Analyzer wird erweitert (manuell) um RL-Integration.

**Was ändert sich**:
- Analyzer lädt RL models
- Analyzer nutzt gelernte Patterns
- Analyzer bietet RL-Vorschläge an
- Analyzer ist abwärtskompatibel (funktioniert ohne RL)

**Code bleibt wartbar**:
```python
# Manuelle Erweiterung in structured_uc_analyzer.py
class StructuredUCAnalyzer:
    def _select_controller(self, material, action):
        # 1. Rule-based (always available)
        rule_based = f"{material}LiquidManager"

        # 2. User preference (if learned)
        if self.user_prefs.has_preference(material, action):
            return self.user_prefs.get_preferred_controller(material, action)

        # 3. RL suggestion (if model loaded)
        if self.rl_agents.get('controller'):
            rl_suggestion = self.rl_agents['controller'].suggest(state)
            if rl_suggestion.confidence > 0.85:
                return rl_suggestion.controller

        # 4. Fallback to rule-based
        return rule_based
```

**NICHT**:
```python
# WRONG: Code-Generierung
def generate_analyzer_code(rl_model):
    code = f"""
    def _select_controller(self, material, action):
        return {rl_model.predict(...)}
    """
    write_to_file("analyzer.py", code)  # ❌ GEFÄHRLICH!
```

---

### Frage 3: "Verbesserungen von SpaCy?"

**Antwort**: NEIN, SpaCy selbst wird NICHT modifiziert. Aber: Post-Processing Layer korrigiert bekannte Fehler.

**Warum nicht SpaCy fine-tunen?**
- SpaCy ist externes pre-trained Model (140MB+)
- Fine-tuning erfordert massive Datenmengen
- Zu komplex für spezifische UC-Domäne
- Updates von SpaCy würden Änderungen überschreiben

**Stattdessen: Correction Layer**
```python
class NLPCorrectionLayer:
    """Post-processes SpaCy output to fix known errors"""

    def correct_parse(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """Apply learned corrections"""

        # Fix 1: "been heated" -> "is heated"
        if self._matches_passive_been_error(doc):
            doc = self._fix_passive_been(doc)

        # Fix 2: Compound nouns "coffee beans" -> "CoffeeBeans"
        if self._has_compound_nouns(doc):
            doc = self._merge_compounds(doc)

        # Fix 3: Domain-specific entity recognition
        doc = self._apply_domain_entities(doc)

        return doc
```

**Vorteile**:
- SpaCy bleibt unangetastet (updatebar)
- Corrections sind nachvollziehbar
- Kann pro Domain unterschiedlich sein
- Einfach zu testen und zu debuggen

---

## 5. Deployment & Versioning

### Model Versioning

```
models/
├── rl/
│   ├── controller_selection/
│   │   ├── v1.0.0_baseline.pth         # Initial model
│   │   ├── v1.1.0_user_feedback.pth    # After 500 UCs
│   │   ├── v1.2.0_multi_domain.pth     # Multiple domains
│   │   └── latest -> v1.2.0_multi_domain.pth  # Symlink
│   └── metadata.json
```

### Rollback Mechanism

```python
# tools/rollback_rl_models.py
def rollback_to_version(agent_name: str, version: str):
    """Rollback RL model to previous version"""

    model_dir = Path(f"models/rl/{agent_name}")
    target_model = model_dir / f"v{version}.pth"

    if not target_model.exists():
        raise ValueError(f"Version {version} not found")

    # Update symlink
    latest_link = model_dir / "latest"
    latest_link.unlink()
    latest_link.symlink_to(target_model.name)

    print(f"[ROLLBACK] {agent_name} rolled back to v{version}")
```

---

## 6. Summary: Integration Points

| Component | Erlernt | Gespeichert in | Angewendet durch | Updatebar |
|-----------|---------|----------------|------------------|-----------|
| **Implizite Anforderungen** | Patterns aus User-Feedback | `domains/{domain}.json` `learned_patterns.implicit_requirements` | `DomainVerbLoader.get_implicit_requirements()` | Wöchentlich |
| **Controller-Auswahl** | Preferences & RL Training | `user_profiles/{user}/preferences.json` + `models/rl/controller_selection/*.pth` | `StructuredUCAnalyzer._select_controller()` | Online/Batch |
| **NLP-Korrekturen** | SpaCy Error Patterns | `learned_knowledge/nlp_corrections.json` | `NLPCorrectionLayer.correct_parse()` | Wöchentlich |
| **Protection Functions** | Trigger Confidence | `domains/{domain}.json` `learned_patterns.controller_preferences` | `StructuredUCAnalyzer._add_protection_functions()` | Wöchentlich |
| **User Naming** | Naming Conventions | `user_profiles/{user}/preferences.json` | `StructuredUCAnalyzer._format_names()` | Online |

---

## 7. Wichtigste Prinzipien

1. **Separation of Concerns**
   - Code bleibt regelbasiert und wartbar
   - Gelerntes Wissen in separaten Dateien (JSON, .pth)
   - Klare Schnittstellen

2. **Graceful Degradation**
   - System funktioniert OHNE RL
   - RL ist Enhancement, nicht Requirement
   - Fallback auf regelbasiert immer verfügbar

3. **Transparency**
   - User sieht was gelernt wurde
   - Confidence scores sind sichtbar
   - User kann Vorschläge ablehnen

4. **Versionierung**
   - Alle gelernten Modelle versioniert
   - Rollback möglich
   - Domain JSON Backups

5. **Incremental Learning**
   - Kleine, häufige Updates (User Prefs)
   - Größere, validierte Updates (Domain JSON)
   - Batch Training mit Qualitätssicherung

---

## Conclusion

Das erlernte Wissen fließt auf 4 Ebenen zurück:

1. **Domain JSON** - Erweitert mit `learned_patterns` Sektion
2. **RL Model Files** - PyTorch/Pickle Models für neuronale Entscheidungen
3. **Pattern Databases** - JSON für NLP-Korrekturen und Patterns
4. **User Preferences** - Individuelle Preferences pro User/Projekt

**Der Python Analyzer bleibt unverändert** (kein Code-Gen), lädt aber die gelernten Informationen und nutzt sie zur Verbesserung der Vorschläge.

**SpaCy bleibt unverändert**, aber ein Post-Processing Layer korrigiert bekannte Fehler basierend auf gelernten Patterns.
