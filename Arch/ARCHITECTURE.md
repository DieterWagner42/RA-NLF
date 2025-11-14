# RA-NLF Software Architecture
## Robustness Analysis Framework with Natural Language Foundation

**Version**: 2.1
**Last Updated**: 2025-11-14
**Domain**: Domain-Independent UC-Methode Robustness Analysis

---

## Documentation Structure

- **ARCHITECTURE.md** (this document) - High-level system architecture overview
- **ARCHITECTURE_DETAILED_PIPELINE.md** - Complete step-by-step analysis pipeline with all UC-Methode rules
- **ARCHITECTURE_DIAGRAM.png** - Layered architecture visualization
- **ARCHITECTURE_COMPONENTS.png** - Component interaction diagram
- **ARCHITECTURE_DATAFLOW.png** - Analysis pipeline flow diagram

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architectural Principles](#architectural-principles)
3. [Core Components](#core-components)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Analysis Pipeline](#analysis-pipeline) ⭐ See ARCHITECTURE_DETAILED_PIPELINE.md for complete details
6. [Domain-Driven Design](#domain-driven-design)
7. [Multi-UC Architecture](#multi-uc-architecture)
8. [Visualization Layer](#visualization-layer)
9. [Extension Points](#extension-points)

---

## System Overview

### Purpose

The RA-NLF (Robustness Analysis - Natural Language Foundation) framework implements systematic robustness analysis for automated systems using the UC-Methode approach. It converts natural language use case specifications into RUP-compliant robustness analysis diagrams.

### Key Capabilities

- **Domain-Independent Analysis**: NLP-based analysis using spaCy with domain JSON configurations
- **Material-Based Controllers**: Aggregation state-aware controller generation (solid/liquid/gas)
- **Multi-UC Support**: Shared component registry across multiple use cases
- **Config-Driven Workflows**: JSON-based configuration for batch processing
- **RUP Compliance**: Official Wikipedia RUP symbols and UC-Methode Rules 1-5

### Technology Stack

```
┌─────────────────────────────────────────┐
│ Python 3.x                              │
├─────────────────────────────────────────┤
│ Core Libraries:                         │
│  • spaCy (en_core_web_md) - NLP        │
│  • Matplotlib - Pure RUP Diagrams      │
│  • SVG Generation - RUP Symbols        │
├─────────────────────────────────────────┤
│ Domain Configurations:                  │
│  • JSON (domains/*.json)               │
│  • UC Files (Use Case/*.txt)           │
└─────────────────────────────────────────┘
```

---

## Architectural Principles

### 1. Domain-Driven Design

**Principle**: All domain knowledge is externalized in JSON configurations, not hardcoded.

```
domains/
├── common_domain.json              # Universal verbs & patterns
├── beverage_preparation.json       # Coffee/beverage domain
├── rocket_science.json             # Aerospace domain
├── automotive.json                 # Automotive domain
└── universal_operational_materials.json
```

**Benefits**:
- Domain switching without code changes
- Easy extension to new domains
- Clear separation of concerns

### 2. Material-Based Controllers

**Principle**: Controllers represent MATERIALS (not actions), verbs become FUNCTIONS.

```python
# Traditional (WRONG):
BrewingController      # ❌ Action-based
GrindingController     # ❌ Action-based

# Material-Based (CORRECT):
CoffeeLiquidManager    # ✅ Material + State
  └── functions: [brew(), heat()]
CoffeeSolidManager     # ✅ Material + State
  └── functions: [grind(), store()]
```

**Benefits**:
- Reusable across use cases
- Clear material lifecycle
- Aggregation state awareness

### 3. Configuration Over Code

**Principle**: Analysis behavior driven by JSON configurations.

```json
{
  "analysis_name": "Multi-UC Analysis",
  "domain": "beverage_preparation",
  "use_cases": [
    {"id": "UC1", "file": "Use Case/UC1.txt", "enabled": true},
    {"id": "UC2", "file": "Use Case/UC2.txt", "enabled": true}
  ],
  "options": {
    "share_controllers": true,
    "generate_combined_diagram": true
  }
}
```

**Benefits**:
- Batch processing support
- Reproducible analysis
- CI/CD integration ready

### 4. Layered Architecture

```
┌──────────────────────────────────────────────────┐
│         Presentation Layer                       │
│  (SVG, PNG, CSV, JSON outputs)                  │
├──────────────────────────────────────────────────┤
│         Application Layer                        │
│  (StructuredUCAnalyzer, Multi-UC Manager)       │
├──────────────────────────────────────────────────┤
│         Domain Layer                             │
│  (Material Controllers, Control Flow Rules)      │
├──────────────────────────────────────────────────┤
│         Infrastructure Layer                     │
│  (DomainVerbLoader, GenerativeContextManager)   │
├──────────────────────────────────────────────────┤
│         Data Layer                               │
│  (Domain JSON, UC Files, NLP Models)            │
└──────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Structured UC Analyzer
**File**: `src/structured_uc_analyzer.py` (6057 lines)

**Purpose**: Main analysis engine that processes UC files line-by-line.

**Key Responsibilities**:
- UC file parsing and section detection
- NLP-based grammatical analysis
- RA class generation (Actors, Boundaries, Controllers, Entities)
- Control flow generation (UC-Methode Rules 1-5)
- Data flow analysis (USE/PROVIDE relationships)
- Multi-UC coordination

**Core Classes**:

```python
class StructuredUCAnalyzer:
    """Main analysis engine"""

    # Configuration
    domain_name: str
    nlp: spacy.Language  # en_core_web_md

    # Components
    verb_loader: DomainVerbLoader
    context_manager: GenerativeContextManager
    material_registry: MaterialControllerRegistry

    # State
    line_analyses: List[LineAnalysis]
    uc_context: UCContext

    # Main Methods
    def analyze_uc_file(file_path: str) -> Tuple[List, str]
    def analyze_from_config(config_file: str) -> Dict
```

**Analysis Phases**:
1. **Section Detection**: Identify Capability, Goal, Preconditions, Basic Flow, etc.
2. **Grammatical Analysis**: Extract verbs, objects, prepositions using spaCy
3. **RA Classification**: Assign Actor/Boundary/Controller/Entity stereotypes
4. **Control Flow**: Apply UC-Methode Rules 1-5 for flow connections
5. **Data Flow**: Determine USE/PROVIDE relationships from prepositions
6. **Actor-Boundary**: Handle user interactions via HMI pattern

---

### 2. Domain Verb Loader
**File**: `src/domain_verb_loader.py` (641 lines)

**Purpose**: Load and merge domain-specific verb classifications.

**Architecture**:

```
┌─────────────────────────────────────────┐
│     DomainVerbLoader                    │
├─────────────────────────────────────────┤
│ Configurations:                         │
│  • common_domain.json (universal)      │
│  • beverage_preparation.json           │
│  • rocket_science.json                 │
│  • automotive.json                     │
│  • ...                                 │
├─────────────────────────────────────────┤
│ Verb Types:                            │
│  • Transaction Verbs (requests, outputs)│
│  • Transformation Verbs (grind, brew)  │
│  • Function Verbs (heat, press)        │
└─────────────────────────────────────────┘
```

**Key Methods**:

```python
def get_verb_classification(verb: str, domain: str) -> VerbType
def get_transformation_for_verb(verb: str, domain: str) -> str
def is_transaction_verb(verb: str, domain: str) -> bool
def detect_domain_from_text(text: str) -> str
```

**Domain Configuration Structure**:

```json
{
  "domain_name": "beverage_preparation",
  "keywords": ["coffee", "brewing", "espresso", "milk"],
  "verb_classification": {
    "transaction_verbs": {
      "verbs": {
        "request": "User requests product",
        "output": "System outputs information"
      }
    },
    "transformation_verbs": {
      "verbs": {
        "grind": "CoffeeBeans -> GroundCoffee",
        "brew": "GroundCoffee + HotWater -> Coffee"
      }
    },
    "function_verbs": {
      "verbs": {
        "heat": "Water -> HotWater",
        "pressurize": "Water -> PressurizedWater"
      }
    }
  },
  "aggregation_states": {
    "solid": {
      "specific_keywords": ["beans", "ground", "powder"],
      "specific_operations": ["grind", "crush"]
    },
    "liquid": {
      "specific_keywords": ["brew", "extraction", "pour"],
      "specific_operations": ["heat", "press", "pump"]
    }
  }
}
```

---

### 3. Material Controller Registry
**File**: `src/material_controller_registry.py` (284 lines)

**Purpose**: Manage shared material-based controllers with aggregation states.

**Architecture**:

```
MaterialControllerRegistry
├── WaterLiquidManager (water, liquid)
│   └── functions: {heat, press, pressurize, pump}
├── CoffeeSolidManager (coffee, solid)
│   └── functions: {grind, store}
├── CoffeeLiquidManager (coffee, liquid)
│   └── functions: {brew, extract, pour}
├── MilkLiquidManager (milk, liquid)
│   └── functions: {add, stop, steam}
├── SugarSolidManager (sugar, solid)
│   └── functions: {add, dispense}
├── FilterManager (filter, None)
│   └── functions: {prepare, insert, remove}
└── CupManager (cup, None)
    └── functions: {present, store, retrieve}
```

**Key Features**:
- **Shared Across UCs**: Same controller instance used by UC1 and UC2
- **Function Accumulation**: Controllers accumulate functions from all UCs
- **State-Aware Selection**: Selects controller based on material + aggregation state

**Core Methods**:

```python
def register_controller(controller: MaterialController) -> None
def find_controller_by_material(material: str) -> Optional[MaterialController]
def get_or_create_controller(material: str, state: str, function: str) -> str
def assign_function(controller_name: str, function: str) -> None
```

---

### 4. Generative Context Manager
**File**: `src/generative_context_manager.py` (530 lines)

**Purpose**: NLP-based context generation for operational materials.

**Context Types**:

```python
class ContextType(Enum):
    OPERATIONAL_MATERIAL = "operational_material"
    SAFETY_CONTEXT = "safety_context"
    HYGIENE_CONTEXT = "hygiene_context"
    FUNCTIONAL_CONTEXT = "functional_context"
    TECHNICAL_CONTEXT = "technical_context"
```

**Semantic Pattern Matching**:

```
Text: "The system heats water to 95°C"
      ↓
NLP Analysis (spaCy):
  - Verb: heat (VERB)
  - Object: water (NOUN)
  - Temperature: 95°C (QUANTITY)
      ↓
Domain Matching:
  - Material: water → WaterLiquidManager
  - Function: heat → temperature control
  - State: liquid → liquid aggregation state
      ↓
Generated Context:
  - Type: OPERATIONAL_MATERIAL
  - Safety Class: temperature_critical
  - Controllers: [WaterLiquidManager]
```

---

### 5. Visualization Layer

#### 5.1 SVG RUP Visualizer
**File**: `src/svg_rup_visualizer.py` (771 lines)

**Purpose**: Generate SVG diagrams with official Wikipedia RUP symbols.

**Features**:
- Official RUP symbols (Actor, Boundary, Controller, Entity)
- Parallel flow nodes (diamonds for P2_START, P2_END, etc.)
- Color-coded flows (blue=control, orange=use, green=provide)
- Data-driven layout using controller groups

**Layout Algorithm**:

```
1. Group controllers by parallel_group number
   - Group 0: Sequential (vertical stack)
   - Group 2: Parallel (P2 - horizontal row)
   - Group 3: Parallel (P3 - horizontal row)

2. Insert flow nodes between groups
   - P2_START before Group 2
   - P2_END after Group 2
   - P3_START before Group 3
   - P3_END after Group 3

3. Draw control flows (blue arrows)
4. Draw data flows (orange/green arrows)
```

**Example Output Structure**:

```xml
<svg width="1400" height="2000">
  <!-- Title -->
  <text>RUP Robustness Analysis - UC1</text>

  <!-- Actors (left column) -->
  <g><!-- User actor symbol --></g>
  <g><!-- Time actor symbol --></g>

  <!-- Controllers (center column) -->
  <g><!-- SystemControlManager --></g>
  <g><!-- P2_START diamond --></g>
  <g><!-- WaterLiquidManager --></g>
  <g><!-- FilterManager --></g>
  <g><!-- P2_END diamond --></g>

  <!-- Control flows (blue arrows) -->
  <path stroke="blue" marker-end="url(#arrowhead)"/>

  <!-- Data flows (orange/green arrows) -->
  <path stroke="orange" marker-end="url(#orangearrowhead)"/>
</svg>
```

#### 5.2 Pure RUP Visualizer
**File**: `src/pure_rup_visualizer.py` (459 lines)

**Purpose**: Generate high-resolution PNG diagrams using matplotlib.

**Features**:
- Pure RUP compliance (no extra decorations)
- Controller groups with parallel positioning
- Professional typography
- 300 DPI output for publications

---

## Data Flow Architecture

### Input Processing

```
Use Case File (UC1.txt)
      ↓
┌─────────────────────────────────┐
│ Section Detection               │
│  - Capability                   │
│  - Goal                         │
│  - Preconditions                │
│  - Basic Flow                   │
│  - Alternative/Extension Flows  │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ Line-by-Line Analysis           │
│  - Step ID extraction (B1, B2a) │
│  - NLP parsing (spaCy)          │
│  - Grammatical analysis         │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ RA Classification               │
│  - Verb → Controller            │
│  - Object → Entity              │
│  - User interaction → Boundary  │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ Control Flow Generation         │
│  - UC-Methode Rules 1-5         │
│  - Parallel flow nodes          │
│  - Actor-Boundary flows         │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ Data Flow Analysis              │
│  - Preposition-based (with/to)  │
│  - USE relationships            │
│  - PROVIDE relationships        │
└─────────────────────────────────┘
      ↓
Output (JSON, CSV, SVG, PNG)
```

---

## Analysis Pipeline

> **📚 Detailed Documentation**: For a complete step-by-step breakdown of the analysis pipeline with all UC-Methode rules, detailed algorithms, and comprehensive examples, see **[ARCHITECTURE_DETAILED_PIPELINE.md](ARCHITECTURE_DETAILED_PIPELINE.md)** (2672 lines, 73KB).

This section provides a **high-level overview** of the 7-phase analysis pipeline. Each phase applies specific UC-Methode rules and processes UC text through multiple stages.

---

### Overview of Analysis Phases

```
Phase 1: Context Analysis
  ↓ UC-Methode Context Rules
Phase 2: Resource Analysis (Betriebsmittel)
  ↓ UC-Methode Resource Rules
Phase 3: Interaction Analysis
  ↓ UC-Methode Interaction Rules (Transaction/Transformation/Function verbs)
Phase 4: Control Flow Generation
  ↓ UC-Methode Rules 1-5 (Serial/Parallel flows)
Phase 5: Data Flow Analysis
  ↓ UC-Methode Data Flow Rules (USE/PROVIDE)
Phase 6: Actor-Boundary Flows
  ↓ UC-Methode Actor Rules (HMI pattern)
Phase 7: RA Classification & Validation
  ↓ UC-Methode Validation Rules
```

---

### Phase 1: Context Analysis

**UC-Methode Rules**:
- **Context Rule 1**: Every UC must have a clearly defined capability and goal
- **Context Rule 2**: Domain must be identified before analysis
- **Context Rule 3**: Preconditions define the operational materials (Betriebsmittel)

**Input**: UC file header (Capability, Goal, Preconditions)

**Detailed Steps**:
1. **Extract UC Header** - Parse capability, goal, preconditions from UC file
2. **Domain Detection** - Match keywords to domain configurations
3. **Load Domain Config** - Load verb classifications, materials, states
4. **Initialize NLP** - Load spaCy model (en_core_web_md)

**Output**: `UCContext` object

```python
@dataclass
class UCContext:
    capability: str
    goal: str
    domain: str
    actors: List[str]
    preconditions: List[str]
```

**Example**:
```
Input UC Header:
  Capability: Coffee Preparation
  Goal: User can drink their milk coffee every morning at 7am
  Domain: beverage_preparation (auto-detected)
```

> 📖 **See**: ARCHITECTURE_DETAILED_PIPELINE.md - Phase 1 for complete algorithms and 4 detailed sub-steps

---

### Phase 2: Resource Analysis (Betriebsmittel)

**UC-Methode Rules**:
- **Resource Rule 1**: All preconditions must be analyzed for operational materials
- **Resource Rule 2**: Each material requires a Supply Boundary
- **Resource Rule 3**: Safety and hygiene requirements must be checked
- **Resource Rule 4**: Material entities must be created for the system state

**Input**: Precondition lines

**Detailed Steps**:
1. **Extract Materials** - Identify material types (water, coffee, milk, sugar)
2. **Generate Supply Boundaries** - Create {Material}SupplyBoundary for each
3. **Check Safety Requirements** - Load temperature/pressure limits from domain JSON
4. **Create Material Entities** - Generate operational material entities

**Example**:

```
Precondition: "Water is available in the system"
      ↓
Steps:
  1. Extract material: "water"
  2. Create boundary: WaterSupplyBoundary
  3. Check safety: temperature_limits (max: 100°C), pressure_limits (max: 15 bar)
  4. Create entity: Water
```

> 📖 **See**: ARCHITECTURE_DETAILED_PIPELINE.md - Phase 2 for complete algorithms and 4 detailed sub-steps

---

### Phase 3: Interaction Analysis

**UC-Methode Rules**:
- **Interaction Rule 1**: Each verb must be classified as Transaction, Transformation, or Function verb
- **Interaction Rule 2**: Transaction verbs create Boundaries for user-system interactions
- **Interaction Rule 3**: Transformation verbs create Controllers + output Entities based on domain transformations
- **Interaction Rule 4**: Function verbs assign functions to existing material-based Controllers
- **Interaction Rule 5**: Controller selection must consider material base name AND aggregation state

**Input**: Basic Flow steps (B1, B2a, B2b, etc.)

**Detailed Steps**:
1. **NLP Grammatical Analysis** - Extract verbs, direct objects, prepositions using spaCy
2. **Verb Classification** - Determine verb type from domain configuration (Transaction/Transformation/Function)
3. **Transaction Verb Processing** - Create Boundaries for user interactions
4. **Transformation Verb Processing** - Create Controllers and output Entities based on material transformations
5. **Function Verb Processing** - Assign functions to existing material Controllers
6. **Aggregation State Determination** - Determine solid/liquid/gas state from keywords and operations

**Example**:

```
Step B2a: "System grinds coffee beans"
      ↓
NLP: verb="grind", object="coffee beans"
Verb Classification: transformation_verb (domain: "grind": "CoffeeBeans -> GroundCoffee")
Extract Output: "GroundCoffee"
Determine State: "solid" (keyword: "ground")
Material Base: "coffee"
Controller: CoffeeSolidManager (coffee + solid)
Function: grind()
Output Entity: GroundCoffee
```

> 📖 **See**: ARCHITECTURE_DETAILED_PIPELINE.md - Phase 3 for complete algorithms and 6 detailed sub-steps

---

### Phase 4: Control Flow Generation

**UC-Methode Rules**:
- **Rule 1 (Serial → Serial)**: Direct connection between consecutive steps
- **Rule 2 (Serial → Parallel)**: Insert distribution node (PX_START) before parallel group
- **Rule 3 (Parallel → Parallel, Same Group)**: All parallel steps in same group connect through same START/END nodes
- **Rule 4 (Parallel → Parallel, Different Group)**: Connect END node of first group to START node of next group
- **Rule 5 (Parallel → Serial)**: Insert merge node (PX_END) after parallel group before next serial step

**Input**: Controllers from Phase 3 with step IDs (B1, B2a, B2b, etc.)

**Detailed Steps**:
1. **Detect Parallel Groups** - Identify groups from step ID patterns (B2a, B2b → Group 2)
2. **Assign Parallel Group Numbers** - Mark controllers with group number and position
3. **Apply Rule 1** - Connect consecutive serial steps directly
4. **Apply Rules 2-3** - Insert PX_START, connect all parallel steps, insert PX_END
5. **Apply Rule 4** - Connect between different parallel groups (P2_END → P3_START)
6. **Apply Rule 5** - Connect parallel group end to next serial step
7. **Validate Flow Connectivity** - Ensure no orphaned nodes

**Example**:

```
Steps: B1, B2a, B2b, B2c, B2d, B3a, B3b, B4
      ↓
Parallel Group Detection:
  - B1: Sequential (group 0)
  - B2a, B2b, B2c, B2d: Group 2
  - B3a, B3b: Group 3
  - B4: Sequential (group 0)
      ↓
Control Flow:
  B1 → P2_START (Rule 2)
  P2_START → B2a → P2_END (Rule 3)
  P2_START → B2b → P2_END (Rule 3)
  P2_START → B2c → P2_END (Rule 3)
  P2_START → B2d → P2_END (Rule 3)
  P2_END → P3_START (Rule 4)
  P3_START → B3a → P3_END (Rule 3)
  P3_START → B3b → P3_END (Rule 3)
  P3_END → B4 (Rule 5)
```

> 📖 **See**: ARCHITECTURE_DETAILED_PIPELINE.md - Phase 4 for complete algorithms and 7 detailed sub-steps

---

### Phase 5: Data Flow Analysis

**UC-Methode Rules**:
- **Data Flow Rule 1**: Prepositions "with", "from", "using" indicate USE relationships (Entity → Controller)
- **Data Flow Rule 2**: Prepositions "to", "for" in output context indicate PROVIDE relationships (Controller → Entity)
- **Data Flow Rule 3**: Transformation verbs create USE relationships for inputs and PROVIDE for outputs
- **Data Flow Rule 4**: Direct objects without prepositions create PROVIDE relationships

**Input**: Control flows with NLP-analyzed step text containing prepositions and objects

**Detailed Steps**:
1. **Extract Preposition Phrases** - Identify prepositional objects from spaCy parse
2. **Classify Preposition Type** - Determine if USE (input) or PROVIDE (output) based on preposition
3. **Create USE Relationships** - Connect input entities to controllers (with, from, using)
4. **Create PROVIDE Relationships** - Connect controllers to output entities (to, for, direct objects)
5. **Apply Transformation Patterns** - Use domain JSON transformations for input/output entities

**Example**:

```
Step B2a: "System grinds coffee beans with grinding degree"
      ↓
NLP Parse:
  - Verb: "grind"
  - Direct Object: "coffee beans"
  - Prepositional Phrase: "with grinding degree"
      ↓
Transformation: "grind": "CoffeeBeans -> GroundCoffee"
Controller: CoffeeSolidManager
      ↓
Data Flows:
  - USE: CoffeeBeans → CoffeeSolidManager (transformation input)
  - USE: GrindingDegree → CoffeeSolidManager (prepositional object "with")
  - PROVIDE: CoffeeSolidManager → GroundCoffee (transformation output)
```

> 📖 **See**: ARCHITECTURE_DETAILED_PIPELINE.md - Phase 5 for complete algorithms and 5 detailed sub-steps

---

### Phase 6: Actor-Boundary Flows

**UC-Methode Rules**:
- **Actor Rule 1**: User interactions must route through HMI pattern (User → Boundary → HMIManager → Controller)
- **Actor Rule 2**: Extension flows (E1, E2) with user triggers must create HMI boundaries
- **Actor Rule 3**: Alternative flows (A1, A2) with user triggers must create HMI boundaries
- **Actor Rule 4**: Time-based triggers create direct Time → Boundary flows (no HMI intermediary)

**Input**: Detected actor interactions from Basic Flow, Extension Flows, Alternative Flows

**Detailed Steps**:
1. **Detect User Interactions** - Identify steps starting with "User" or user-related triggers
2. **Create HMI Boundaries** - Generate purpose-specific boundaries (e.g., SugarRequestBoundary)
3. **Apply HMI Pattern** - Connect User → Boundary → HMIManager → Target Controller
4. **Handle Extension/Alternative Triggers** - Parse triggers like "E1 B4-B5 (trigger) User wants sugar"
5. **Create Time-Based Flows** - Connect Time actor directly to time-trigger boundaries

**Example**:

```
Extension Flow: "E1 B4-B5 (trigger) User wants sugar"
      ↓
Detection:
  - Flow type: Extension (E1)
  - Trigger range: B4-B5
  - Text: "User wants sugar"
  - Actor: User
      ↓
HMI Pattern:
  1. Create Boundary: SugarRequestBoundary
  2. Create Flow: User → SugarRequestBoundary
  3. Create Flow: SugarRequestBoundary → HMIManager
  4. Create Flow: HMIManager → SugarSolidManager (target controller)
```

> 📖 **See**: ARCHITECTURE_DETAILED_PIPELINE.md - Phase 6 for complete algorithms and 5 detailed sub-steps

---

### Phase 7: RA Classification & Validation

**UC-Methode Rules**:
- **Validation Rule 1**: All Boundaries must connect to at least one Controller
- **Validation Rule 2**: All Controllers must have at least one assigned function
- **Validation Rule 3**: Parallel flow groups must have both PX_START and PX_END nodes
- **Validation Rule 4**: No direct Actor → Controller connections (must route through Boundary)
- **Validation Rule 5**: Material Controllers must specify aggregation state if material has multiple states

**Input**: Complete RA class set from all previous phases

**Detailed Steps**:
1. **Validate Boundary Connections** - Ensure all boundaries have outgoing flows to controllers
2. **Validate Controller Functions** - Ensure all controllers have assigned functions from verbs
3. **Validate Parallel Flow Nodes** - Check START/END pairs for each parallel group
4. **Validate Actor Patterns** - Ensure no direct Actor → Controller violations
5. **Generate Warnings** - Flag implementation elements, missing aggregation states, orphaned components

**Example Validation**:

```
Components:
  - Boundaries: [WaterSupplyBoundary, SugarRequestBoundary, ...]
  - Controllers: [WaterLiquidManager, SugarSolidManager, HMIManager, ...]
  - Flow Nodes: [P2_START, P2_END, P3_START, P3_END]
      ↓
Validation Checks:
  ✅ All boundaries connected to controllers
  ✅ All controllers have functions
  ✅ P2_START/P2_END pair exists
  ✅ P3_START/P3_END pair exists
  ✅ No direct Actor → Controller connections
  ✅ Material controllers have aggregation states
      ↓
Warnings:
  ⚠️ "Filter" detected as implementation element (should be abstracted)
```

> 📖 **See**: ARCHITECTURE_DETAILED_PIPELINE.md - Phase 7 for complete algorithms and 5 detailed sub-steps

---

## Domain-Driven Design

### Domain JSON Structure

```json
{
  "domain_name": "beverage_preparation",
  "description": "Coffee and beverage preparation domain",

  "keywords": ["coffee", "espresso", "brewing", "milk", "tea"],

  "materials": {
    "coffee": {
      "base_name": "coffee",
      "variants": ["coffee beans", "ground coffee", "brewed coffee"],
      "aggregation_states": ["solid", "liquid"],
      "transformations": {
        "solid_to_liquid": {
          "input": "GroundCoffee + HotWater",
          "process": "brew",
          "output": "Coffee"
        }
      }
    }
  },

  "verb_classification": {
    "transaction_verbs": {
      "description": "User-System interactions",
      "verbs": {
        "request": "User requests a product or service",
        "output": "System provides information to user"
      }
    },
    "transformation_verbs": {
      "description": "Material transformations",
      "verbs": {
        "grind": "CoffeeBeans -> GroundCoffee",
        "brew": "GroundCoffee + HotWater -> Coffee"
      }
    },
    "function_verbs": {
      "description": "Material functions without transformation",
      "verbs": {
        "heat": "Water -> HotWater (temperature change)",
        "pressurize": "Water -> PressurizedWater (state change)"
      }
    }
  },

  "aggregation_states": {
    "solid": {
      "specific_keywords": ["beans", "ground", "powder", "crystals"],
      "specific_operations": ["grind", "crush", "mill"],
      "general_keywords": ["solid", "granular", "particulate"]
    },
    "liquid": {
      "specific_keywords": ["brew", "brewing", "extraction", "pour"],
      "specific_operations": ["heat", "press", "pump", "pour"],
      "general_keywords": ["liquid", "fluid", "beverage"]
    },
    "gas": {
      "specific_keywords": ["steam", "vapor", "pressure"],
      "specific_operations": ["vaporize", "steam", "pressurize"],
      "general_keywords": ["gas", "vapor", "steam"]
    }
  },

  "safety_requirements": {
    "water": {
      "temperature_limits": {
        "max": 100,
        "unit": "celsius",
        "critical": true
      },
      "pressure_limits": {
        "max": 15,
        "unit": "bar",
        "critical": true
      }
    }
  },

  "hygiene_requirements": {
    "coffee": {
      "cleaning_cycle": {
        "frequency": "after_each_brew",
        "agents": ["cleaning_solution", "water"],
        "critical": false
      }
    }
  }
}
```

### Domain Detection

```python
def detect_domain(uc_text: str) -> str:
    """
    Detect domain from UC text using keyword matching

    Priority:
    1. Specific domain keywords (higher weight)
    2. Material types
    3. Common verbs
    """
    scores = {}

    for domain_name, config in domain_configs.items():
        score = 0
        keywords = config.get('keywords', [])

        for keyword in keywords:
            if keyword.lower() in uc_text.lower():
                score += 1

        scores[domain_name] = score

    return max(scores, key=scores.get)
```

---

## Multi-UC Architecture

### Shared Component Registry

**Problem**: UC1 and UC2 both use WaterLiquidManager, but with different functions.

**Solution**: Single shared `MaterialControllerRegistry` across all UCs.

```
UC1 Analysis:
  WaterLiquidManager.heat()     ← Function added

UC2 Analysis (same registry):
  WaterLiquidManager.pressurize() ← Function added

Final WaterLiquidManager:
  functions: {heat, pressurize}  ← Both functions!
```

**Implementation**:

```python
class StructuredUCAnalyzer:
    def __init__(self, domain_name: str, shared_registry: MaterialControllerRegistry = None):
        if shared_registry:
            self.material_registry = shared_registry  # Reuse registry
        else:
            self.material_registry = MaterialControllerRegistry()  # New registry

# Multi-UC Analysis
def analyze_from_config(config_file: str):
    config = load_config(config_file)

    # Create ONE shared registry
    shared_registry = MaterialControllerRegistry()

    for uc_config in config['use_cases']:
        # Pass shared registry to each analyzer
        analyzer = StructuredUCAnalyzer(
            domain_name=config['domain'],
            shared_registry=shared_registry  # ← Same instance!
        )
        analyzer.analyze_uc_file(uc_config['file'])
```

---

### Combined Diagram Generation

**Process**:

```
1. Load UC1 JSON analysis
2. Load UC2 JSON analysis
3. Merge components:
   - Identify shared controllers (same name)
   - Deduplicate boundaries and entities
   - Merge control flows
   - Merge data flows
   - Combine parallel flow nodes
4. Generate combined SVG
```

**Merge Algorithm**:

```python
def merge_uc_analyses(uc1_data: dict, uc2_data: dict) -> dict:
    # Classify controllers
    uc1_controllers = {c['name']: c for c in uc1_data['components']['controllers']}
    uc2_controllers = {c['name']: c for c in uc2_data['components']['controllers']}

    shared = set(uc1_controllers.keys()) & set(uc2_controllers.keys())
    uc1_only = set(uc1_controllers.keys()) - set(uc2_controllers.keys())
    uc2_only = set(uc2_controllers.keys()) - set(uc1_controllers.keys())

    # Annotate shared controllers
    for name in shared:
        controller = uc1_controllers[name]
        controller['shared'] = True
        controller['use_cases'] = ['UC1', 'UC2']

        # Merge function descriptions
        uc2_desc = uc2_controllers[name]['description']
        if uc2_desc != controller['description']:
            controller['description'] += f" | UC2: {uc2_desc}"

    # Merge all components
    combined = {
        'meta': {...},
        'components': {
            'actors': deduplicate(uc1['actors'] + uc2['actors']),
            'controllers': merge_controllers(uc1_controllers, uc2_controllers),
            'boundaries': deduplicate(uc1['boundaries'] + uc2['boundaries']),
            'entities': deduplicate(uc1['entities'] + uc2['entities']),
            'control_flow_nodes': deduplicate(uc1['flow_nodes'] + uc2['flow_nodes'])
        },
        'relationships': {
            'control_flows': uc1['control_flows'] + uc2['control_flows'],
            'data_flows': uc1['data_flows'] + uc2['data_flows']
        }
    }

    return combined
```

---

## Visualization Layer

### RUP Symbol Library

**Location**: `svg/Robustness_Diagram_*.svg`

**Symbols**:
- `Robustness_Diagram_Actor.svg` - Stick figure
- `Robustness_Diagram_Boundary.svg` - Circle with line
- `Robustness_Diagram_Control.svg` - Circle with arrow
- `Robustness_Diagram_Entity.svg` - Circle

**Usage in SVG**:

```python
def load_rup_symbol(symbol_type: str) -> str:
    """Load official Wikipedia RUP symbol"""
    symbol_path = f"svg/Robustness_Diagram_{symbol_type}.svg"
    with open(symbol_path, 'r', encoding='utf-8') as f:
        return f.read()

def insert_symbol(svg_content: str, x: int, y: int, scale: float) -> str:
    """Insert symbol at position with scaling"""
    return f'<g transform="translate({x}, {y}) scale({scale})">{svg_content}</g>'
```

---

### Layout Engine

**Coordinate System**:

```
(0, 0) ──────────────────────────→ X (1400px)
│
│  Actors    Controllers    Entities
│   (84)       (812)         (1240)
│
│    👤         ⚙️            ⭕
│    User      System        Data
│
│
↓ Y (2000px)
```

**Layout Algorithm**:

```python
def calculate_layout(components: Dict) -> Dict[str, Tuple[int, int]]:
    """Calculate positions for all components"""

    positions = {}
    y_offset = 80  # Start Y

    # Actors (left column)
    for actor in components['actors']:
        positions[actor['name']] = (84, y_offset)
        y_offset += 70

    # Reset Y for controllers
    y_offset = 80

    # Sequential controllers (center, vertical)
    for controller in get_sequential_controllers(components):
        positions[controller['name']] = (812, y_offset)
        y_offset += 122

    # Parallel flow START node
    positions[f'P{group}_START'] = (812, y_offset)
    y_offset += 100

    # Parallel controllers (horizontal row)
    x_offset = 662
    for controller in get_parallel_controllers(components, group):
        positions[controller['name']] = (x_offset, y_offset)
        x_offset += 100

    y_offset += 122

    # Parallel flow END node
    positions[f'P{group}_END'] = (812, y_offset)
    y_offset += 100

    return positions
```

---

## Extension Points

### 1. New Domain Support

**Steps**:
1. Create `domains/new_domain.json`
2. Define keywords, materials, verbs
3. Add aggregation state rules
4. Test with sample UC

**Template**:

```json
{
  "domain_name": "new_domain",
  "keywords": ["domain", "specific", "terms"],
  "materials": {
    "material_name": {
      "base_name": "material",
      "variants": ["variant1", "variant2"],
      "aggregation_states": ["solid", "liquid"]
    }
  },
  "verb_classification": {
    "transaction_verbs": {"verbs": {}},
    "transformation_verbs": {"verbs": {}},
    "function_verbs": {"verbs": {}}
  }
}
```

### 2. New Analysis Rules

**Location**: `src/structured_uc_analyzer.py`

**Extension Points**:

```python
class StructuredUCAnalyzer:

    def _generate_control_flows(self):
        """Extend with new UC-Methode rules"""
        # Rule 1-5 implementation
        # Add Rule 6, Rule 7, etc. here
        pass

    def _analyze_data_flows(self):
        """Extend with new preposition patterns"""
        # Current: "with", "from", "to", "for"
        # Add: "by", "using", "via", etc.
        pass
```

### 3. New Output Formats

**Current**: JSON, CSV, SVG, PNG

**Extension**:

```python
class OutputGenerator:
    def generate_graphml(self, analysis_data: Dict) -> str:
        """Generate GraphML for yEd/Gephi"""
        pass

    def generate_plantuml(self, analysis_data: Dict) -> str:
        """Generate PlantUML diagram"""
        pass

    def generate_html(self, analysis_data: Dict) -> str:
        """Generate interactive HTML report"""
        pass
```

### 4. Custom Visualizers

**Interface**:

```python
class BaseVisualizer(ABC):
    @abstractmethod
    def generate_diagram(self, json_path: str, output_path: str):
        """Generate diagram from JSON analysis"""
        pass

    @abstractmethod
    def render_component(self, component: RAClass) -> Any:
        """Render single RA component"""
        pass

    @abstractmethod
    def render_flow(self, flow: ControlFlow) -> Any:
        """Render control/data flow"""
        pass
```

**Implementation**:

```python
class CustomVisualizer(BaseVisualizer):
    def generate_diagram(self, json_path: str, output_path: str):
        # Custom rendering logic
        pass
```

---

## Configuration Reference

### Analysis Configuration

**File**: `uc_analysis_config.json`

```json
{
  "analysis_name": "Multi-UC Analysis",
  "description": "Combined analysis of multiple use cases",
  "domain": "beverage_preparation",

  "use_cases": [
    {
      "id": "UC1",
      "file": "Use Case/UC1.txt",
      "name": "Prepare Milk Coffee",
      "enabled": true
    },
    {
      "id": "UC2",
      "file": "Use Case/UC2.txt",
      "name": "Prepare Espresso",
      "enabled": true
    }
  ],

  "analysis_mode": "combined",

  "output": {
    "directory": "new/multi_uc",
    "formats": ["json", "csv", "svg", "png"],
    "diagram_name": "UC1_UC2_Combined_RA_Diagram"
  },

  "options": {
    "share_controllers": true,
    "show_controller_functions": true,
    "merge_duplicate_entities": true,
    "generate_combined_diagram": true
  }
}
```

---

## Usage Examples

### Single UC Analysis

```bash
# Command-line
python src/structured_uc_analyzer.py --uc-file "Use Case/UC1.txt" --domain beverage_preparation

# Output:
#  - new/UC1_Structured_RA_Analysis.json
#  - new/UC1_Structured_UC_Steps_RA_Classes.csv
#  - new/UC1_Structured_SVG_RA_Diagram.svg
#  - new/UC1_Structured_Pure_RA_Diagram.png
```

### Multi-UC Analysis

```bash
# Config-based
python src/structured_uc_analyzer.py --config uc_analysis_config.json

# Output:
#  - new/multi_uc/UC1_Structured_RA_Analysis.json
#  - new/multi_uc/UC2_Structured_RA_Analysis.json
#  - new/multi_uc/multi_uc_analysis_summary.json
#  - new/multi_uc/*.svg, *.png, *.csv
```

### Combined Diagram

```bash
# Generate combined UC1+UC2 diagram
python generate_uc1_uc2_combined_svg.py

# Output:
#  - new/multi_uc/UC1_UC2_Combined_SVG_RA_Analysis.json
#  - new/multi_uc/UC1_UC2_Combined_SVG_RA_Diagram.svg
```

---

## Performance Considerations

### NLP Model Loading

```python
# Load once, reuse across UCs
nlp = spacy.load("en_core_web_md")  # ~100MB, 1-2 seconds

# NOT: Load for each UC (slow!)
```

### Material Registry

```python
# Shared registry pattern (good)
registry = MaterialControllerRegistry()
analyzer1 = StructuredUCAnalyzer(shared_registry=registry)
analyzer2 = StructuredUCAnalyzer(shared_registry=registry)

# NOT: New registry per UC (memory waste)
```

### JSON Caching

```python
# Cache domain configurations
domain_configs = {}  # Load once, reuse
```

---

## Quality Assurance

### Test Coverage

```
tests/
├── test_control_flow.py           # UC-Methode Rules 1-5
├── test_controller_enhancement.py # Material controllers
├── test_domain_autodetection.py   # Domain detection
├── test_domain_orchestrator.py    # Multi-domain
├── test_hmi_integration.py        # User interactions
├── test_multi_uc_analysis.py      # Multi-UC scenarios
├── test_precondition_analysis.py  # Operational materials
├── test_safety_hygiene_analysis.py # Safety/hygiene
└── test_uc1_uc2.py                # UC1+UC2 integration
```

### Validation Rules

1. **All Boundaries Must Connect to Controllers**
2. **All Controllers Must Have Functions**
3. **Parallel Flows Must Have START and END Nodes**
4. **Extension Triggers (E1, A1) Must Route Through HMI**
5. **Material Controllers Must Have Aggregation State (if applicable)**

---

## Future Enhancements

### Roadmap

1. **Real-time Analysis**
   - Watch UC files for changes
   - Auto-regenerate diagrams

2. **Interactive Web UI**
   - Drag-and-drop UC upload
   - Live diagram preview
   - Interactive editing

3. **AI-Assisted UC Generation**
   - Natural language to UC conversion
   - Suggestion engine for steps

4. **Version Control Integration**
   - Git hooks for auto-analysis
   - Diff visualization

5. **Domain Marketplace**
   - Community-contributed domains
   - Domain validation toolkit

---

## References

### UC-Methode
- **Control Flow Rules 1-5**: Sequential, parallel, and hybrid flow patterns
- **RUP Compliance**: Official Wikipedia Robustness Analysis symbols

### Standards
- **UML 2.5**: Use Case Diagrams
- **ISO/IEC 19505**: Unified Modeling Language specification

### Tools
- **spaCy**: Industrial-strength NLP (https://spacy.io)
- **Matplotlib**: Publication-quality diagrams
- **SVG**: Scalable Vector Graphics standard

---

## Contact & Support

**Repository**: https://github.com/DieterWagner42/RA-NLF
**Documentation**: See CLAUDE.md for development guidelines
**License**: See LICENSE file

---

**End of Architecture Documentation**
