# RA-NLF Detailed Analysis Pipeline
## Complete Step-by-Step UC-Methode Implementation

**Version**: 2.0
**Last Updated**: 2025-11-14

---

## Table of Contents

1. [Phase 1: Context Analysis](#phase-1-context-analysis)
2. [Phase 2: Resource Analysis](#phase-2-resource-analysis)
3. [Phase 3: Interaction Analysis](#phase-3-interaction-analysis)
4. [Phase 4: Control Flow Generation](#phase-4-control-flow-generation)
5. [Phase 5: Data Flow Analysis](#phase-5-data-flow-analysis)
6. [Phase 6: Actor-Boundary Flows](#phase-6-actor-boundary-flows)
7. [Phase 7: RA Classification & Validation](#phase-7-ra-classification--validation)

---

## Phase 1: Context Analysis

### UC-Methode Rules Applied

**Context Rule 1**: Every UC must have a clearly defined capability and goal
**Context Rule 2**: Domain must be identified before analysis
**Context Rule 3**: Preconditions define the operational materials (Betriebsmittel)

### Input

```
Use Case File: Use Case/UC1.txt

Capability: Coffee Preparation
Goal: User can drink their milk coffee every morning at 7am
Preconditions:
  - Coffee beans are available in the system
  - Water is available in the system
  - Milk is available in the system
  - Sugar is available in the system
```

### Detailed Steps

#### Step 1.1: Extract UC Header Information

**Algorithm**:
```python
def extract_context(uc_lines: List[str]) -> UCContext:
    context = UCContext()

    for line in uc_lines:
        if line.startswith("Capability:"):
            context.capability = line.replace("Capability:", "").strip()
        elif line.startswith("Goal:"):
            context.goal = line.replace("Goal:", "").strip()
        elif line.startswith("Preconditions:"):
            in_preconditions = True
        elif in_preconditions and line.startswith("  -"):
            context.preconditions.append(line.strip("  - "))

    return context
```

**Example**:
```
Input:  "Capability: Coffee Preparation"
Output: UCContext(capability="Coffee Preparation")

Input:  "Goal: User can drink their milk coffee every morning at 7am"
Output: UCContext(goal="User can drink their milk coffee...")
```

#### Step 1.2: Domain Detection

**Algorithm**:
```python
def detect_domain(uc_text: str, domain_configs: Dict) -> str:
    """
    Detect domain using keyword matching

    Priority:
    1. Domain-specific keywords (coffee, rocket, car)
    2. Material names
    3. Verb types
    """
    scores = {}

    for domain_name, config in domain_configs.items():
        score = 0

        # Check domain keywords
        for keyword in config.get('keywords', []):
            if keyword.lower() in uc_text.lower():
                score += 2  # Higher weight

        # Check material names
        for material in config.get('materials', {}).keys():
            if material.lower() in uc_text.lower():
                score += 1

        scores[domain_name] = score

    return max(scores, key=scores.get) if scores else DEFAULT_DOMAIN
```

**Example**:
```
UC Text: "Capability: Coffee Preparation... coffee beans... water... milk..."

Keyword Matching:
  beverage_preparation: "coffee" (x3) + "milk" (x1) = 8 points
  rocket_science: "fuel" (x0) + "oxygen" (x0) = 0 points
  automotive: "gasoline" (x0) + "oil" (x0) = 0 points

Detected Domain: beverage_preparation
```

#### Step 1.3: Load Domain Configuration

**Algorithm**:
```python
def load_domain_config(domain_name: str) -> Dict:
    """Load domain JSON configuration"""
    domain_path = f"domains/{domain_name}.json"

    with open(domain_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config
```

**Loaded Configuration**:
```json
{
  "domain_name": "beverage_preparation",
  "verb_classification": {
    "transaction_verbs": {...},
    "transformation_verbs": {
      "grind": "CoffeeBeans -> GroundCoffee",
      "brew": "GroundCoffee + HotWater -> Coffee"
    },
    "function_verbs": {
      "heat": "Water -> HotWater"
    }
  },
  "aggregation_states": {
    "solid": {...},
    "liquid": {...}
  }
}
```

#### Step 1.4: Initialize NLP Engine

**Algorithm**:
```python
def initialize_nlp() -> spacy.Language:
    """Load spaCy model for NLP analysis"""
    try:
        nlp = spacy.load("en_core_web_md")
        return nlp
    except OSError:
        print("ERROR: spaCy model not found. Run: python -m spacy download en_core_web_md")
        sys.exit(1)
```

### Output

```python
UCContext(
    capability="Coffee Preparation",
    goal="User can drink their milk coffee every morning at 7am",
    domain="beverage_preparation",
    actors=["User", "Time"],
    preconditions=[
        "Coffee beans are available in the system",
        "Water is available in the system",
        "Milk is available in the system",
        "Sugar is available in the system"
    ]
)
```

---

## Phase 2: Resource Analysis (Betriebsmittel)

### UC-Methode Rules Applied

**Resource Rule 1**: All preconditions must be analyzed for operational materials
**Resource Rule 2**: Each material requires a Supply Boundary
**Resource Rule 3**: Safety and hygiene requirements must be checked
**Resource Rule 4**: Material entities must be created for the system state

### Input

```
Preconditions:
  - Coffee beans are available in the system
  - Water is available in the system
  - Milk is available in the system
  - Sugar is available in the system
```

### Detailed Steps

#### Step 2.1: Extract Materials from Preconditions

**Algorithm**:
```python
def extract_materials_from_preconditions(preconditions: List[str], domain_config: Dict) -> List[Material]:
    """
    Extract material names from precondition text
    """
    materials = []
    material_types = domain_config.get('materials', {})

    for precondition in preconditions:
        precond_lower = precondition.lower()

        # Check each known material type
        for material_name, material_info in material_types.items():
            # Check base name
            if material_name.lower() in precond_lower:
                materials.append(Material(
                    name=material_name.title(),
                    variants=material_info.get('variants', []),
                    states=material_info.get('aggregation_states', [])
                ))
                continue

            # Check variants
            for variant in material_info.get('variants', []):
                if variant.lower() in precond_lower:
                    materials.append(Material(
                        name=material_name.title(),
                        variants=material_info.get('variants', []),
                        states=material_info.get('aggregation_states', [])
                    ))
                    break

    return materials
```

**Example**:
```
Input:  "Coffee beans are available in the system"

Matching:
  - "coffee" found in material_types
  - Variant "coffee beans" matches exactly

Output: Material(
    name="Coffee",
    variants=["coffee beans", "ground coffee", "brewed coffee"],
    states=["solid", "liquid"]
)
```

#### Step 2.2: Generate Supply Boundaries

**UC-Methode Rule**: Each operational material requires a boundary for supply

**Algorithm**:
```python
def generate_supply_boundaries(materials: List[Material]) -> List[RAClass]:
    """
    Generate Supply Boundary for each material

    Naming Convention: {Material}SupplyBoundary
    """
    boundaries = []

    for material in materials:
        boundary = RAClass(
            name=f"{material.name}SupplyBoundary",
            ra_type=RAType.BOUNDARY,
            stereotype="<<boundary>>",
            description=f"Boundary for {material.name.lower()} supply from preconditions",
            step_id="Preconditions",
            element_type="supply"
        )
        boundaries.append(boundary)

    return boundaries
```

**Example**:
```
Material: Coffee
  ↓
Boundary: CoffeeSupplyBoundary
  - ra_type: BOUNDARY
  - stereotype: <<boundary>>
  - description: "Boundary for coffee supply from preconditions"
  - step_id: "Preconditions"

Material: Water
  ↓
Boundary: WaterSupplyBoundary
  - ra_type: BOUNDARY
  - stereotype: <<boundary>>
  - description: "Boundary for water supply from preconditions"
  - step_id: "Preconditions"
```

#### Step 2.3: Check Safety Requirements

**Algorithm**:
```python
def check_safety_requirements(material: Material, domain_config: Dict) -> Dict:
    """
    Check if material has safety requirements
    """
    safety_reqs = domain_config.get('safety_requirements', {})
    material_safety = safety_reqs.get(material.name.lower(), {})

    if not material_safety:
        return {}

    requirements = {}

    # Temperature limits
    if 'temperature_limits' in material_safety:
        temp = material_safety['temperature_limits']
        requirements['temperature'] = {
            'max': temp.get('max'),
            'unit': temp.get('unit'),
            'critical': temp.get('critical', False)
        }

    # Pressure limits
    if 'pressure_limits' in material_safety:
        pressure = material_safety['pressure_limits']
        requirements['pressure'] = {
            'max': pressure.get('max'),
            'unit': pressure.get('unit'),
            'critical': pressure.get('critical', False)
        }

    return requirements
```

**Example**:
```
Material: Water

Domain JSON Safety Requirements:
{
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
}

Output:
{
  'temperature': {'max': 100, 'unit': 'celsius', 'critical': True},
  'pressure': {'max': 15, 'unit': 'bar', 'critical': True}
}
```

#### Step 2.4: Create Material Entities

**UC-Methode Rule**: Materials in preconditions become entities in the system state

**Algorithm**:
```python
def create_material_entities(materials: List[Material]) -> List[RAClass]:
    """
    Create Entity for each material from preconditions
    """
    entities = []

    for material in materials:
        entity = RAClass(
            name=material.name,
            ra_type=RAType.ENTITY,
            stereotype="<<entity>>",
            description=f"{material.name} available in system (precondition)",
            step_id="Preconditions",
            element_type="operational_material"
        )
        entities.append(entity)

    return entities
```

**Example**:
```
Materials from Preconditions:
  1. Coffee beans
  2. Water
  3. Milk
  4. Sugar

Generated Entities:
  1. Coffee (<<entity>>) - "Coffee available in system (precondition)"
  2. Water (<<entity>>) - "Water available in system (precondition)"
  3. Milk (<<entity>>) - "Milk available in system (precondition)"
  4. Sugar (<<entity>>) - "Sugar available in system (precondition)"
```

### Output

```python
ResourceAnalysis(
    boundaries=[
        RAClass(name="CoffeeSupplyBoundary", ra_type=BOUNDARY, ...),
        RAClass(name="WaterSupplyBoundary", ra_type=BOUNDARY, ...),
        RAClass(name="MilkSupplyBoundary", ra_type=BOUNDARY, ...),
        RAClass(name="SugarSupplyBoundary", ra_type=BOUNDARY, ...)
    ],
    entities=[
        RAClass(name="Coffee", ra_type=ENTITY, ...),
        RAClass(name="Water", ra_type=ENTITY, ...),
        RAClass(name="Milk", ra_type=ENTITY, ...),
        RAClass(name="Sugar", ra_type=ENTITY, ...)
    ],
    safety_requirements={
        "Water": {
            "temperature": {"max": 100, "unit": "celsius", "critical": True},
            "pressure": {"max": 15, "unit": "bar", "critical": True}
        }
    }
)
```

---

## Phase 3: Interaction Analysis

### UC-Methode Rules Applied

**Interaction Rule 1**: Transaction verbs create Boundaries (User ↔ System)
**Interaction Rule 2**: Transformation verbs create Controllers + Entities
**Interaction Rule 3**: Function verbs assign functions to Material Controllers
**Interaction Rule 4**: Controllers must be material-based, not action-based
**Interaction Rule 5**: Aggregation state determines controller selection

### Input

```
Basic Flow:
B1 (trigger) System clock reaches the user defined time of 7:00h (Radio clock)
B2a The system heats water
B2b The system prepares filter
B2c The system grinds the user defined amount of coffee beans
B2d The system retrieves cup from storage container
B3a The system begins brewing coffee with the user defined amount of water
B3b The system adds milk to the cup
B4 The system outputs a message to user
B5 The system presents cup to user
```

### Detailed Steps

#### Step 3.1: NLP Grammatical Analysis

**Algorithm**:
```python
def analyze_grammar(line_text: str, nlp: spacy.Language) -> GrammaticalAnalysis:
    """
    Extract grammatical components using spaCy
    """
    doc = nlp(line_text)

    analysis = GrammaticalAnalysis()

    # Find main verb
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ["ROOT", "relcl"]:
            analysis.main_verb = token.text
            analysis.verb_lemma = token.lemma_
            break

    # Find direct object
    for token in doc:
        if token.dep_ in ["dobj", "pobj"] and token.head.lemma_ == analysis.verb_lemma:
            analysis.direct_object = token.text
            # Get compound nouns
            compound = []
            for child in token.children:
                if child.dep_ == "compound":
                    compound.append(child.text)
            if compound:
                analysis.direct_object = " ".join(compound + [token.text])
            break

    # Find prepositional objects
    for token in doc:
        if token.dep_ == "prep":
            preposition = token.text
            for child in token.children:
                if child.dep_ == "pobj":
                    analysis.prepositional_objects.append((preposition, child.text))

    return analysis
```

**Example - Step B2a**:
```
Input: "B2a The system heats water"

spaCy Analysis:
  Token: "system"  | POS: NOUN  | DEP: nsubj
  Token: "heats"   | POS: VERB  | DEP: ROOT    ← Main verb
  Token: "water"   | POS: NOUN  | DEP: dobj    ← Direct object

Output: GrammaticalAnalysis(
    main_verb="heats",
    verb_lemma="heat",
    direct_object="water",
    prepositional_objects=[]
)
```

**Example - Step B2c**:
```
Input: "B2c The system grinds the user defined amount of coffee beans"

spaCy Analysis:
  Token: "system"  | POS: NOUN  | DEP: nsubj
  Token: "grinds"  | POS: VERB  | DEP: ROOT    ← Main verb
  Token: "coffee"  | POS: NOUN  | DEP: compound ← Compound
  Token: "beans"   | POS: NOUN  | DEP: dobj    ← Direct object

Output: GrammaticalAnalysis(
    main_verb="grinds",
    verb_lemma="grind",
    direct_object="coffee beans",
    prepositional_objects=[]
)
```

#### Step 3.2: Classify Verb Type

**Algorithm**:
```python
def classify_verb(verb_lemma: str, domain_config: Dict) -> VerbType:
    """
    Classify verb as TRANSACTION, TRANSFORMATION, or FUNCTION
    """
    verb_classification = domain_config.get('verb_classification', {})

    # Check transaction verbs
    transaction_verbs = verb_classification.get('transaction_verbs', {}).get('verbs', {})
    if verb_lemma in transaction_verbs:
        return VerbType.TRANSACTION_VERB

    # Check transformation verbs
    transformation_verbs = verb_classification.get('transformation_verbs', {}).get('verbs', {})
    if verb_lemma in transformation_verbs:
        return VerbType.TRANSFORMATION_VERB

    # Check function verbs
    function_verbs = verb_classification.get('function_verbs', {}).get('verbs', {})
    if verb_lemma in function_verbs:
        return VerbType.FUNCTION_VERB

    return None
```

**Example - Transaction Verb**:
```
Verb: "output"

Domain JSON:
{
  "transaction_verbs": {
    "verbs": {
      "output": "System provides information to user"
    }
  }
}

Classification: TRANSACTION_VERB
```

**Example - Transformation Verb**:
```
Verb: "grind"

Domain JSON:
{
  "transformation_verbs": {
    "verbs": {
      "grind": "CoffeeBeans -> GroundCoffee"
    }
  }
}

Classification: TRANSFORMATION_VERB
Transformation: CoffeeBeans -> GroundCoffee
```

**Example - Function Verb**:
```
Verb: "heat"

Domain JSON:
{
  "function_verbs": {
    "verbs": {
      "heat": "Water -> HotWater (temperature change)"
    }
  }
}

Classification: FUNCTION_VERB
```

#### Step 3.3: Transaction Verb Processing

**UC-Methode Rule**: Transaction verbs create Boundaries

**Algorithm**:
```python
def process_transaction_verb(grammatical: GrammaticalAnalysis, step_id: str) -> List[RAClass]:
    """
    Transaction verbs create boundaries for user-system interaction

    Patterns:
    - "User requests X" → HMI Boundary (input)
    - "System outputs X to user" → Display Boundary (output)
    - "System presents X to user" → Delivery Boundary (output)
    """
    ra_classes = []
    verb = grammatical.verb_lemma

    # Determine boundary type from verb semantics
    if verb in ['request', 'enter', 'press', 'select']:
        # Input boundaries
        boundary = RAClass(
            name="HMIRequestBoundary",
            ra_type=RAType.BOUNDARY,
            stereotype="<<boundary>>",
            description=f"User input boundary for {verb} transaction",
            step_id=step_id
        )
        ra_classes.append(boundary)

    elif verb in ['output', 'display', 'show']:
        # Output boundaries
        boundary = RAClass(
            name="MessageDisplayBoundary",
            ra_type=RAType.BOUNDARY,
            stereotype="<<boundary>>",
            description=f"Output boundary for {verb} transaction",
            step_id=step_id
        )
        ra_classes.append(boundary)

    elif verb in ['present', 'deliver', 'provide']:
        # Product delivery boundaries
        boundary = RAClass(
            name="ProductDeliveryBoundary",
            ra_type=RAType.BOUNDARY,
            stereotype="<<boundary>>",
            description=f"Product delivery boundary for {verb} transaction",
            step_id=step_id
        )
        ra_classes.append(boundary)

    return ra_classes
```

**Example - B4**:
```
Input: "B4 The system outputs a message to user"

Grammatical Analysis:
  verb_lemma: "output"
  verb_type: TRANSACTION_VERB
  direct_object: "message"

Processing:
  verb "output" → Output boundary type

Output: RAClass(
    name="MessageDisplayBoundary",
    ra_type=BOUNDARY,
    stereotype="<<boundary>>",
    description="Output boundary for output transaction",
    step_id="B4"
)
```

#### Step 3.4: Transformation Verb Processing

**UC-Methode Rule**: Transformation verbs create Controllers AND Entities

**Algorithm**:
```python
def process_transformation_verb(
    grammatical: GrammaticalAnalysis,
    step_id: str,
    domain_config: Dict,
    material_registry: MaterialControllerRegistry
) -> Tuple[RAClass, List[RAClass]]:
    """
    Transformation verbs create:
    1. Controller (material-based, with aggregation state)
    2. Output Entity (from transformation)
    """
    verb = grammatical.verb_lemma

    # Get transformation from domain JSON
    transformation_info = domain_config['verb_classification']['transformation_verbs']['verbs'][verb]
    # Example: "CoffeeBeans -> GroundCoffee"

    # Parse transformation
    if '->' in transformation_info:
        parts = transformation_info.split('->')
        input_materials = parts[0].strip()
        output_material = parts[1].strip()

    # STEP 1: Determine aggregation state from output
    aggregation_state = determine_aggregation_state(output_material, verb, domain_config)

    # STEP 2: Extract base material
    base_material = extract_base_material(output_material)
    # "GroundCoffee" -> "coffee"

    # STEP 3: Select or create material controller
    controller_name = material_registry.get_or_create_controller(
        material=base_material,
        aggregation_state=aggregation_state,
        function=verb
    )

    controller = RAClass(
        name=controller_name,
        ra_type=RAType.CONTROLLER,
        stereotype="<<controller>>",
        description=f"Manages {base_material} ({aggregation_state} state): {verb}() in {step_id}",
        step_id=step_id
    )

    # STEP 4: Create output entity
    output_entity = RAClass(
        name=output_material,
        ra_type=RAType.ENTITY,
        stereotype="<<entity>>",
        description=f"Product of {verb} transformation",
        step_id=step_id
    )

    # STEP 5: Create input entities if specified
    input_entities = []
    if '+' in input_materials:
        for input_mat in input_materials.split('+'):
            input_entity = RAClass(
                name=input_mat.strip(),
                ra_type=RAType.ENTITY,
                stereotype="<<entity>>",
                description=f"Input for {verb} transformation",
                step_id=step_id
            )
            input_entities.append(input_entity)

    return controller, [output_entity] + input_entities
```

**Example - B2c (Transformation Verb "grind")**:
```
Input: "B2c The system grinds the user defined amount of coffee beans"

Step-by-Step:

1. Grammatical Analysis:
   verb_lemma: "grind"
   verb_type: TRANSFORMATION_VERB
   direct_object: "coffee beans"

2. Get Transformation from Domain:
   transformation_info: "CoffeeBeans -> GroundCoffee"
   input: "CoffeeBeans"
   output: "GroundCoffee"

3. Determine Aggregation State:
   output_material: "GroundCoffee"
   keyword "ground" found in domain_config['aggregation_states']['solid']['specific_keywords']
   → aggregation_state: "solid"

4. Extract Base Material:
   "GroundCoffee" → check domain materials
   found "coffee" in material variants
   → base_material: "coffee"

5. Select Controller:
   material_registry.get_or_create_controller(
       material="coffee",
       aggregation_state="solid",
       function="grind"
   )
   → controller_name: "CoffeeSolidManager"

6. Create Output Entity:
   name: "GroundCoffee"
   ra_type: ENTITY
   description: "Product of grind transformation"

Output:
  Controller: CoffeeSolidManager (<<controller>>)
     - Manages coffee (solid state): grind() in B2c

  Entity: GroundCoffee (<<entity>>)
     - Product of grind transformation
```

#### Step 3.5: Determine Aggregation State (Detailed)

**Algorithm**:
```python
def determine_aggregation_state(
    output_material: str,
    verb: str,
    domain_config: Dict
) -> Optional[str]:
    """
    Determine aggregation state from:
    1. Output material keywords
    2. Verb operation type
    3. Material-specific context
    """
    output_lower = output_material.lower()
    aggregation_states = domain_config.get('aggregation_states', {})

    # STEP 1: Check output material name for state keywords
    for state, state_info in aggregation_states.items():
        specific_keywords = state_info.get('specific_keywords', [])
        for keyword in specific_keywords:
            if keyword in output_lower:
                print(f"[STATE] Detected {state} from output '{output_material}' (keyword: {keyword})")
                return state

    # STEP 2: Check verb operation type
    for state, state_info in aggregation_states.items():
        specific_operations = state_info.get('specific_operations', [])
        if verb in specific_operations:
            print(f"[STATE] Detected {state} from verb '{verb}' operation")
            return state

    # STEP 3: Check material-specific context
    material_contexts = domain_config.get('material_specific_contexts', {})
    for material_name, contexts in material_contexts.items():
        if material_name.lower() in output_lower:
            # Check default state for this material
            default_state = contexts.get('default_state')
            if default_state:
                print(f"[STATE] Using default state {default_state} for material {material_name}")
                return default_state

    return None
```

**Example - "GroundCoffee"**:
```
Input:
  output_material: "GroundCoffee"
  verb: "grind"

Domain Config - aggregation_states:
{
  "solid": {
    "specific_keywords": ["beans", "ground", "powder", "crystals"],
    "specific_operations": ["grind", "crush", "mill"]
  }
}

Processing:

STEP 1: Check output material keywords
  output_lower = "groundcoffee"
  Check solid keywords: ["beans", "ground", "powder", "crystals"]
  → "ground" found in "groundcoffee"

  Result: state = "solid"

Output: "solid"
```

**Example - "Coffee" (from brewing)**:
```
Input:
  output_material: "Coffee"
  verb: "brew"

Domain Config - aggregation_states:
{
  "liquid": {
    "specific_keywords": ["brew", "brewing", "extraction", "pour"],
    "specific_operations": ["heat", "press", "pump", "pour"]
  }
}

Processing:

STEP 2: Check verb operation type
  verb = "brew"
  Check liquid operations: ["heat", "press", "pump", "pour"]
  → "brew" NOT in operations list

STEP 1: Check output material keywords (retry)
  output_lower = "coffee"
  Check liquid keywords: ["brew", "brewing", "extraction", "pour"]
  → "brew" substring NOT found in "coffee"

STEP 2: Check verb again with specific_keywords
  verb = "brew"
  → "brew" found in liquid.specific_keywords

  Result: state = "liquid"

Output: "liquid"
```

#### Step 3.6: Function Verb Processing

**UC-Methode Rule**: Function verbs assign functions to Material Controllers

**Algorithm**:
```python
def process_function_verb(
    grammatical: GrammaticalAnalysis,
    step_id: str,
    domain_config: Dict,
    material_registry: MaterialControllerRegistry
) -> RAClass:
    """
    Function verbs assign functions to existing material controllers
    """
    verb = grammatical.verb_lemma
    direct_object = grammatical.direct_object

    # STEP 1: Extract material from direct object
    material = extract_material_from_object(direct_object, domain_config)

    # STEP 2: Determine aggregation state from verb
    aggregation_state = determine_state_from_verb(verb, domain_config)

    # STEP 3: Get or create controller
    controller_name = material_registry.get_or_create_controller(
        material=material,
        aggregation_state=aggregation_state,
        function=verb
    )

    controller = RAClass(
        name=controller_name,
        ra_type=RAType.CONTROLLER,
        stereotype="<<controller>>",
        description=f"Manages {material} ({aggregation_state} state): {verb}() in {step_id}",
        step_id=step_id
    )

    return controller
```

**Example - B2a (Function Verb "heat")**:
```
Input: "B2a The system heats water"

Step-by-Step:

1. Grammatical Analysis:
   verb_lemma: "heat"
   verb_type: FUNCTION_VERB
   direct_object: "water"

2. Extract Material:
   direct_object: "water"
   → material: "water"

3. Determine State from Verb:
   verb: "heat"
   Check domain_config['aggregation_states']['liquid']['specific_operations']
   → "heat" found in liquid operations
   → aggregation_state: "liquid"

4. Get Controller:
   material_registry.get_or_create_controller(
       material="water",
       aggregation_state="liquid",
       function="heat"
   )
   → controller_name: "WaterLiquidManager"

Output:
  Controller: WaterLiquidManager (<<controller>>)
     - Manages water (liquid state): heat() in B2a
```

### Output Phase 3

```python
InteractionAnalysis(
    controllers=[
        RAClass(name="SystemControlManager", ra_type=CONTROLLER, step_id="B1"),
        RAClass(name="WaterLiquidManager", ra_type=CONTROLLER, step_id="B2a"),
        RAClass(name="FilterManager", ra_type=CONTROLLER, step_id="B2b"),
        RAClass(name="CoffeeSolidManager", ra_type=CONTROLLER, step_id="B2c"),
        RAClass(name="CupManager", ra_type=CONTROLLER, step_id="B2d"),
        RAClass(name="CoffeeLiquidManager", ra_type=CONTROLLER, step_id="B3a"),
        RAClass(name="MilkLiquidManager", ra_type=CONTROLLER, step_id="B3b"),
        RAClass(name="HMIManager", ra_type=CONTROLLER, step_id="B4"),
    ],
    boundaries=[
        RAClass(name="TimingBoundary", ra_type=BOUNDARY, step_id="B1"),
        RAClass(name="MessageDisplayBoundary", ra_type=BOUNDARY, step_id="B4"),
        RAClass(name="ProductDeliveryBoundary", ra_type=BOUNDARY, step_id="B5"),
    ],
    entities=[
        RAClass(name="HotWater", ra_type=ENTITY, step_id="B2a"),
        RAClass(name="Filter", ra_type=ENTITY, step_id="B2b"),
        RAClass(name="GroundCoffee", ra_type=ENTITY, step_id="B2c"),
        RAClass(name="Cup", ra_type=ENTITY, step_id="B2d"),
        RAClass(name="Coffee", ra_type=ENTITY, step_id="B3a"),
        RAClass(name="Message", ra_type=ENTITY, step_id="B4"),
    ]
)
```

---

## Phase 4: Control Flow Generation (UC-Methode Rules 1-5)

### UC-Methode Rules Applied

**Control Flow Rule 1**: Serial → Serial = Direct Connection
**Control Flow Rule 2**: Serial → Parallel = Add Distribution Node (PX_START)
**Control Flow Rule 3**: Parallel → Parallel (Same Step) = Via Distribution/Merge Nodes
**Control Flow Rule 4**: Parallel → Parallel (Different Step) = Connect Merge to Distribution
**Control Flow Rule 5**: Parallel → Serial = Add Merge Node (PX_END)

### Input

```python
Steps with Controllers:
  B1 (serial, group=0) → SystemControlManager
  B2a (parallel, group=2) → WaterLiquidManager
  B2b (parallel, group=2) → FilterManager
  B2c (parallel, group=2) → CoffeeSolidManager
  B2d (parallel, group=2) → CupManager
  B3a (parallel, group=3) → CoffeeLiquidManager
  B3b (parallel, group=3) → MilkLiquidManager
  B4 (serial, group=0) → HMIManager
  B5 (serial, group=0) → CupManager
```

### Detailed Steps

#### Step 4.1: Detect Parallel Groups

**Algorithm**:
```python
def detect_parallel_groups(steps: List[Step]) -> Dict[int, List[Step]]:
    """
    Detect parallel groups from step IDs

    Pattern: B2a, B2b, B2c → Group 2 (same base "B2", different suffix 'a', 'b', 'c')
    """
    groups = {}

    for step in steps:
        # Match pattern: B2a, B3b, E1.2c
        match = re.match(r'^([BAE]\d+(?:\.\d+)?)([a-z])$', step.step_id)
        if match:
            base_step = match.group(1)  # "B2", "B3", "E1.2"
            suffix = match.group(2)      # "a", "b", "c"

            # Extract group number
            num_match = re.search(r'\d+', base_step)
            if num_match:
                group_num = int(num_match.group())

                if group_num not in groups:
                    groups[group_num] = []

                groups[group_num].append(step)

    return groups
```

**Example**:
```
Input Steps:
  B1 → No suffix → serial (group 0)
  B2a → suffix 'a' → parallel
  B2b → suffix 'b' → parallel
  B2c → suffix 'c' → parallel
  B2d → suffix 'd' → parallel
  B3a → suffix 'a' → parallel
  B3b → suffix 'b' → parallel
  B4 → No suffix → serial (group 0)

Detection:
  Step B2a: base="B2", suffix="a", group_num=2
  Step B2b: base="B2", suffix="b", group_num=2
  Step B2c: base="B2", suffix="c", group_num=2
  Step B2d: base="B2", suffix="d", group_num=2

  Step B3a: base="B3", suffix="a", group_num=3
  Step B3b: base="B3", suffix="b", group_num=3

Result:
  groups[2] = [B2a, B2b, B2c, B2d]
  groups[3] = [B3a, B3b]
```

#### Step 4.2: Create Parallel Flow Nodes

**UC-Methode Rule**: Each parallel group needs START (distribution) and END (merge) nodes

**Algorithm**:
```python
def create_parallel_flow_nodes(parallel_groups: Dict[int, List[Step]]) -> List[RAClass]:
    """
    Create distribution and merge nodes for each parallel group
    """
    flow_nodes = []

    for group_num, steps in parallel_groups.items():
        # Distribution node (START)
        start_node = RAClass(
            name=f"P{group_num}_START",
            ra_type=RAType.CONTROL_FLOW_NODE,
            stereotype="<<flow>>",
            description=f"Distribution node for parallel flow P{group_num}",
            element_type="distribution",
            parallel_group=0  # Flow nodes themselves are not in parallel groups
        )
        flow_nodes.append(start_node)

        # Merge node (END)
        end_node = RAClass(
            name=f"P{group_num}_END",
            ra_type=RAType.CONTROL_FLOW_NODE,
            stereotype="<<flow>>",
            description=f"Merge node for parallel flow P{group_num}",
            element_type="merge",
            parallel_group=0
        )
        flow_nodes.append(end_node)

    return flow_nodes
```

**Example**:
```
Parallel Groups:
  Group 2: [B2a, B2b, B2c, B2d]
  Group 3: [B3a, B3b]

Created Nodes:
  P2_START (<<flow>>) - Distribution node for parallel flow P2
  P2_END (<<flow>>) - Merge node for parallel flow P2
  P3_START (<<flow>>) - Distribution node for parallel flow P3
  P3_END (<<flow>>) - Merge node for parallel flow P3
```

#### Step 4.3: Apply UC-Methode Rule 1 (Serial → Serial)

**UC-Methode Rule 1**: Direct connection between sequential steps

**Algorithm**:
```python
def apply_rule_1(current_step: Step, next_step: Step) -> Optional[ControlFlow]:
    """
    Rule 1: Serial → Serial
    Create direct connection
    """
    if current_step.is_serial() and next_step.is_serial():
        # Check if they are in the same flow scope (not crossing Alt/Ext boundaries)
        if are_in_same_flow_scope(current_step, next_step):
            flow = ControlFlow(
                source_step=current_step.step_id,
                target_step=next_step.step_id,
                source_controller=current_step.controller.name,
                target_controller=next_step.controller.name,
                flow_type='sequential',
                rule='Rule 1 - Serial to Serial',
                description=f"Sequential flow from {current_step.step_id} to {next_step.step_id}"
            )
            return flow

    return None
```

**Example**:
```
Current: B4 (HMIManager, serial)
Next: B5 (CupManager, serial)

Check:
  ✓ Both are serial
  ✓ Same flow scope (both in main flow)

Create Flow:
  source: B4 (HMIManager)
  target: B5 (CupManager)
  rule: "Rule 1 - Serial to Serial"

Visual:
  HMIManager ──→ CupManager
```

#### Step 4.4: Apply UC-Methode Rule 2 (Serial → Parallel)

**UC-Methode Rule 2**: Insert distribution node when transitioning to parallel

**Algorithm**:
```python
def apply_rule_2(current_step: Step, next_step: Step, flow_nodes: Dict) -> List[ControlFlow]:
    """
    Rule 2: Serial → Parallel
    Insert distribution node (PX_START)
    """
    flows = []

    if current_step.is_serial() and next_step.is_parallel():
        parallel_group = next_step.parallel_group
        start_node = flow_nodes[f'P{parallel_group}_START']

        # Flow 1: Serial Controller → Distribution Node
        flow1 = ControlFlow(
            source_step=current_step.step_id,
            target_step=start_node.name,
            source_controller=current_step.controller.name,
            target_controller=start_node.name,
            flow_type='distribution',
            rule='Rule 2 - Serial to Parallel Distribution',
            description=f"{current_step.step_id} distributes to parallel group {parallel_group}"
        )
        flows.append(flow1)

    return flows
```

**Example**:
```
Current: B1 (SystemControlManager, serial)
Next: B2a (WaterLiquidManager, parallel, group=2)

Process:
  1. Detect transition: serial → parallel
  2. Get parallel group: 2
  3. Get distribution node: P2_START

Create Flow:
  source: B1 (SystemControlManager)
  target: P2_START
  rule: "Rule 2 - Serial to Parallel Distribution"

Visual:
  SystemControlManager ──→ P2_START ⬥
```

#### Step 4.5: Apply UC-Methode Rule 3 (Parallel → Parallel, Same Step)

**UC-Methode Rule 3**: Parallel steps in same group connect via distribution/merge nodes

**Algorithm**:
```python
def apply_rule_3(
    parallel_steps: List[Step],
    parallel_group: int,
    flow_nodes: Dict
) -> List[ControlFlow]:
    """
    Rule 3: Parallel → Parallel (same step number)
    All parallel steps connect via START and END nodes
    """
    flows = []

    start_node = flow_nodes[f'P{parallel_group}_START']
    end_node = flow_nodes[f'P{parallel_group}_END']

    for step in parallel_steps:
        # Flow 1: Distribution Node → Parallel Controller
        flow_in = ControlFlow(
            source_step=start_node.name,
            target_step=step.step_id,
            source_controller=start_node.name,
            target_controller=step.controller.name,
            flow_type='parallel_start',
            rule='Rule 3: parallel -> parallel (same step number)',
            description=f"Parallel branch from P{parallel_group}_START to {step.step_id}"
        )
        flows.append(flow_in)

        # Flow 2: Parallel Controller → Merge Node
        flow_out = ControlFlow(
            source_step=step.step_id,
            target_step=end_node.name,
            source_controller=step.controller.name,
            target_controller=end_node.name,
            flow_type='parallel_end',
            rule='Rule 3: parallel -> parallel (same step number)',
            description=f"Parallel branch from {step.step_id} to P{parallel_group}_END"
        )
        flows.append(flow_out)

    return flows
```

**Example - Group 2 (B2a, B2b, B2c, B2d)**:
```
Parallel Steps in Group 2:
  B2a (WaterLiquidManager)
  B2b (FilterManager)
  B2c (CoffeeSolidManager)
  B2d (CupManager)

Flow Nodes:
  P2_START (distribution)
  P2_END (merge)

Created Flows:

Inbound Flows (Rule 3):
  P2_START → B2a (WaterLiquidManager)
  P2_START → B2b (FilterManager)
  P2_START → B2c (CoffeeSolidManager)
  P2_START → B2d (CupManager)

Outbound Flows (Rule 3):
  B2a (WaterLiquidManager) → P2_END
  B2b (FilterManager) → P2_END
  B2c (CoffeeSolidManager) → P2_END
  B2d (CupManager) → P2_END

Visual:
                    ┌→ WaterLiquidManager ──┐
                    │                        │
  P2_START ⬥ ───────┼→ FilterManager ───────┼──→ ⬥ P2_END
                    │                        │
                    ├→ CoffeeSolidManager ───┤
                    │                        │
                    └→ CupManager ───────────┘
```

#### Step 4.6: Apply UC-Methode Rule 4 (Parallel → Parallel, Different Step)

**UC-Methode Rule 4**: Connect merge node of one group to distribution node of next group

**Algorithm**:
```python
def apply_rule_4(
    current_group: int,
    next_group: int,
    flow_nodes: Dict
) -> ControlFlow:
    """
    Rule 4: Parallel → Parallel (different step numbers)
    Connect merge of current group to distribution of next group
    """
    current_end = flow_nodes[f'P{current_group}_END']
    next_start = flow_nodes[f'P{next_group}_START']

    flow = ControlFlow(
        source_step=current_end.name,
        target_step=next_start.name,
        source_controller=current_end.name,
        target_controller=next_start.name,
        flow_type='group_transition',
        rule='Rule 4 - Parallel Group Transition',
        description=f"Transition from parallel group {current_group} to group {next_group}"
    )

    return flow
```

**Example - Group 2 to Group 3**:
```
Current Group: 2 (B2a, B2b, B2c, B2d)
Next Group: 3 (B3a, B3b)

Flow Nodes:
  P2_END (merge of group 2)
  P3_START (distribution of group 3)

Created Flow:
  source: P2_END
  target: P3_START
  rule: "Rule 4 - Parallel Group Transition"

Visual:
  P2_END ⬥ ──→ ⬥ P3_START
```

#### Step 4.7: Apply UC-Methode Rule 5 (Parallel → Serial)

**UC-Methode Rule 5**: Merge node connects to next serial step

**Algorithm**:
```python
def apply_rule_5(
    last_parallel_group: int,
    next_serial_step: Step,
    flow_nodes: Dict
) -> ControlFlow:
    """
    Rule 5: Parallel → Serial
    Merge node connects to serial controller
    """
    merge_node = flow_nodes[f'P{last_parallel_group}_END']

    flow = ControlFlow(
        source_step=merge_node.name,
        target_step=next_serial_step.step_id,
        source_controller=merge_node.name,
        target_controller=next_serial_step.controller.name,
        flow_type='convergence',
        rule='Rule 5 - Parallel to Serial Merge',
        description=f"Merge from parallel group {last_parallel_group} to serial step {next_serial_step.step_id}"
    )

    return flow
```

**Example - Group 3 to B4**:
```
Last Parallel Group: 3 (B3a, B3b)
Next Serial Step: B4 (HMIManager)

Flow Nodes:
  P3_END (merge of group 3)

Created Flow:
  source: P3_END
  target: B4 (HMIManager)
  rule: "Rule 5 - Parallel to Serial Merge"

Visual:
  P3_END ⬥ ──→ HMIManager
```

### Complete Control Flow Example (UC1)

```
Full Flow:

1. SystemControlManager
       ↓ (Rule 1 → Rule 2)
2. P2_START ⬥
       ├──→ WaterLiquidManager ──┐
       ├──→ FilterManager ────────┤
       ├──→ CoffeeSolidManager ───┤ (Rule 3)
       └──→ CupManager ───────────┘
            ↓ (all merge to)
3. P2_END ⬥
       ↓ (Rule 4)
4. P3_START ⬥
       ├──→ CoffeeLiquidManager ──┐
       └──→ MilkLiquidManager ─────┘ (Rule 3)
            ↓ (merge to)
5. P3_END ⬥
       ↓ (Rule 5)
6. HMIManager
       ↓ (Rule 1)
7. CupManager
```

### Output Phase 4

```python
ControlFlowAnalysis(
    control_flows=[
        # Rule 1: B1 → B4 (after parallel groups)
        ControlFlow(source="SystemControlManager", target="P2_START", rule="Rule 2"),

        # Rule 2: SystemControlManager → P2_START
        ControlFlow(source="SystemControlManager", target="P2_START", rule="Rule 2"),

        # Rule 3: P2_START → Parallel Group 2
        ControlFlow(source="P2_START", target="WaterLiquidManager", rule="Rule 3"),
        ControlFlow(source="P2_START", target="FilterManager", rule="Rule 3"),
        ControlFlow(source="P2_START", target="CoffeeSolidManager", rule="Rule 3"),
        ControlFlow(source="P2_START", target="CupManager", rule="Rule 3"),

        # Rule 3: Parallel Group 2 → P2_END
        ControlFlow(source="WaterLiquidManager", target="P2_END", rule="Rule 3"),
        ControlFlow(source="FilterManager", target="P2_END", rule="Rule 3"),
        ControlFlow(source="CoffeeSolidManager", target="P2_END", rule="Rule 3"),
        ControlFlow(source="CupManager", target="P2_END", rule="Rule 3"),

        # Rule 4: P2_END → P3_START
        ControlFlow(source="P2_END", target="P3_START", rule="Rule 4"),

        # Rule 3: P3_START → Parallel Group 3
        ControlFlow(source="P3_START", target="CoffeeLiquidManager", rule="Rule 3"),
        ControlFlow(source="P3_START", target="MilkLiquidManager", rule="Rule 3"),

        # Rule 3: Parallel Group 3 → P3_END
        ControlFlow(source="CoffeeLiquidManager", target="P3_END", rule="Rule 3"),
        ControlFlow(source="MilkLiquidManager", target="P3_END", rule="Rule 3"),

        # Rule 5: P3_END → HMIManager
        ControlFlow(source="P3_END", target="HMIManager", rule="Rule 5"),

        # Rule 1: HMIManager → CupManager
        ControlFlow(source="HMIManager", target="CupManager", rule="Rule 1"),
    ],
    parallel_flow_nodes=[
        RAClass(name="P2_START", ra_type=CONTROL_FLOW_NODE),
        RAClass(name="P2_END", ra_type=CONTROL_FLOW_NODE),
        RAClass(name="P3_START", ra_type=CONTROL_FLOW_NODE),
        RAClass(name="P3_END", ra_type=CONTROL_FLOW_NODE),
    ]
)
```

---

## Phase 5: Data Flow Analysis

### UC-Methode Rules Applied

**Data Flow Rule 1**: USE relationships represent entity consumption by controllers
**Data Flow Rule 2**: PROVIDE relationships represent entity creation by controllers
**Data Flow Rule 3**: Prepositions determine flow direction ("with" = USE, "to" = PROVIDE)
**Data Flow Rule 4**: Transformation patterns create both USE and PROVIDE flows

### Input

```
Steps with Grammatical Analysis:
B2a: "The system heats water"
  - verb: heat
  - object: water
  - prepositions: []

B2c: "The system grinds the user defined amount of coffee beans with the user defined grinding degree"
  - verb: grind
  - object: coffee beans
  - prepositions: [("with", "grinding degree")]

B3a: "The system begins brewing coffee with the user defined amount of water into the cup"
  - verb: brew
  - object: coffee
  - prepositions: [("with", "water"), ("into", "cup")]

B4: "The system outputs a message to user"
  - verb: output
  - object: message
  - prepositions: [("to", "user")]
```

### Detailed Steps

#### Step 5.1: Identify Prepositional Objects

**Algorithm**:
```python
def identify_prepositional_objects(grammatical: GrammaticalAnalysis) -> List[Tuple[str, str]]:
    """
    Extract prepositional phrases using spaCy dependency parsing
    """
    prep_objects = []

    for token in grammatical.doc:
        if token.dep_ == "prep":
            preposition = token.text

            # Find the object of this preposition
            for child in token.children:
                if child.dep_ == "pobj":
                    object_text = child.text

                    # Handle compound nouns
                    compounds = []
                    for subchild in child.children:
                        if subchild.dep_ == "compound":
                            compounds.append(subchild.text)

                    if compounds:
                        object_text = " ".join(compounds + [child.text])

                    prep_objects.append((preposition, object_text))

    return prep_objects
```

**Example - B2c**:
```
Input: "The system grinds the user defined amount of coffee beans with the user defined grinding degree"

spaCy Parsing:
  Token "with" | DEP: prep
    ├─ child "degree" | DEP: pobj
    │   └─ child "grinding" | DEP: compound

Output: [("with", "grinding degree")]
```

**Example - B3a**:
```
Input: "The system begins brewing coffee with the user defined amount of water into the cup"

spaCy Parsing:
  Token "with" | DEP: prep
    └─ child "water" | DEP: pobj

  Token "into" | DEP: prep
    └─ child "cup" | DEP: pobj

Output: [("with", "water"), ("into", "cup")]
```

#### Step 5.2: Classify Preposition Semantics

**Algorithm**:
```python
def classify_preposition(preposition: str, verb: str, object: str) -> str:
    """
    Classify preposition as USE or PROVIDE based on semantics

    USE Prepositions (input/consumption):
    - "with", "from", "using", "via"

    PROVIDE Prepositions (output/creation):
    - "to", "for", "into" (when output context)
    """
    # USE: Input or tool
    if preposition.lower() in ['with', 'from', 'using', 'via', 'by']:
        return 'USE'

    # PROVIDE: Output or destination
    if preposition.lower() in ['to', 'for']:
        return 'PROVIDE'

    # "into" is context-dependent
    if preposition.lower() == 'into':
        # If verb is transformation/creation, "into" = PROVIDE
        if verb in ['brew', 'pour', 'mix', 'add', 'grind']:
            return 'PROVIDE'
        else:
            return 'USE'

    return 'UNKNOWN'
```

**Example**:
```
Preposition: "with"
  → Classification: USE (input/tool)

Preposition: "to"
  → Classification: PROVIDE (output/destination)

Preposition: "into" + verb "brew"
  → Classification: PROVIDE (transformation output)

Preposition: "into" + verb "retrieve"
  → Classification: USE (retrieval source)
```

#### Step 5.3: Create USE Relationships

**Data Flow Rule 1**: USE = Entity → Controller (input)

**Algorithm**:
```python
def create_use_relationships(
    step: Step,
    controller: RAClass,
    prep_objects: List[Tuple[str, str]]
) -> List[DataFlow]:
    """
    Create USE relationships for input entities
    """
    use_flows = []

    for preposition, object_name in prep_objects:
        flow_type = classify_preposition(preposition, step.verb, object_name)

        if flow_type == 'USE':
            # Find or create entity
            entity_name = normalize_entity_name(object_name)

            data_flow = DataFlow(
                step_id=step.step_id,
                controller=controller.name,
                entity=entity_name,
                flow_type='use',
                preposition=preposition,
                description=f"{controller.name} uses {entity_name} (via {preposition})"
            )
            use_flows.append(data_flow)

    return use_flows
```

**Example - B2c**:
```
Step: B2c
Controller: CoffeeSolidManager
Prepositional Objects: [("with", "grinding degree")]

Process:
  1. preposition: "with"
  2. object: "grinding degree"
  3. classify_preposition("with") → USE
  4. entity_name: "GrindingDegree"

Created DataFlow:
  controller: CoffeeSolidManager
  entity: GrindingDegree
  flow_type: USE
  preposition: "with"
  description: "CoffeeSolidManager uses GrindingDegree (via with)"

Visual:
  GrindingDegree ──use──> CoffeeSolidManager
```

#### Step 5.4: Create PROVIDE Relationships

**Data Flow Rule 2**: PROVIDE = Controller → Entity (output)

**Algorithm**:
```python
def create_provide_relationships(
    step: Step,
    controller: RAClass,
    prep_objects: List[Tuple[str, str]],
    output_entity: Optional[RAClass]
) -> List[DataFlow]:
    """
    Create PROVIDE relationships for output entities
    """
    provide_flows = []

    # Explicit PROVIDE from prepositions
    for preposition, object_name in prep_objects:
        flow_type = classify_preposition(preposition, step.verb, object_name)

        if flow_type == 'PROVIDE':
            entity_name = normalize_entity_name(object_name)

            data_flow = DataFlow(
                step_id=step.step_id,
                controller=controller.name,
                entity=entity_name,
                flow_type='provide',
                preposition=preposition,
                description=f"{controller.name} provides {entity_name} (via {preposition})"
            )
            provide_flows.append(data_flow)

    # Implicit PROVIDE from transformation output
    if output_entity:
        data_flow = DataFlow(
            step_id=step.step_id,
            controller=controller.name,
            entity=output_entity.name,
            flow_type='provide',
            preposition='',
            description=f"{controller.name} produces {output_entity.name}"
        )
        provide_flows.append(data_flow)

    return provide_flows
```

**Example - B4**:
```
Step: B4
Controller: HMIManager
Prepositional Objects: [("to", "user")]
Output Entity: Message

Process:
  1. preposition: "to"
  2. object: "user"
  3. classify_preposition("to") → PROVIDE
  4. entity_name: "User" (Actor, special case)

Created DataFlow:
  controller: HMIManager
  entity: Message
  flow_type: PROVIDE
  preposition: "to"
  description: "HMIManager provides Message (via to)"

Visual:
  HMIManager ──provide──> Message ──→ User
```

#### Step 5.5: Transformation Pattern Data Flows

**Data Flow Rule 4**: Transformations create both USE (inputs) and PROVIDE (output)

**Algorithm**:
```python
def create_transformation_dataflows(
    step: Step,
    controller: RAClass,
    transformation_info: str
) -> Tuple[List[DataFlow], List[DataFlow]]:
    """
    Create USE and PROVIDE flows from transformation pattern

    Example: "CoffeeBeans + HotWater -> Coffee"
    USE: CoffeeBeans → Controller, HotWater → Controller
    PROVIDE: Controller → Coffee
    """
    use_flows = []
    provide_flows = []

    if '->' not in transformation_info:
        return use_flows, provide_flows

    # Parse transformation
    parts = transformation_info.split('->')
    inputs = parts[0].strip()
    output = parts[1].strip()

    # CREATE USE FLOWS for each input
    if '+' in inputs:
        input_materials = [m.strip() for m in inputs.split('+')]
    else:
        input_materials = [inputs]

    for input_mat in input_materials:
        use_flow = DataFlow(
            step_id=step.step_id,
            controller=controller.name,
            entity=input_mat,
            flow_type='use',
            preposition='',
            description=f"{controller.name} uses {input_mat} for transformation"
        )
        use_flows.append(use_flow)

    # CREATE PROVIDE FLOW for output
    provide_flow = DataFlow(
        step_id=step.step_id,
        controller=controller.name,
        entity=output,
        flow_type='provide',
        preposition='',
        description=f"{controller.name} produces {output} from transformation"
    )
    provide_flows.append(provide_flow)

    return use_flows, provide_flows
```

**Example - B3a (brew)**:
```
Step: B3a "The system begins brewing coffee"
Controller: CoffeeLiquidManager
Transformation: "GroundCoffee + HotWater -> Coffee"

Parse Transformation:
  inputs: "GroundCoffee + HotWater"
  output: "Coffee"

Split Inputs:
  input_materials: ["GroundCoffee", "HotWater"]

Created USE Flows:
  1. GroundCoffee → CoffeeLiquidManager (USE)
  2. HotWater → CoffeeLiquidManager (USE)

Created PROVIDE Flow:
  CoffeeLiquidManager → Coffee (PROVIDE)

Visual:
  GroundCoffee ─┐
                ├──use──> CoffeeLiquidManager ──provide──> Coffee
   HotWater ────┘
```

### Output Phase 5

```python
DataFlowAnalysis(
    use_flows=[
        # B2c: CoffeeSolidManager uses GrindingDegree
        DataFlow(
            step_id="B2c",
            controller="CoffeeSolidManager",
            entity="GrindingDegree",
            flow_type="use",
            preposition="with"
        ),

        # B3a: CoffeeLiquidManager uses GroundCoffee + HotWater
        DataFlow(
            step_id="B3a",
            controller="CoffeeLiquidManager",
            entity="GroundCoffee",
            flow_type="use",
            preposition=""
        ),
        DataFlow(
            step_id="B3a",
            controller="CoffeeLiquidManager",
            entity="HotWater",
            flow_type="use",
            preposition=""
        ),
    ],

    provide_flows=[
        # B2a: WaterLiquidManager provides HotWater
        DataFlow(
            step_id="B2a",
            controller="WaterLiquidManager",
            entity="HotWater",
            flow_type="provide",
            preposition=""
        ),

        # B2c: CoffeeSolidManager provides GroundCoffee
        DataFlow(
            step_id="B2c",
            controller="CoffeeSolidManager",
            entity="GroundCoffee",
            flow_type="provide",
            preposition=""
        ),

        # B3a: CoffeeLiquidManager provides Coffee
        DataFlow(
            step_id="B3a",
            controller="CoffeeLiquidManager",
            entity="Coffee",
            flow_type="provide",
            preposition=""
        ),

        # B4: HMIManager provides Message
        DataFlow(
            step_id="B4",
            controller="HMIManager",
            entity="Message",
            flow_type="provide",
            preposition="to"
        ),
    ]
)
```

---

## Phase 6: Actor-Boundary Flows

### UC-Methode Rules Applied

**Actor Rule 1**: Actors interact with system via Boundaries only
**Actor Rule 2**: User interactions route through HMI Controller
**Actor Rule 3**: Extension/Alternative triggers use HMI pattern for user interactions
**Actor Rule 4**: Time-based triggers connect directly to System Controller

### Input

```
Steps with Actor Interactions:
B1 (trigger): System clock reaches 7:00h → Time actor
B4: System outputs message to user → User actor
B5: System presents cup to user → User actor
E1 (trigger): User wants sugar in coffee → User actor
```

### Detailed Steps

#### Step 6.1: Identify Actor Types

**Algorithm**:
```python
def identify_actors(uc_context: UCContext, steps: List[Step]) -> List[RAClass]:
    """
    Identify actors from:
    1. UC context (explicit actor list)
    2. Step text analysis (mentions of "user", "system", "timer", etc.)
    """
    actors = []
    seen_actors = set()

    # From UC context
    for actor_name in uc_context.actors:
        if actor_name not in seen_actors:
            actor = RAClass(
                name=actor_name,
                ra_type=RAType.ACTOR,
                stereotype="<<actor>>",
                description=f"{actor_name} actor from UC context"
            )
            actors.append(actor)
            seen_actors.add(actor_name)

    # From step text
    for step in steps:
        line_lower = step.line_text.lower()

        # Check for "user" mentions
        if is_real_user_interaction(line_lower, step.grammatical):
            if 'User' not in seen_actors:
                actor = RAClass(
                    name='User',
                    ra_type=RAType.ACTOR,
                    stereotype="<<actor>>",
                    description="User actor"
                )
                actors.append(actor)
                seen_actors.add('User')

        # Check for time-based triggers
        if 'clock' in line_lower or 'timer' in line_lower or 'time' in line_lower:
            if 'Time' not in seen_actors:
                actor = RAClass(
                    name='Time',
                    ra_type=RAType.ACTOR,
                    stereotype="<<actor>>",
                    description="Time/Clock actor"
                )
                actors.append(actor)
                seen_actors.add('Time')

    return actors
```

**Example**:
```
UC Context Actors: []

Step Analysis:
  B1: "System clock reaches the user defined time of 7:00h"
    → Contains "clock", "time" → Time actor

  B4: "The system outputs a message to user"
    → Contains "to user" → User actor

  E1: "User wants sugar in coffee"
    → Starts with "User" → User actor (already added)

Identified Actors:
  1. Time (<<actor>>) - "Time/Clock actor"
  2. User (<<actor>>) - "User actor"
```

#### Step 6.2: Detect User Interactions

**Algorithm**:
```python
def is_real_user_interaction(line_text: str, grammatical: GrammaticalAnalysis) -> bool:
    """
    Detect if line involves real user interaction (not "user" as adjective)

    Real user interactions:
    - "User wants X" - User as subject
    - "to user" - Output to user
    - "from user" - Input from user

    NOT user interactions:
    - "user defined time" - "user" as adjective
    - "user preferences" - "user" as adjective
    """
    # Pattern 1: "to user" or "from user"
    if 'to user' in line_text or 'from user' in line_text:
        return True

    # Pattern 2: Check if "user" is adjective
    adjective_patterns = [
        'user defined', 'user specified', 'user configured',
        'user selected', 'user preferences', 'user settings'
    ]
    for pattern in adjective_patterns:
        if pattern in line_text:
            return False

    # Pattern 3: "User" at start (subject)
    # Remove step ID prefix: "E1 B4-B5 (trigger) User wants..." → "user wants..."
    text_without_prefix = re.sub(
        r'^[bae]\d+(?:\.\d+)?[a-z]?\s*(?:at\s+)?(?:[bae]\d+-[bae]\d+)?\s*(\([^)]+\))?\s*',
        '',
        line_text
    )

    if text_without_prefix.startswith('user '):
        return True

    return False
```

**Example - Real User Interaction**:
```
Input: "E1 B4-B5 (trigger) User wants sugar in coffee"

Processing:
  1. Check "to user" / "from user": No
  2. Check adjective patterns: No
  3. Remove prefix: "user wants sugar in coffee"
  4. Starts with "user ": Yes

Result: True (real user interaction)
```

**Example - NOT User Interaction**:
```
Input: "B1 (trigger) System clock reaches the user defined time of 7:00h"

Processing:
  1. Check "to user" / "from user": No
  2. Check adjective patterns: "user defined" found

Result: False (not a user interaction)
```

#### Step 6.3: Create Actor-Boundary Flows

**UC-Methode Rule**: Actors connect to Boundaries, not directly to Controllers

**Algorithm**:
```python
def create_actor_boundary_flows(
    step: Step,
    actor: RAClass,
    boundary: RAClass
) -> ControlFlow:
    """
    Create flow from Actor to Boundary
    """
    flow = ControlFlow(
        source_step=actor.name,
        target_step=boundary.name,
        source_controller=actor.name,
        target_controller=boundary.name,
        flow_type='actor_interaction',
        rule='Actor-Boundary interaction',
        description=f"Actor {actor.name} interacts with {boundary.name}"
    )

    return flow
```

**Example - B4**:
```
Step: B4 "The system outputs a message to user"
Actor: User
Boundary: MessageDisplayBoundary

Created Flow:
  source: User
  target: MessageDisplayBoundary
  rule: "Actor-Boundary interaction"

Visual:
  User ──→ MessageDisplayBoundary
```

#### Step 6.4: HMI Pattern for User Interactions

**UC-Methode Rule**: User interactions route through HMI Controller

**Pattern**:
```
User Transaction:
  User → Boundary → HMIManager → Material-Controller

Non-User Transaction:
  Boundary → Material-Controller (direct)
```

**Algorithm**:
```python
def create_hmi_flows(
    step: Step,
    boundary: RAClass,
    target_controller: RAClass
) -> List[ControlFlow]:
    """
    Create HMI routing for user transactions

    Flow: Boundary → HMIManager → Target Controller
    """
    flows = []

    # Flow 1: Boundary → HMIManager
    flow1 = ControlFlow(
        source_step=boundary.name,
        target_step='HMIManager',
        source_controller=boundary.name,
        target_controller='HMIManager',
        flow_type='hmi_transaction',
        rule='User-Interaction Boundary to HMI',
        description=f"User interaction {boundary.name} processed by HMIManager"
    )
    flows.append(flow1)

    # Flow 2: HMIManager → Target Controller
    flow2 = ControlFlow(
        source_step='HMIManager',
        target_step=target_controller.name,
        source_controller='HMIManager',
        target_controller=target_controller.name,
        flow_type='hmi_routing',
        rule='HMI to Material-Controller',
        description=f"HMIManager routes to {target_controller.name}"
    )
    flows.append(flow2)

    return flows
```

**Example - E1 (Extension)**:
```
Step: E1 "User wants sugar in coffee"
Action Step: E1.1 "The system adds sugar to the cup"

Step-by-Step:

1. User Interaction Detected:
   is_real_user_interaction("user wants sugar") → True

2. Boundary Created:
   SugarRequestBoundary (User interaction boundary)

3. Action Controller:
   E1.1 → SugarSolidManager

4. HMI Pattern Applied:
   Flow 1: User → SugarRequestBoundary
   Flow 2: SugarRequestBoundary → HMIManager
   Flow 3: HMIManager → SugarSolidManager

Visual:
  User ──→ SugarRequestBoundary ──→ HMIManager ──→ SugarSolidManager
```

#### Step 6.5: Time-Based Trigger Pattern

**UC-Methode Rule**: Time triggers connect to System Controller

**Algorithm**:
```python
def create_timing_flows(
    step: Step,
    time_actor: RAClass,
    boundary: RAClass,
    system_controller: RAClass
) -> List[ControlFlow]:
    """
    Create flows for time-based triggers

    Flow: Time → TimingBoundary → SystemController
    """
    flows = []

    # Flow 1: Time → Boundary
    flow1 = ControlFlow(
        source_step=time_actor.name,
        target_step=boundary.name,
        source_controller=time_actor.name,
        target_controller=boundary.name,
        flow_type='timing_signal',
        rule='Actor-Boundary interaction',
        description=f"{time_actor.name} signals {boundary.name}"
    )
    flows.append(flow1)

    # Flow 2: Boundary → SystemController
    flow2 = ControlFlow(
        source_step=boundary.name,
        target_step=system_controller.name,
        source_controller=boundary.name,
        target_controller=system_controller.name,
        flow_type='timing_activation',
        rule='Timing Boundary to System Controller',
        description=f"{boundary.name} activates {system_controller.name}"
    )
    flows.append(flow2)

    return flows
```

**Example - B1**:
```
Step: B1 "System clock reaches the user defined time of 7:00h"
Actor: Time
Boundary: TimingBoundary
Controller: SystemControlManager

Created Flows:
  Flow 1: Time → TimingBoundary
  Flow 2: TimingBoundary → SystemControlManager

Visual:
  Time ──→ TimingBoundary ──→ SystemControlManager
```

### Output Phase 6

```python
ActorBoundaryFlows(
    actor_to_boundary=[
        ControlFlow(source="Time", target="TimingBoundary", rule="Actor-Boundary"),
        ControlFlow(source="User", target="MessageDisplayBoundary", rule="Actor-Boundary"),
        ControlFlow(source="User", target="ProductDeliveryBoundary", rule="Actor-Boundary"),
        ControlFlow(source="User", target="SugarRequestBoundary", rule="Actor-Boundary"),
    ],

    boundary_to_hmi=[
        ControlFlow(source="TimingBoundary", target="SystemControlManager", rule="Timing"),
        ControlFlow(source="MessageDisplayBoundary", target="HMIManager", rule="HMI"),
        ControlFlow(source="SugarRequestBoundary", target="HMIManager", rule="HMI"),
    ],

    hmi_to_controller=[
        ControlFlow(source="HMIManager", target="SugarSolidManager", rule="HMI Routing"),
        ControlFlow(source="HMIManager", target="CupManager", rule="HMI Routing"),
    ]
)
```

---

## Phase 7: RA Classification & Validation

### UC-Methode Rules Applied

**Validation Rule 1**: All Boundaries must connect to Controllers
**Validation Rule 2**: All Controllers must have at least one function
**Validation Rule 3**: Parallel flows must have START and END nodes
**Validation Rule 4**: No direct Actor-Controller connections
**Validation Rule 5**: Material Controllers must have aggregation state (if applicable)

### Detailed Steps

#### Step 7.1: Validate Boundary Connections

**Algorithm**:
```python
def validate_boundary_connections(
    boundaries: List[RAClass],
    control_flows: List[ControlFlow]
) -> List[ValidationError]:
    """
    Validate all boundaries connect to at least one controller
    """
    errors = []

    for boundary in boundaries:
        # Find outgoing flows from this boundary
        outgoing = [cf for cf in control_flows if cf.source_step == boundary.name]

        if not outgoing:
            error = ValidationError(
                severity='ERROR',
                rule='Validation Rule 1',
                component=boundary.name,
                message=f"Boundary '{boundary.name}' has no outgoing control flows"
            )
            errors.append(error)

    return errors
```

#### Step 7.2: Validate Controller Functions

**Algorithm**:
```python
def validate_controller_functions(
    controllers: List[RAClass],
    material_registry: MaterialControllerRegistry
) -> List[ValidationError]:
    """
    Validate all controllers have at least one function
    """
    errors = []

    for controller in controllers:
        # Get functions from registry
        registry_controller = material_registry.get_controller(controller.name)

        if registry_controller and len(registry_controller.functions) == 0:
            error = ValidationError(
                severity='WARNING',
                rule='Validation Rule 2',
                component=controller.name,
                message=f"Controller '{controller.name}' has no assigned functions"
            )
            errors.append(error)

    return errors
```

#### Step 7.3: Validate Parallel Flow Nodes

**Algorithm**:
```python
def validate_parallel_flows(
    parallel_groups: Dict[int, List[Step]],
    flow_nodes: List[RAClass]
) -> List[ValidationError]:
    """
    Validate each parallel group has START and END nodes
    """
    errors = []

    for group_num, steps in parallel_groups.items():
        # Check for START node
        start_node = f"P{group_num}_START"
        if not any(node.name == start_node for node in flow_nodes):
            error = ValidationError(
                severity='ERROR',
                rule='Validation Rule 3',
                component=f"Group {group_num}",
                message=f"Parallel group {group_num} missing START node"
            )
            errors.append(error)

        # Check for END node
        end_node = f"P{group_num}_END"
        if not any(node.name == end_node for node in flow_nodes):
            error = ValidationError(
                severity='ERROR',
                rule='Validation Rule 3',
                component=f"Group {group_num}",
                message=f"Parallel group {group_num} missing END node"
            )
            errors.append(error)

    return errors
```

#### Step 7.4: Validate Actor-Controller Separation

**Algorithm**:
```python
def validate_actor_controller_separation(
    actors: List[RAClass],
    controllers: List[RAClass],
    control_flows: List[ControlFlow]
) -> List[ValidationError]:
    """
    Validate no direct Actor → Controller connections
    """
    errors = []

    actor_names = {actor.name for actor in actors}
    controller_names = {controller.name for controller in controllers}

    for flow in control_flows:
        if flow.source_step in actor_names and flow.target_step in controller_names:
            error = ValidationError(
                severity='ERROR',
                rule='Validation Rule 4',
                component=f"{flow.source_step} → {flow.target_step}",
                message=f"Direct Actor-Controller connection: {flow.source_step} → {flow.target_step}. Must route via Boundary."
            )
            errors.append(error)

    return errors
```

#### Step 7.5: Validate Material Controller States

**Algorithm**:
```python
def validate_material_states(
    controllers: List[RAClass],
    material_registry: MaterialControllerRegistry
) -> List[ValidationError]:
    """
    Validate material controllers have appropriate aggregation states
    """
    errors = []

    for controller in controllers:
        registry_controller = material_registry.get_controller(controller.name)

        if registry_controller:
            material = registry_controller.material
            state = registry_controller.aggregation_state

            # Check if material should have a state
            if material in ['water', 'coffee', 'milk'] and state is None:
                error = ValidationError(
                    severity='WARNING',
                    rule='Validation Rule 5',
                    component=controller.name,
                    message=f"Material controller '{controller.name}' for '{material}' has no aggregation state"
                )
                errors.append(error)

    return errors
```

### Output Phase 7

```python
ValidationResult(
    errors=[
        # Example errors (if any)
    ],
    warnings=[
        # Example warnings (if any)
    ],
    statistics={
        'total_actors': 2,
        'total_boundaries': 7,
        'total_controllers': 9,
        'total_entities': 15,
        'total_control_flow_nodes': 4,
        'total_control_flows': 45,
        'total_data_flows': 28,
        'parallel_groups': 2,
        'validation_passed': True
    },
    final_ra_diagram={
        'meta': {...},
        'components': {...},
        'relationships': {...}
    }
)
```

---

## Complete Example: Step B2c

Let's trace one complete step through all phases:

### Input Line
```
B2c The system grinds the user defined amount of coffee beans with the user defined grinding degree directly into the filter
```

### Phase 1: Context Analysis
Already completed - Domain: beverage_preparation

### Phase 2: Resource Analysis
Not applicable (not a precondition)

### Phase 3: Interaction Analysis

**Step 3.1: NLP Analysis**
```
spaCy Parse:
  - system (NOUN, nsubj)
  - grinds (VERB, ROOT) ← main verb
  - coffee (NOUN, compound)
  - beans (NOUN, dobj) ← direct object
  - with (ADP, prep)
    └─ degree (NOUN, pobj)
       └─ grinding (NOUN, compound)
  - into (ADP, prep)
    └─ filter (NOUN, pobj)

GrammaticalAnalysis:
  main_verb: "grinds"
  verb_lemma: "grind"
  direct_object: "coffee beans"
  prepositional_objects: [("with", "grinding degree"), ("into", "filter")]
```

**Step 3.2: Classify Verb**
```
Check domain JSON:
  transformation_verbs:
    "grind": "CoffeeBeans -> GroundCoffee"

Result: TRANSFORMATION_VERB
```

**Step 3.3: Transformation Processing**
```
1. Parse transformation: "CoffeeBeans -> GroundCoffee"
2. Determine state from "GroundCoffee":
   - keyword "ground" → solid
3. Extract base material: "coffee"
4. Select controller: CoffeeSolidManager
5. Create entity: GroundCoffee

Output:
  Controller: CoffeeSolidManager (manages coffee, solid state)
  Entity: GroundCoffee (output)
```

### Phase 4: Control Flow Generation

**Parallel Group Detection**
```
Step ID: B2c
Pattern match: B2 (base) + c (suffix)
Group: 2

Other steps in group: B2a, B2b, B2d
```

**Apply Rules**
```
Rule 3: Parallel → Parallel (same group)
  P2_START → CoffeeSolidManager
  CoffeeSolidManager → P2_END
```

### Phase 5: Data Flow Analysis

**USE Relationships**
```
Prep object: ("with", "grinding degree")
Classification: USE
Entity: GrindingDegree

DataFlow:
  GrindingDegree ──use──> CoffeeSolidManager
```

**PROVIDE Relationships**
```
Transformation output: GroundCoffee
Entity: GroundCoffee

DataFlow:
  CoffeeSolidManager ──provide──> GroundCoffee
```

### Phase 6: Actor-Boundary Flows
Not applicable (no actor interaction in this step)

### Phase 7: Validation
```
✓ CoffeeSolidManager has function: grind()
✓ CoffeeSolidManager has aggregation state: solid
✓ Parallel group 2 has P2_START and P2_END
✓ All connections valid
```

### Final Result for B2c

```yaml
Step: B2c
Controller:
  name: CoffeeSolidManager
  type: CONTROLLER
  functions: [grind]
  material: coffee
  state: solid
  parallel_group: 2

Entities:
  - GroundCoffee (output)
  - GrindingDegree (input parameter)

Control Flows:
  - P2_START → CoffeeSolidManager
  - CoffeeSolidManager → P2_END

Data Flows:
  - GrindingDegree ──use──> CoffeeSolidManager
  - CoffeeSolidManager ──provide──> GroundCoffee
```

---

**End of Detailed Analysis Pipeline Documentation**
