# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## general guidance
Program language is english!
The solution shall be domain independent. Domain spezicific information is stored in the domain json.
The solution is based on NLF usig sapcy large language model.

## Framework Overview

This is the RA-NLF (UC-Methode Robustness Analysis Framework), implementing systematic robustness analysis for automated systems. The framework converts use cases into RUP-compliant robustness analysis diagrams using a 5-phase UC-Methode approach.

## Core Architecture

### Main Components
- **`src/generic_uc_analyzer.py`** - Primary analysis engine (38k+ lines), handles domain-agnostic UC analysis
- **`analyzers/integrated_uc_analyzer.py`** - Complete 5-phase UC-Methode implementation
- **`generators/`** - Diagram generators for single-UC and multi-UC scenarios
- **`domains/`** - JSON-based domain knowledge configurations
- **`Use Case/`** - UC definitions (UC1: Milchkaffee, UC2: Espresso, UC3: Rocket Launch, etc.)

### Analysis Phases
1. **Phase 1**: Context analysis and domain identification
2. **Phase 2**: Resource analysis (Betriebsmittel) with safety/hygiene requirements
3. **Phase 3**: Interaction analysis with controller pattern recognition
4. **Phase 4**: Control flow analysis implementing UC-Methode Rules 1-5
5. **Phase 5**: Data flow analysis with USE/PROVIDE relationships

## Common Commands

### Dependencies Setup
```bash
python -m spacy download en_core_web_md
pip install matplotlib spacy
```

### Primary Analysis Commands
```bash
# Single UC analysis (UC1 by default)
python src/generic_uc_analyzer.py

# Generate RA diagram with data flows
python generators/generate_uc1_ra_diagram_with_dataflows.py

# Multi-UC combined analysis (UC1+UC2)
python generators/generate_uc1_uc2_combined_ra_diagram.py

# 5-phase integrated analysis
python analyzers/integrated_uc_analyzer.py
```

### Testing Commands
```bash
# Multi-UC analysis testing
python tests/test_uc1_uc2.py

# Domain orchestrator testing
python tests/test_domain_orchestrator.py

# Safety/hygiene analysis testing
python tests/test_safety_hygiene_analysis.py
```

## Key Architectural Patterns

### UC-Methode Compliance
- **Actor → Boundary → Controller → Entity** pattern enforcement
- **Control Flow Rules 1-5**: Systematic boundary-controller-entity relationships
- **RUP Stereotypes**: Strict adherence to robustness analysis class types

### Domain-Driven Analysis
- **Domain Detection**: Automatic identification from UC text using keyword matching
- **Verb Classification**: Domain-specific transaction, transformation, and function verbs
- **Knowledge Base**: JSON configurations in `domains/` with safety/hygiene requirements

### Data Flow Semantics
- **USE relationships**: Controllers use input entities (prepositions: "with", "from", "into")
- **PROVIDE relationships**: Controllers provide output entities (prepositions: "to", "for")
- **Transformation patterns**: `Input -> Output` entities (e.g., "CoffeeBeans -> GroundCoffee")

### Domain Orchestrator Pattern
- Each domain gets a `{Domain}DomainOrchestrator` controller
- Implicit coordination of all domain-specific controllers
- Central coordination point for multi-UC scenarios

## Naming Conventions

### Controllers
- **Managers**: `{Function}Manager` (e.g., `BrewingManager`, `GrindingManager`)
- **Supply Controllers**: `{Resource}SupplyController` for resource provisioning
- **Domain Orchestrators**: `{Domain}DomainOrchestrator` for domain coordination

### Entities
- **Domain Objects**: Functional entities (Coffee, Water, Milk)
- **Implementation Elements**: Physical components (Heater, Motor) - flagged with warnings

### Boundaries
- **Purpose-based**: `{Purpose}Boundary` (e.g., `HMIStatusDisplayBoundary`, `ProductDeliveryBoundary`)

## Domain Configuration Structure

Located in `domains/*.json`, each domain configuration includes:
```json
{
  "domain_name": "beverage_preparation",
  "keywords": ["brewing", "coffee", "tea"],
  "verb_classification": {
    "transaction_verbs": {"verbs": {"pour": "transfers liquid"}},
    "transformation_verbs": {"verbs": {"grind": "CoffeeBeans -> GroundCoffee"}},
    "function_verbs": {"verbs": {"heat": "Water -> HotWater"}}
  },
  "safety_requirements": {...},
  "hygiene_requirements": {...}
}
```

## Output Formats and Locations

- **Diagrams**: `output/*.png`, `output/*.svg` - High-resolution RA diagrams
- **Analysis Results**: `Zwischenprodukte/*.json` - Structured intermediate results
- **Generated Diagrams**: Root directory - Final combined diagrams

## Key Features

### Multi-UC Integration
- Shared component detection across use cases
- Combined RA diagrams showing entity relationships between UCs
- Domain orchestrator coordination for complex scenarios

### Operational Materials Framework
- Safety and hygiene analysis for Betriebsstoffe (operational materials)
- Precondition handling for resource availability
- Implementation element detection and warnings

### Advanced Analysis
- **Preposition-based Semantics**: Advanced entity relationship inference
- **Parallel Flow Analysis**: Rule 4 implementation for concurrent operations
- **Controller Pattern Recognition**: Automatic identification of management patterns

## Testing Patterns

Standard test structure:
```python
analyzer = GenericUCAnalyzer(domain_name="beverage_preparation")
verb_analyses, ra_classes = analyzer.analyze_uc_file("Use Case/UC1.txt")
```

Tests focus on:
- Cross-UC component sharing analysis
- Domain orchestrator functionality
- Data flow relationship validation
- UC-Methode rule compliance