# Generic UC Analyzer - Dokumentations-Übersicht

## Generierte Dokumentation

Diese Übersicht zeigt alle erstellten Diagramme und Dokumentationen für den `generic_uc_analyzer.py`.

### 📊 Ablaufdiagramme

1. **Generic_UC_Analyzer_Workflow.svg** 
   - Vollständiges visuelles Ablaufdiagramm
   - Zeigt alle 7 Verarbeitungsphasen
   - Input/Output Datenstrukturen
   - Beispiel-Ergebnisse für UC3

2. **Generic_UC_Analyzer_Improvements.svg**
   - Übersicht der wichtigsten Verbesserungen
   - 3 Haupt-Enhancements hervorgehoben
   - Domain-agnostische Features
   - Ergebnis-Statistiken

### 📋 Textuelle Dokumentation

3. **generic_uc_analyzer_workflow.md**
   - Detaillierte Markdown-Dokumentation
   - Mermaid-Diagramme für jede Phase
   - Technische Implementierungsdetails
   - Datenstruktur-Beschreibungen

4. **workflow_summary.txt**
   - Kompakte Text-Zusammenfassung
   - Input/Processing/Output Übersicht
   - Key Enhancements Liste
   - Beispiel-Ergebnisse

## Haupt-Verbesserungen im Analyzer

### ✅ 1. Enhanced UC-Methode Rules
- **Rule 1**: Actor + Transaction Verb → Boundary generation
- **Rule 2**: Verbesserte Controller-Controller Flows mit Parallel-Handling
- **Rules 3-5**: Vollständige UC-Methode Compliance
- **Parallel Flow Logic**: Korrekte Verlinkung für parallele Steps

### ✅ 2. Multiple Data Flows per Step
- **Enhanced Preposition Semantics**: 
  - Before preposition → USE relationship
  - After preposition → PROVIDE relationship
- **Multiple Entities**: Controller kann mehrere USE und PROVIDE Entities haben
- **Complete Traceability**: Alle Entity-Beziehungen werden erfasst

### ✅ 3. Enhanced CSV Export (12 Columns)
**Ursprünglich (6 Spalten):**
- UC_Schritt, Schritt_Text, RA_Klasse, RA_Typ, Stereotype, Beschreibung

**Neu (6 zusätzliche Spalten):**
- Control_Flow_Source, Control_Flow_Type, Control_Flow_Rule
- Data_Flow_Entity, Data_Flow_Type, Data_Flow_Description

**Verbesserung**: Eine CSV-Zeile pro Datenfluss-Beziehung

## Workflow-Phasen im Detail

### Phase 1: UC File Parsing
```
UC File (.txt) → Parse Steps/Actors/Preconditions → UCStep[], Actor[], Precondition[]
```

### Phase 2: Domain Detection
```
Step Keywords → Domain Matching → Load Domain Config → Verb Classifications
```

### Phase 3: NLP Processing
```
Step Text → Compound Noun Preprocessing → spaCy NLP → Verb Analysis → VerbAnalysis[]
```

### Phase 4: RA Class Generation
```
VerbAnalysis → Generate Controllers → Generate Entities → Generate Boundaries → RAClass[]
```

### Phase 5: Control Flow Analysis
```
RAClass[] → UC-Methode Rules 1-5 → Parallel Detection → ControlFlow[]
```

### Phase 6: Data Flow Analysis
```
VerbAnalysis + RAClass[] → Preposition Semantics → Multiple Flows → DataFlow[]
```

### Phase 7: Output Generation
```
All Analysis Data → JSON Export + Enhanced CSV + RA Diagrams
```

## Beispiel-Ergebnisse (UC3 Rocket Launch)

| Metrik | Wert | Beschreibung |
|--------|------|-------------|
| **RA Classes** | 99 | Generated (Actor, Boundary, Controller, Entity) |
| **Control Flows** | 65 | UC-Methode Rule-compliant flows |
| **Data Flows** | 15 | Preposition-based entity relationships |
| **UC Steps** | 35 | Analyzed (Main + Alternative + Extension flows) |
| **Parallel Patterns** | 5 | Detected (B2a/B2b, B4a/B4b, B5a/B5b/B5c, etc.) |
| **Domains** | 4+ | Supported (rocket_science, beverage_preparation, etc.) |

## Domain-Agnostische Unterstützung

Der Analyzer arbeitet mit beliebigen Domänen:
- **rocket_science** (Raketenstart)
- **beverage_preparation** (Getränkezubereitung)
- **automotive** (Fahrzeugtechnik)
- **nuclear** (Nukleartechnik)
- **robotics** (Robotik)
- **Beliebige neue Domänen** (durch JSON-Konfiguration erweiterbar)

## Technische Features

### Advanced NLP
- Compound Noun Preprocessing (`LaunchWindow`, `FlightProgram`)
- spaCy Integration für syntaktische Analyse
- Domain-spezifische Verb-Klassifikation
- Context-aware Entity-Erkennung

### Multi-UC Support
- Combined RA Diagrams für mehrere Use Cases
- Shared Component Detection
- Domain Orchestrator Pattern
- Integration Views für komplexe Szenarien

### Comprehensive Validation
- UC-Methode Compliance Checking
- Actor Usage Validation
- Implementation Element Warnings
- Best Practice Suggestions

## Verwendung

### Einfache Analyse
```bash
python src/generic_uc_analyzer.py
```

### Multi-UC Analyse
```bash
python analyzers/integrated_uc_analyzer.py
```

### Diagram Generation
```bash
python generators/generate_uc1_ra_diagram_with_dataflows.py
```

## Output-Formate

1. **JSON**: Vollständige Analyse-Metadaten
2. **Enhanced CSV**: 12-Spalten Traceability
3. **RA Diagrams**: RUP-compliant SVG/PNG
4. **Multi-UC Views**: Integration Analysis
5. **Safety Analysis**: Operational Materials Framework

## Nächste Schritte

- Integration mit weiteren Domänen
- Erweiterung der Parallel-Flow Erkennung
- Optimization für große UC-Sammlungen
- Integration mit externen Tools (PlantUML, etc.)

---

**Dokumentation erstellt**: 2024-10-29  
**Analyzer Version**: Enhanced UC-Methode compliant  
**Hauptverbesserungen**: Multiple Data Flows + Enhanced CSV + Parallel Handling