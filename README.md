# RA-NLF: UC-Methode Robustness Analysis Framework

Dieses Repository implementiert eine systematische Robustheitsanalyse basierend auf der UC-Methode (Use Case Method) für automatisierte Systeme.

## Überblick

Das Framework analysiert Use Cases und generiert automatisch:
- Robustheitsanalyse-Diagramme (RA-Diagramme) 
- Controller-Entity-Zuordnungen
- Datenfluss-Beziehungen (USE/PROVIDE)
- Kontrollfluss-Visualisierungen
- Multi-UC Domain-Integration

## Verzeichnisstruktur

```
RA-NLF/
├── src/                    # Hauptquellcode
│   └── generic_uc_analyzer.py    # Kern-UC-Analyzer
├── generators/             # Diagramm- und Visualisierungsgeneratoren
├── analyzers/              # Spezialisierte Analyzer
├── tests/                  # Test-Dateien
├── docs/                   # Dokumentation
├── output/                 # Generierte Diagramme und Ergebnisse
├── legacy/                 # Veraltete/experimentelle Implementierungen
├── domains/                # Domain-spezifisches Wissen
├── Use Case/               # Use Case Definitionen
├── Zwischenprodukte/       # Analyse-Zwischenergebnisse
└── RA diagrams/            # Referenz-Diagramme
```

## Hauptkomponenten

### Core Analyzer
- **generic_uc_analyzer.py**: Hauptanalyse-Engine mit Datenfluss-Analyse

### Diagramm-Generatoren
- **generate_uc1_ra_diagram_with_dataflows.py**: UC1 RA-Diagramm mit Datenflüssen
- **generate_uc1_uc2_combined_ra_diagram.py**: Kombinierte UC1+UC2 Analyse

### Unterstützte Use Cases
- **UC1**: Milchkaffee zubereiten (Getränkezubereitung)
- **UC2**: Espresso zubereiten (mit Druckkompression)
- **UC3**: Rocket Launch (Raumfahrt)
- **UC4**: Nuclear Shutdown (Nuklearenergie)  
- **UC5**: Robot Assembly (Robotik)

## Features

### UC-Methode Compliance
- Robustheitsanalyse nach UC-Methode Standards
- Actor → Boundary → Controller → Entity Patterns
- Kontrollfluss-Regeln (Rules 1-5)

### Erweiterte Datenfluss-Analyse
- **USE-Beziehungen**: Controller nutzt Input-Entity
- **PROVIDE-Beziehungen**: Controller stellt Output-Entity bereit
- **Präpositionsbasierte Semantik**: "into/in" → USE, "from/of" → USE
- **SupplyController-Pattern**: Automatische Ressourcenbereitstellung

### Domain-spezifisches Wissen
- Getränkezubereitung (beverage_preparation.json)
- Raumfahrt (aerospace.json)
- Nuklearenergie (nuclear.json)
- Robotik (robotics.json)
- Automotive (automotive.json)

## Schnellstart

### UC1 Analyse ausführen
```bash
python src/generic_uc_analyzer.py
```

### RA-Diagramm für UC1 generieren
```bash
python generators/generate_uc1_ra_diagram_with_dataflows.py
```

### Kombinierte UC1+UC2 Analyse
```bash
python generators/generate_uc1_uc2_combined_ra_diagram.py
```

### Multi-UC Test
```bash
python tests/test_uc1_uc2.py
```

## Beispiel-Ergebnisse

### UC1: Milchkaffee RA-Diagramm
- 24 Datenfluss-Beziehungen (13 USE, 11 PROVIDE)
- Vollständige Kontrollfluss-Abdeckung
- Domain Orchestrator Koordination

### UC1+UC2: Kombinierte Analyse  
- 67 einzigartige Komponenten
- 43 geteilte Komponenten zwischen UC1/UC2
- Kompression-Controller für Espresso (UC2)

## Wichtige Erkenntnisse

1. **CompressorManager**: Implementiert die Kompressionsfunktion in UC2
2. **Präpositionssemantik**: "into/in" zeigt Container-Nutzung, nicht Entity-Erzeugung
3. **Domain Orchestrator**: Zentraler Koordinator für Multi-UC Szenarien
4. **SupplyController**: Automatische PROVIDE-Beziehungen für Ressourcenmanagement

## Dokumentation

Siehe `docs/conversation_summary_uc_methode_analysis.md` für detaillierte Entwicklungshistorie und technische Erkenntnisse.

## Domain Knowledge

Das Framework nutzt domain-spezifisches Wissen für:
- Verb-Klassifikation (transaction, transformation, function)
- Entity-Transformations-Regeln  
- Implementierungselement-Warnings
- Controller-Zuordnungs-Heuristiken

## Ausgabeformate

- **PNG/SVG**: Hochauflösende RA-Diagramme
- **JSON**: Strukturierte Analyse-Ergebnisse
- **HTML**: Interaktive Diagramm-Viewer (legacy)