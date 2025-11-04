# RA Diagram Visualization Engines

A comprehensive visualization system for generating UC-Methode compliant Robustness Analysis diagrams from JSON output, **without using Graphviz**. Uses matplotlib and advanced layout algorithms for professional-quality diagrams.

## Overview

This visualization system consists of three main engines:

1. **Basic RA Engine** (`ra_diagram_engine.py`) - Clean, efficient diagram generation
2. **Advanced RA Engine** (`advanced_ra_engine.py`) - Enhanced features with UC-Methode compliance
3. **Unified Interface** (`unified_ra_visualizer.py`) - Command-line and programmatic access

## Features

### ✨ UC-Methode Compliant Layout
- **Actor-Boundary-Controller-Entity** positioning following UC-Methode rules
- Automatic left-to-right flow: Actors → Boundaries → Controllers → Entities
- Optimized component spacing and relationship-aware positioning

### 🎨 Enhanced Styling
- **Component Types**: Actors (ellipses), Boundaries (rectangles), Controllers (ellipses), Entities (rectangles)
- **Element Type Variations**: Functional, Control, Container, Implementation elements
- **UC-Methode Rule Colors**: Different colors for Control Flow Rules 1-5
- **Warning Indicators**: Special styling for components with warnings or critical status

### 🔗 Relationship Visualization
- **Data Flows**: USE/PROVIDE relationships with blue coloring
- **Control Flows**: UC-Methode Rules 1-5 with distinct colors and line styles
- **Edge Labels**: Transformations, rule information, and step references
- **Smart Edge Routing**: Optimal connection points and label placement

### 📊 Advanced Features
- **Shared Component Detection**: Special indicators for multi-UC components
- **Safety/Hygiene Integration**: Visual indicators for operational materials
- **Implementation Warnings**: Highlighted implementation elements with suggestions
- **Intelligent Text Wrapping**: Automatic component label formatting

## Installation

### Prerequisites
```bash
pip install matplotlib numpy pathlib
```

### Optional (for networkx features)
```bash
pip install networkx
```

## Quick Start

### Command Line Usage

```bash
# Auto-generate diagrams for all visualization JSON files
python unified_ra_visualizer.py --auto

# Generate specific diagram with advanced styling
python unified_ra_visualizer.py --file output/UC1_visualization.json --style advanced

# Generate both basic and advanced versions in PNG and SVG
python unified_ra_visualizer.py --auto --style both --format both

# Validate JSON structure
python unified_ra_visualizer.py --validate output/UC1_visualization.json
```

### Programmatic Usage

```python
from ra_diagram_engine import RADiagramEngine
from advanced_ra_engine import AdvancedRAEngine
from unified_ra_visualizer import UnifiedRAVisualizer, DiagramStyle, OutputFormat

# Basic engine
basic_engine = RADiagramEngine()
png_path = basic_engine.create_diagram("output/UC1_visualization.json")
svg_path = basic_engine.create_svg_diagram("output/UC1_visualization.json")

# Advanced engine with enhanced features
advanced_engine = AdvancedRAEngine(figure_size=(28, 20))
advanced_path = advanced_engine.create_advanced_diagram("output/UC1_visualization.json")

# Unified interface for batch processing
visualizer = UnifiedRAVisualizer()
results = visualizer.auto_discover_and_generate(
    input_dir="output",
    style=DiagramStyle.ADVANCED,
    format=OutputFormat.PNG
)
```

## Engine Comparison

| Feature | Basic Engine | Advanced Engine |
|---------|-------------|-----------------|
| **Generation Speed** | Fast (~1-2 seconds) | Moderate (~2-4 seconds) |
| **UC-Methode Compliance** | ✓ Basic | ✓✓ Full compliance |
| **Element Type Support** | Basic | All types (Functional, Control, Container, Implementation) |
| **Warning Indicators** | Basic | Advanced (triangles, special borders) |
| **Edge Styling** | Simple | UC-Methode Rule-specific colors |
| **Layout Optimization** | Basic | Relationship-aware positioning |
| **Shared Component Detection** | No | Yes |
| **Safety Integration** | No | Yes |
| **Output Formats** | PNG, SVG | PNG (SVG planned) |

## JSON Input Format

The visualization engines expect JSON files with the following structure:

```json
{
  "metadata": {
    "uc_name": "UC1",
    "domain": "beverage_preparation",
    "framework_version": "UC-Methode RA-NLF"
  },
  "graph": {
    "nodes": [
      {
        "id": "User",
        "label": "User",
        "type": "actor",
        "stereotype": "«actor»",
        "element_type": "functional"
      }
    ],
    "edges": [
      {
        "source": "User", 
        "target": "HMIBoundary",
        "type": "control_flow",
        "flow_rule": 1,
        "label": "Rule 1: Boundary→Controller"
      }
    ]
  },
  "layout": {
    "actors": {"position": "left", "color": "#FFE4B5"},
    "boundaries": {"position": "left-center", "color": "#E0E0E0"},
    "controllers": {"position": "center", "color": "#98FB98"},
    "entities": {"position": "right", "color": "#FFA07A"}
  },
  "styling": {
    "default_styles": { /* component styling */ },
    "special_styling": { /* warning/critical styling */ },
    "warnings": [ /* component warnings */ ]
  }
}
```

## Output Examples

### Generated Diagram Structure
- **Left Column**: Actors (External triggers, users, timers)
- **Left-Center**: Boundaries (HMI, supply monitoring, product delivery)
- **Center-Right**: Controllers (Managers, orchestrators, supply controllers)
- **Right Column**: Entities (Functional objects, data, containers)

### File Naming Convention
- Basic Engine: `{UC_Name}_RA_Diagram_Engine_{timestamp}.png/svg`
- Advanced Engine: `{UC_Name}_Advanced_RA_Diagram_{timestamp}.png`
- Unified Interface: `{UC_Name}_RA_Diagram_{timestamp}_{Style}.{format}`

## Configuration

### Layout Configuration
```python
layout_config = {
    "actor_x": 0.08,           # Actors position (left)
    "boundary_x": 0.28,        # Boundaries position
    "controller_x": 0.58,      # Controllers position  
    "entity_x": 0.88,          # Entities position (right)
    "vertical_spacing": 0.06,   # Space between components
    "component_width": 0.14,    # Component width
    "component_height": 0.05    # Component height
}
```

### Styling Configuration
```python
component_styles = {
    ComponentType.ACTOR: {
        ElementType.FUNCTIONAL: {
            "shape": "ellipse", 
            "facecolor": "#FFE4B5",  # Moccasin
            "edgecolor": "#DAA520"   # Goldenrod
        }
    }
    # ... more component types
}
```

## UC-Methode Rule Compliance

### Control Flow Rules
1. **Rule 1** (Red, Solid): Boundary → Controller
2. **Rule 2** (Orange, Dashed): Controller → Controller
3. **Rule 3** (Red-Orange, Dotted): Sequential Execution
4. **Rule 4** (Dark Orange, Dash-Dot): Parallel Split/Join
5. **Rule 5** (Deep Pink, Solid): Controller → Boundary

### Data Flow Relationships
- **USE** (Blue, Solid): Controllers use input entities
- **PROVIDE** (Royal Blue, Solid): Controllers provide output entities

## Advanced Features

### Multi-UC Integration
- Shared component detection across multiple UCs
- Cross-UC interaction visualization
- Domain orchestrator pattern recognition

### Safety & Hygiene Integration
- Critical material highlighting
- Warning triangle indicators
- Implementation element warnings
- Safety constraint visualization

### Layout Optimization
- Relationship-aware component positioning
- Edge crossing minimization
- Automatic text wrapping and sizing
- Optimal connection point calculation

## Command Line Interface

### Available Commands
```bash
# Auto-discovery
python unified_ra_visualizer.py --auto [options]

# Single file processing
python unified_ra_visualizer.py --file <json_file> [options]

# Multiple files
python unified_ra_visualizer.py --files <file1> <file2> ... [options]

# Validation
python unified_ra_visualizer.py --validate <json_file>
```

### Options
- `--output-dir DIR` - Output directory (default: output)
- `--style {basic,advanced,both}` - Diagram style (default: advanced)
- `--format {png,svg,both}` - Output format (default: png)
- `--custom-name NAME` - Custom output filename
- `--report` - Generate detailed report
- `--quiet` - Suppress progress output

## Performance

### Typical Performance (UC with 30+ components)
- **Basic Engine**: 1-2 seconds
- **Advanced Engine**: 2-4 seconds
- **Batch Processing**: ~1 second per diagram

### Memory Usage
- **Basic Engine**: ~50MB RAM
- **Advanced Engine**: ~80MB RAM
- **Large Diagrams** (100+ components): ~150MB RAM

## Troubleshooting

### Common Issues

1. **"No components found in JSON data"**
   - Check JSON structure: ensure `graph.nodes` or `components.nodes` exists
   - Validate JSON format with `--validate` option

2. **"Invalid edge source/target"**
   - Verify edge references match component IDs exactly
   - Check for typos in component names

3. **"Layout optimization failed"**
   - Reduce component count or increase figure size
   - Check for circular dependencies in control flows

4. **Unicode encoding errors**
   - Ensure UTF-8 encoding for JSON files
   - Use ASCII-compatible component names if needed

### Performance Optimization
```python
# For large diagrams (100+ components)
engine = AdvancedRAEngine(figure_size=(40, 30))  # Larger canvas
engine.layout_config["vertical_spacing"] = 0.04  # Tighter spacing

# For faster generation
engine = RADiagramEngine(figure_size=(20, 14))   # Smaller basic engine
```

## Examples and Demos

### Run Complete Demo
```bash
python demo_visualizer.py
```

This runs comprehensive demonstrations of:
- JSON validation
- Basic engine performance
- Advanced engine features
- Engine comparison
- Unified interface capabilities

### Sample Output
The demo generates multiple diagram types:
- Basic RA diagrams with clean UC-Methode layout
- Advanced diagrams with enhanced styling and rule compliance
- Comparison reports showing performance and feature differences

## Integration with UC-Methode Framework

### JSON Export Integration
This visualization system is designed to work seamlessly with the enhanced `generic_uc_analyzer.py`:

```python
# Generate JSON for visualization
analyzer = GenericUCAnalyzer(domain_name="beverage_preparation")
output_files = analyzer.export_to_json("Use Case/UC1.txt", include_safety_hygiene=True)

# Generate diagrams from JSON
visualizer = UnifiedRAVisualizer()
results = visualizer.generate_diagram(
    output_files["visualization"],
    style=DiagramStyle.ADVANCED
)
```

### System Engineering Workflow
1. **UC Analysis** → `generic_uc_analyzer.py` → JSON export
2. **Visualization** → This engine system → Professional RA diagrams
3. **Integration** → System engineering tools → Further processing

## License and Attribution

Part of the UC-Methode RA-NLF (Robustness Analysis Framework) system.
Generated with advanced matplotlib-based visualization without Graphviz dependencies.

## Contributing

When extending the visualization engines:

1. **Maintain UC-Methode Compliance** - Follow layout and styling rules
2. **Preserve Performance** - Optimize for speed while maintaining quality
3. **Test with Multiple UCs** - Ensure compatibility across different domains
4. **Update Documentation** - Keep README and demos current

## Future Enhancements

### Planned Features
- [ ] Interactive HTML diagrams with zoom/pan
- [ ] SVG support for advanced engine
- [ ] Animation for multi-step UC flows
- [ ] Integration with web-based diagram editors
- [ ] Export to diagram interchange formats (PlantUML, Mermaid)
- [ ] Real-time collaboration features

### Extension Points
- Custom layout algorithms
- Additional styling themes
- Domain-specific visualizations
- Performance monitoring
- Diagram validation rules