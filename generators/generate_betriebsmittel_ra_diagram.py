#!/usr/bin/env python3
"""
Generate Betriebsmittel-oriented RA Diagram for UC1+UC2
Focuses on operational materials (Betriebsmittel) rather than functions
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from generic_uc_analyzer import GenericUCAnalyzer
import subprocess

def generate_betriebsmittel_ra_diagram():
    """Generate RA diagram with Betriebsmittel-oriented controller naming"""
    
    print("=" * 80)
    print("BETRIEBSMITTEL-ORIENTED RA DIAGRAM GENERATOR")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = GenericUCAnalyzer(domain_name="beverage_preparation")
    
    # Analyze UC1
    print("\n=== ANALYZING UC1 WITH BETRIEBSMITTEL-ORIENTED CONTROLLERS ===")
    uc1_path = "Use Case/UC1.txt"
    
    if not Path(uc1_path).exists():
        print(f"ERROR: UC1 file not found: {uc1_path}")
        return
        
    # Get standard analysis
    verb_analyses1, ra_classes1 = analyzer.analyze_uc_file(uc1_path)
    
    # Apply Betriebsmittel-oriented controller generalization
    betriebsmittel_ra_classes1 = apply_betriebsmittel_generalization(ra_classes1)
    
    # Analyze UC2
    print("\n=== ANALYZING UC2 WITH BETRIEBSMITTEL-ORIENTED CONTROLLERS ===")
    uc2_path = "Use Case/UC2.txt"
    betriebsmittel_ra_classes2 = []
    
    if Path(uc2_path).exists():
        verb_analyses2, ra_classes2 = analyzer.analyze_uc_file(uc2_path)
        betriebsmittel_ra_classes2 = apply_betriebsmittel_generalization(ra_classes2)
    else:
        print("UC2 file not found, creating expected UC2 controllers")
        betriebsmittel_ra_classes2 = create_expected_uc2_controllers()
    
    # Generate combined Graphviz diagram
    print("\n=== GENERATING BETRIEBSMITTEL-ORIENTED GRAPHVIZ DIAGRAM ===")
    dot_content = generate_betriebsmittel_dot_graph(betriebsmittel_ra_classes1, betriebsmittel_ra_classes2)
    
    # Save DOT file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    dot_file = output_dir / "UC1_UC2_Betriebsmittel_RA_Diagram.dot"
    with open(dot_file, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    
    # Generate PNG and SVG
    try:
        # Generate PNG
        png_file = output_dir / "UC1_UC2_Betriebsmittel_RA_Diagram.png"
        subprocess.run(['dot', '-Tpng', str(dot_file), '-o', str(png_file)], check=True)
        
        # Generate SVG  
        svg_file = output_dir / "UC1_UC2_Betriebsmittel_RA_Diagram.svg"
        subprocess.run(['dot', '-Tsvg', str(dot_file), '-o', str(svg_file)], check=True)
        
        print(f"\\n✅ Betriebsmittel-oriented RA diagrams generated successfully!")
        print(f"   - DOT: {dot_file}")
        print(f"   - PNG: {png_file}")
        print(f"   - SVG: {svg_file}")
        
        return str(png_file)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating diagrams: {e}")
        return None
    except FileNotFoundError:
        print("❌ Graphviz not found. Please install Graphviz to generate visual diagrams.")
        print(f"   DOT file saved: {dot_file}")
        return str(dot_file)

def apply_betriebsmittel_generalization(ra_classes):
    """Apply Betriebsmittel-oriented generalization to controllers"""
    
    # Betriebsmittel-oriented generalization rules
    betriebsmittel_mapping = {
        # Original Controller -> Betriebsmittel Controller
        "MilkManager": "MilkController",
        "HeaterManager": "WaterController",  # Heater manages water heating
        "CoffeeManager": "CoffeeController", 
        "FilterManager": "FilterController",
        "AmountManager": "CoffeeController",  # Amount relates to coffee grinding
        "CupManager": "CupController",
        "SugarManager": "SugarController",
        "WaterSupplyController": "WaterController",
        "MilkSupplyController": "MilkController", 
        "CoffeeBeansSupplyController": "CoffeeController",
        "SugarSupplyController": "SugarController",
        # Keep some as-is
        "HMIController": "HMIController",
        "MessageManager": "MessageController",
        "TimeManager": "TimeController",
        "AdditionManager": "MilkController",  # Addition relates to milk
        "A1ConditionManager": "WaterController",  # A1 is water condition
        "A2ConditionManager": "MilkController",   # A2 is milk condition
        "ActionsManager": "SystemController",
        "Manager": "SystemController",
        "UserRequestManager": "HMIController"
    }
    
    # Apply generalization
    generalized_classes = []
    controller_functions = {}  # Track functions per controller
    
    for ra_class in ra_classes:
        if ra_class.type == "Controller":
            # Get new name based on Betriebsmittel
            original_name = ra_class.name
            new_name = betriebsmittel_mapping.get(original_name, original_name)
            
            # Track functions for this controller
            if new_name not in controller_functions:
                controller_functions[new_name] = {
                    'steps': [],
                    'descriptions': [],
                    'betriebsmittel': extract_betriebsmittel_from_name(new_name)
                }
            
            controller_functions[new_name]['steps'].extend(ra_class.step_references)
            controller_functions[new_name]['descriptions'].append(ra_class.description)
            
            print(f"  {original_name} -> {new_name}")
        else:
            # Keep other classes as-is
            generalized_classes.append(ra_class)
    
    # Create consolidated controllers
    for controller_name, functions in controller_functions.items():
        consolidated_controller = type('RAClass', (), {
            'name': controller_name,
            'type': 'Controller',
            'stereotype': '«control»',
            'element_type': 'functional',
            'step_references': list(set(functions['steps'])),  # Remove duplicates
            'description': f"Manages {functions['betriebsmittel']} operations: {', '.join(set(functions['descriptions']))}",
            'betriebsmittel': functions['betriebsmittel']
        })()
        
        generalized_classes.append(consolidated_controller)
    
    return generalized_classes

def extract_betriebsmittel_from_name(controller_name):
    """Extract the Betriebsmittel (operational material) from controller name"""
    betriebsmittel_map = {
        'MilkController': 'Milk',
        'WaterController': 'Water', 
        'CoffeeController': 'Coffee/CoffeeBeans',
        'FilterController': 'Filter',
        'CupController': 'Cup/Container',
        'SugarController': 'Sugar',
        'HMIController': 'User Interface',
        'MessageController': 'Messages/Communication',
        'TimeController': 'Time/Scheduling',
        'SystemController': 'System Resources',
        'PressureController': 'Pressure/Compression'
    }
    return betriebsmittel_map.get(controller_name, controller_name.replace('Controller', ''))

def create_expected_uc2_controllers():
    """Create expected UC2 controllers based on Espresso preparation"""
    
    uc2_controllers = []
    
    # UC2-specific controllers
    pressure_controller = type('RAClass', (), {
        'name': 'PressureController',
        'type': 'Controller', 
        'stereotype': '«control»',
        'element_type': 'functional',
        'step_references': ['B2c_UC2'],
        'description': 'Manages water pressure and compression for espresso',
        'betriebsmittel': 'Pressure/Compression',
        'uc_source': 'UC2'
    })()
    
    uc2_controllers.append(pressure_controller)
    
    return uc2_controllers

def generate_betriebsmittel_dot_graph(uc1_classes, uc2_classes):
    """Generate Graphviz DOT content for Betriebsmittel-oriented RA diagram"""
    
    dot_content = '''digraph BetriebsmittelRA {
    rankdir=TB;
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=8];
    
    // Graph styling
    graph [bgcolor=white, fontname="Arial", fontsize=12, labelloc=top, 
           label="UC1 + UC2: Betriebsmittel-oriented RA Diagram (UC-Methode)\\nOperational Materials Take Priority"];
    
    // Subgraph for UC1
    subgraph cluster_uc1 {
        label="UC1: Milk Coffee (Betriebsmittel-oriented)";
        style=rounded;
        bgcolor=lightblue;
        
'''
    
    # Add UC1 controllers grouped by Betriebsmittel
    betriebsmittel_groups = group_by_betriebsmittel(uc1_classes)
    
    for betriebsmittel, controllers in betriebsmittel_groups.items():
        if controllers:
            dot_content += f'        // {betriebsmittel} Controllers\\n'
            for controller in controllers:
                if controller.type == "Controller":
                    functions = "\\n".join(controller.step_references[:3])  # Show first 3 steps
                    dot_content += f'        "{controller.name}" [shape=ellipse, style=filled, fillcolor=lightgreen, '
                    dot_content += f'label="{controller.name}\\n{betriebsmittel}\\n{functions}"];\\n'
    
    dot_content += '''    }
    
    // Subgraph for UC2
    subgraph cluster_uc2 {
        label="UC2: Espresso (Betriebsmittel-oriented)";
        style=rounded;
        bgcolor=lightcoral;
        
'''
    
    # Add UC2 controllers
    for controller in uc2_classes:
        if controller.type == "Controller":
            functions = "\\n".join(controller.step_references)
            dot_content += f'        "{controller.name}" [shape=ellipse, style=filled, fillcolor=orange, '
            dot_content += f'label="{controller.name}\\n{controller.betriebsmittel}\\n{functions}"];\\n'
    
    dot_content += '''    }
    
    // Shared Controllers
    subgraph cluster_shared {
        label="Shared Controllers (Both UC1 & UC2)";
        style=rounded; 
        bgcolor=lightyellow;
        
        "WaterController" [shape=ellipse, style=filled, fillcolor=yellow, 
                          label="WaterController\\nWater\\nHeating, Quality, Supply"];
        "CoffeeController" [shape=ellipse, style=filled, fillcolor=yellow,
                           label="CoffeeController\\nCoffee/Beans\\nGrinding, Brewing, Quality"];
        "CupController" [shape=ellipse, style=filled, fillcolor=yellow,
                        label="CupController\\nCup/Container\\nRetrieval, Positioning, Delivery"];
        "HMIController" [shape=ellipse, style=filled, fillcolor=yellow,
                        label="HMIController\\nUser Interface\\nInput, Output, Interaction"];
    }
    
    // Key Betriebsmittel relationships
    "MilkController" -> "CupController" [label="add milk to", color=blue];
    "CoffeeController" -> "CupController" [label="brew into", color=brown];
    "WaterController" -> "CoffeeController" [label="provide heated water", color=cyan];
    "PressureController" -> "WaterController" [label="compress water", color=red, style=bold];
    
    // Legend
    subgraph cluster_legend {
        label="Betriebsmittel Priority Principle";
        style=rounded;
        bgcolor=white;
        
        "Legend" [shape=note, style=filled, fillcolor=lightyellow,
                 label="Controller Naming:\\n• MilkController (not StorageController)\\n• WaterController (not TemperatureController)\\n• CoffeeController (not ProcessController)\\n\\nBetriebsmittel > Functions"];
    }
    
}'''
    
    return dot_content

def group_by_betriebsmittel(ra_classes):
    """Group controllers by their Betriebsmittel (operational material)"""
    
    groups = {
        'Milk': [],
        'Water': [], 
        'Coffee': [],
        'Sugar': [],
        'Cup': [],
        'Filter': [],
        'System': [],
        'Interface': []
    }
    
    for ra_class in ra_classes:
        if ra_class.type == "Controller":
            if hasattr(ra_class, 'betriebsmittel'):
                betriebsmittel = ra_class.betriebsmittel
                if 'Milk' in betriebsmittel:
                    groups['Milk'].append(ra_class)
                elif 'Water' in betriebsmittel:
                    groups['Water'].append(ra_class)
                elif 'Coffee' in betriebsmittel:
                    groups['Coffee'].append(ra_class)
                elif 'Sugar' in betriebsmittel:
                    groups['Sugar'].append(ra_class)
                elif 'Cup' in betriebsmittel:
                    groups['Cup'].append(ra_class)
                elif 'Filter' in betriebsmittel:
                    groups['Filter'].append(ra_class)
                elif 'Interface' in betriebsmittel:
                    groups['Interface'].append(ra_class)
                else:
                    groups['System'].append(ra_class)
    
    return groups

if __name__ == "__main__":
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    
    # Generate Betriebsmittel-oriented RA diagram
    result = generate_betriebsmittel_ra_diagram()
    
    if result:
        print(f"\\n🎉 Betriebsmittel-oriented RA diagram generated successfully!")
        print(f"📊 Diagram file: {result}")
        print(f"\\n✅ Key improvements:")
        print(f"   • MilkController manages all milk operations (including cooling)")
        print(f"   • WaterController manages all water operations (including heating)")  
        print(f"   • CoffeeController manages all coffee operations")
        print(f"   • Controller names reflect Betriebsmittel, not functions")
        print(f"   • Clear path to functional architecture")
    else:
        print(f"❌ Failed to generate Betriebsmittel-oriented RA diagram")