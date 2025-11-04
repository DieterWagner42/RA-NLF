#!/usr/bin/env python3
"""
Generate UC1 RA Diagram using Graphviz DOT format
Focus on UC1: Prepare Milk Coffee only
"""

import sys
import os
from pathlib import Path
import subprocess

def generate_uc1_graphviz_diagram():
    """Generate UC1 RA diagram in Graphviz DOT format"""
    
    print("=" * 80)
    print("UC1 GRAPHVIZ RA DIAGRAM GENERATOR")
    print("=" * 80)
    
    # Generate DOT content for UC1
    dot_content = generate_uc1_dot_content()
    
    # Save DOT file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    dot_file = output_dir / "UC1_RA_Diagram_Graphviz.dot"
    with open(dot_file, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    
    print(f"SUCCESS: DOT file saved: {dot_file}")
    
    # Try to generate visual outputs
    try:
        # Generate PNG
        png_file = output_dir / "UC1_RA_Diagram_Graphviz.png"
        subprocess.run(['dot', '-Tpng', str(dot_file), '-o', str(png_file)], check=True)
        
        # Generate SVG  
        svg_file = output_dir / "UC1_RA_Diagram_Graphviz.svg"
        subprocess.run(['dot', '-Tsvg', str(dot_file), '-o', str(svg_file)], check=True)
        
        print(f"SUCCESS: UC1 RA diagrams generated successfully!")
        print(f"   - DOT: {dot_file}")
        print(f"   - PNG: {png_file}")
        print(f"   - SVG: {svg_file}")
        
        return str(png_file)
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Error generating visual diagrams: {e}")
        return None
    except FileNotFoundError:
        print("ERROR: Graphviz not found. Please install Graphviz to generate visual diagrams.")
        print(f"   DOT file saved: {dot_file}")
        return str(dot_file)

def generate_uc1_dot_content():
    """Generate Graphviz DOT content for UC1 RA diagram"""
    
    dot_content = '''digraph UC1_RA_Diagram {
    rankdir=TB;
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=8];
    
    // Graph styling
    graph [bgcolor=white, fontname="Arial", fontsize=14, labelloc=top, 
           label="UC1: Prepare Milk Coffee - RA Diagram (UC-Methode)\\nRobustness Analysis with Betriebsmittel-oriented Controllers"];
    
    // Define node styles
    node [shape=ellipse, style=filled];
    
    // ===== ACTORS =====
    subgraph cluster_actors {
        label="Actors";
        style=rounded;
        bgcolor=lightgray;
        
        "Timer" [shape=box, style=filled, fillcolor=white, label="<<actor>>\\nTimer"];
        "User" [shape=box, style=filled, fillcolor=white, label="<<actor>>\\nUser"];
    }
    
    // ===== BOUNDARIES =====
    subgraph cluster_boundaries {
        label="Boundaries";
        style=rounded;
        bgcolor=lightblue;
        
        "TimeTriggerBoundary" [fillcolor=lightblue, label="TimeTriggerBoundary\\n(Radio Clock Interface)"];
        "HMIInputBoundary" [fillcolor=lightblue, label="HMIInputBoundary\\n(User Input Interface)"];
        "HMIOutputBoundary" [fillcolor=lightblue, label="HMIOutputBoundary\\n(Display Interface)"];
        "ProductDeliveryBoundary" [fillcolor=lightblue, label="ProductDeliveryBoundary\\n(Cup Delivery Interface)"];
        "WaterSupplyBoundary" [fillcolor=lightblue, label="WaterSupplyBoundary\\n(Water Input Interface)"];
        "CoffeeBeansSupplyBoundary" [fillcolor=lightblue, label="CoffeeBeansSupplyBoundary\\n(Beans Input Interface)"];
        "MilkSupplyBoundary" [fillcolor=lightblue, label="MilkSupplyBoundary\\n(Milk Input Interface)"];
        "SugarSupplyBoundary" [fillcolor=lightblue, label="SugarSupplyBoundary\\n(Sugar Input Interface)"];
    }
    
    // ===== CONTROLLERS (Betriebsmittel-oriented) =====
    subgraph cluster_controllers {
        label="Controllers (Betriebsmittel-oriented)";
        style=rounded;
        bgcolor=lightgreen;
        
        // Main Controllers
        "WaterController" [fillcolor=lightgreen, label="WaterController\\nMaterial: Water\\nSteps: B2a, A1.1\\nHeating & Quality"];
        "CoffeeController" [fillcolor=lightgreen, label="CoffeeController\\nMaterial: Coffee/Beans\\nSteps: B2c, B3a\\nGrinding & Brewing"];
        "MilkController" [fillcolor=lightgreen, label="MilkController\\nMaterial: Milk\\nSteps: B3b, A2.1\\nMilk Addition"];
        "CupController" [fillcolor=lightgreen, label="CupController\\nMaterial: Cup\\nSteps: B2d, B5\\nRetrieval & Delivery"];
        "FilterController" [fillcolor=lightgreen, label="FilterController\\nMaterial: Filter\\nSteps: B2b\\nFilter Preparation"];
        "SugarController" [fillcolor=lightgreen, label="SugarController\\nMaterial: Sugar\\nSteps: E1.1\\nSugar Addition"];
        
        // System Controllers
        "SystemController" [fillcolor=yellow, label="SystemController\\nMaterial: System\\nSteps: A3.1, A3.2\\nSystem Management"];
        "TimeController" [fillcolor=yellow, label="TimeController\\nMaterial: Time\\nSteps: B1\\nScheduling"];
        "MessageController" [fillcolor=yellow, label="MessageController\\nMaterial: Messages\\nSteps: B4, A1.2, A2.2\\nCommunication"];
        "HMIController" [fillcolor=yellow, label="HMIController\\nMaterial: Interface\\nSteps: B4, B5, E1\\nUser Interaction"];
    }
    
    // ===== ENTITIES =====
    subgraph cluster_entities {
        label="Entities";
        style=rounded;
        bgcolor=lightyellow;
        
        "Water" [fillcolor=lightyellow, label="Water\\nPrecondition Material"];
        "CoffeeBeans" [fillcolor=lightyellow, label="CoffeeBeans\\nPrecondition Material"];
        "Milk" [fillcolor=lightyellow, label="Milk\\nPrecondition Material"];
        "Sugar" [fillcolor=lightyellow, label="Sugar\\nPrecondition Material"];
        "Cup" [fillcolor=lightyellow, label="Cup\\nContainer"];
        "Filter" [fillcolor=lightyellow, label="Filter\\nBrewing Component"];
        "Coffee" [fillcolor=lightyellow, label="Coffee\\nFinal Product"];
        "Message" [fillcolor=lightyellow, label="Message\\nCommunication"];
        "Error" [fillcolor=lightyellow, label="Error\\nError Information"];
    }
    
    // ===== CONTROL FLOWS =====
    
    // Actor to Boundary flows
    "Timer" -> "TimeTriggerBoundary" [label="7:00h trigger"];
    "User" -> "HMIInputBoundary" [label="sugar request"];
    
    // Boundary to Controller flows
    "TimeTriggerBoundary" -> "TimeController" [label="time signal"];
    "HMIInputBoundary" -> "HMIController" [label="user input"];
    "WaterSupplyBoundary" -> "WaterController" [label="water input"];
    "CoffeeBeansSupplyBoundary" -> "CoffeeController" [label="beans input"];
    "MilkSupplyBoundary" -> "MilkController" [label="milk input"];
    "SugarSupplyBoundary" -> "SugarController" [label="sugar input"];
    
    // Controller coordination flows (main sequence)
    "TimeController" -> "WaterController" [label="start process"];
    "WaterController" -> "FilterController" [label="prepare filter"];
    "FilterController" -> "CoffeeController" [label="grind coffee"];
    "CoffeeController" -> "CupController" [label="retrieve cup"];
    "CupController" -> "CoffeeController" [label="brew coffee"];
    "CoffeeController" -> "MilkController" [label="add milk"];
    "MilkController" -> "MessageController" [label="notify completion"];
    "MessageController" -> "HMIController" [label="display message"];
    "HMIController" -> "CupController" [label="deliver cup"];
    
    // Alternative flow controllers
    "WaterController" -> "SystemController" [label="low water error"];
    "MilkController" -> "SystemController" [label="low milk error"];
    "SystemController" -> "MessageController" [label="error messages"];
    
    // Extension flow
    "HMIController" -> "SugarController" [label="sugar request"];
    "SugarController" -> "CupController" [label="add sugar"];
    
    // Controller to Boundary flows (outputs)
    "HMIController" -> "HMIOutputBoundary" [label="status display"];
    "MessageController" -> "HMIOutputBoundary" [label="messages"];
    "CupController" -> "ProductDeliveryBoundary" [label="deliver cup"];
    
    // Boundary to Actor flows (outputs)
    "HMIOutputBoundary" -> "User" [label="feedback"];
    "ProductDeliveryBoundary" -> "User" [label="coffee delivery"];
    
    // ===== ASSOCIATIONS (Controller-Entity relationships) =====
    // These are associations, not control flows
    "WaterController" -> "Water" [style=dotted, label="manages"];
    "CoffeeController" -> "CoffeeBeans" [style=dotted, label="manages"];
    "CoffeeController" -> "Coffee" [style=dotted, label="produces"];
    "MilkController" -> "Milk" [style=dotted, label="manages"];
    "CupController" -> "Cup" [style=dotted, label="manages"];
    "FilterController" -> "Filter" [style=dotted, label="manages"];
    "SugarController" -> "Sugar" [style=dotted, label="manages"];
    "MessageController" -> "Message" [style=dotted, label="manages"];
    "SystemController" -> "Error" [style=dotted, label="handles"];
    
    // ===== LEGEND =====
    subgraph cluster_legend {
        label="Legend";
        style=rounded;
        bgcolor=white;
        
        "LegendActor" [shape=plaintext, label="Actor (stick figure)", fillcolor=white];
        "LegendBoundary" [fillcolor=lightblue, label="Boundary\\n(circle with T)"];
        "LegendController" [fillcolor=lightgreen, label="Controller\\n(circle with arrow)"];
        "LegendEntity" [fillcolor=lightyellow, label="Entity\\n(circle with line)"];
        
        // Legend connections
        "LegendActor" -> "LegendBoundary" [label="solid line = control flow"];
        "LegendController" -> "LegendEntity" [style=dotted, label="dotted line = association"];
    }
    
    // Layout hints
    {rank=same; "Timer"; "User";}
    {rank=same; "TimeTriggerBoundary"; "HMIInputBoundary"; "WaterSupplyBoundary";}
    {rank=same; "TimeController"; "WaterController"; "CoffeeController";}
    {rank=same; "Water"; "CoffeeBeans"; "Milk";}
}'''
    
    return dot_content

if __name__ == "__main__":
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    
    # Generate UC1 Graphviz RA diagram
    result = generate_uc1_graphviz_diagram()
    
    if result:
        print(f"\nSUCCESS: UC1 Graphviz RA diagram generated successfully!")
        print(f"Diagram file: {result}")
        print(f"\nKey features:")
        print(f"   - Betriebsmittel-oriented controller naming")
        print(f"   - Proper UML RA symbols using Graphviz")
        print(f"   - Complete UC1 flow with alternatives and extensions")
        print(f"   - Clear separation of control flows vs associations")
        print(f"   - Professional diagram layout")
    else:
        print(f"ERROR: Failed to generate UC1 Graphviz RA diagram")