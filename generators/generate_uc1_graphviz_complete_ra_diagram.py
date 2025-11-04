#!/usr/bin/env python3
"""
Generate complete RA diagram for UC1 with Graphviz including control flows and data flows (USE/PROVIDE)
Based on data flows from generate_uc1_ra_diagram_with_dataflows.py
"""

import os
import subprocess

def create_uc1_complete_graphviz_diagram():
    """Create complete UC1 RA diagram with control and data flows using Graphviz"""
    
    dot_content = '''
digraph UC1_RA_Complete {
    // Graph settings
    rankdir=TB;
    bgcolor=white;
    fontname="Arial";
    fontsize=14;
    
    // Title
    label="UC1: Prepare Milk Coffee - Complete RA Diagram (UC-Methode)\\nRobustness Analysis with Control Flows and Data Flows";
    labelloc=t;
    labeljust=c;
    
    // Node styles
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=8];
    
    // Actor nodes (stick figure style)
    subgraph cluster_actors {
        label="Actors";
        style=filled;
        fillcolor=lightgray;
        bgcolor=lightgray;
        
        Timer [shape=box, style="filled,rounded", fillcolor=lightgray, label="<<actor>>\\nTimer"];
        ExternalTrigger [shape=box, style="filled,rounded", fillcolor=lightgray, label="<<actor>>\\nExternalTrigger"];
    }
    
    // Boundary nodes (circle with T)
    subgraph cluster_boundaries {
        label="Boundaries";
        style=filled;
        fillcolor=lightblue;
        bgcolor=lightblue;
        
        TimeTriggerBoundary [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\nTimeTriggerBoundary"];
        WaterSupplyBoundary [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\nWaterSupplyBoundary"];
        CoffeeBeansSupplyBoundary [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\nCoffeeBeansSupplyBoundary"];
        MilkSupplyBoundary [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\nMilkSupplyBoundary"];
        SugarSupplyBoundary [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\nSugarSupplyBoundary"];
        HMIAdditiveInputBoundary [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\nHMIAdditiveInputBoundary"];
        HMIStatusDisplayBoundary [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\nHMIStatusDisplayBoundary"];
        HMIErrorDisplayBoundary [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\nHMIErrorDisplayBoundary"];
        ProductDeliveryBoundary [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\nProductDeliveryBoundary"];
    }
    
    // Controller nodes (circle with arrow)
    subgraph cluster_controllers {
        label="Controllers";
        style=filled;
        fillcolor=lightgreen;
        bgcolor=lightgreen;
        
        // Domain Orchestrator (special styling)
        DomainOrchestrator [shape=circle, style=filled, fillcolor=gold, label="<<controller>>\\nDomainOrchestrator", penwidth=3];
        
        // Main Controllers
        TimeManager [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nTimeManager"];
        HeaterManager [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nHeaterManager"];
        FilterManager [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nFilterManager"];
        AmountManager [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nAmountManager"];
        CupManager [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nCupManager"];
        CoffeeManager [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nCoffeeManager"];
        MilkManager [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nMilkManager"];
        MessageManager [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nMessageManager"];
        HMIController [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nHMIController"];
        SugarManager [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nSugarManager"];
        
        // Supply Controllers
        WaterSupplyController [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nWaterSupplyController"];
        CoffeeBeansSupplyController [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nCoffeeBeansSupplyController"];
        MilkSupplyController [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nMilkSupplyController"];
        SugarSupplyController [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\nSugarSupplyController"];
    }
    
    // Entity nodes (circle with underline)
    subgraph cluster_entities {
        label="Entities";
        style=filled;
        fillcolor=lightyellow;
        bgcolor=lightyellow;
        
        Filter [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nFilter"];
        HotWater [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nHotWater"];
        GroundCoffee [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nGroundCoffee"];
        Amount [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nAmount"];
        Cup [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nCup"];
        Coffee [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nCoffee"];
        Water [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nWater"];
        CoffeeBeans [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nCoffeeBeans"];
        Milk [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nMilk"];
        Sugar [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nSugar"];
        Message [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nMessage"];
        Error [shape=circle, style=filled, fillcolor=lightyellow, label="<<entity>>\\nError"];
    }
    
    // === CONTROL FLOWS (UC-Methode Rules) ===
    
    // Rule 1: Actor -> Boundary
    Timer -> TimeTriggerBoundary [color=black, style=solid, label="triggers"];
    ExternalTrigger -> WaterSupplyBoundary [color=black, style=solid];
    ExternalTrigger -> CoffeeBeansSupplyBoundary [color=black, style=solid];
    ExternalTrigger -> MilkSupplyBoundary [color=black, style=solid];
    ExternalTrigger -> SugarSupplyBoundary [color=black, style=solid];
    ExternalTrigger -> HMIAdditiveInputBoundary [color=black, style=solid];
    
    // Rule 2: Boundary -> Controller
    TimeTriggerBoundary -> TimeManager [color=black, style=solid];
    WaterSupplyBoundary -> WaterSupplyController [color=black, style=solid];
    CoffeeBeansSupplyBoundary -> CoffeeBeansSupplyController [color=black, style=solid];
    MilkSupplyBoundary -> MilkSupplyController [color=black, style=solid];
    SugarSupplyBoundary -> SugarSupplyController [color=black, style=solid];
    HMIAdditiveInputBoundary -> HMIController [color=black, style=solid];
    
    // Rule 3: Controller -> Controller (Domain Orchestrator coordination)
    TimeManager -> DomainOrchestrator [color=black, style=solid];
    DomainOrchestrator -> HeaterManager [color=black, style=solid];
    DomainOrchestrator -> FilterManager [color=black, style=solid];
    DomainOrchestrator -> AmountManager [color=black, style=solid];
    DomainOrchestrator -> CupManager [color=black, style=solid];
    DomainOrchestrator -> CoffeeManager [color=black, style=solid];
    DomainOrchestrator -> MilkManager [color=black, style=solid];
    DomainOrchestrator -> MessageManager [color=black, style=solid];
    DomainOrchestrator -> HMIController [color=black, style=solid];
    DomainOrchestrator -> SugarManager [color=black, style=solid];
    DomainOrchestrator -> WaterSupplyController [color=black, style=solid];
    DomainOrchestrator -> CoffeeBeansSupplyController [color=black, style=solid];
    DomainOrchestrator -> MilkSupplyController [color=black, style=solid];
    DomainOrchestrator -> SugarSupplyController [color=black, style=solid];
    
    // Rule 5: Controller -> Boundary (External outputs)
    HMIController -> HMIStatusDisplayBoundary [color=black, style=solid];
    HMIController -> HMIErrorDisplayBoundary [color=black, style=solid];
    CupManager -> ProductDeliveryBoundary [color=black, style=solid];
    MessageManager -> HMIStatusDisplayBoundary [color=black, style=solid];
    MessageManager -> HMIErrorDisplayBoundary [color=black, style=solid];
    
    // === DATA FLOWS ===
    
    // USE RELATIONSHIPS (Blue dashed lines) - Controller uses Entity
    CoffeeManager -> GroundCoffee [color=blue, style=dashed, label="use"];
    CoffeeManager -> HotWater [color=blue, style=dashed, label="use"];
    CoffeeManager -> Filter [color=blue, style=dashed, label="use"];
    CoffeeManager -> Water [color=blue, style=dashed, label="use"];
    CoffeeManager -> Cup [color=blue, style=dashed, label="use"];
    
    AmountManager -> CoffeeBeans [color=blue, style=dashed, label="use"];
    AmountManager -> Filter [color=blue, style=dashed, label="use"];
    
    MilkManager -> Coffee [color=blue, style=dashed, label="use"];
    MilkManager -> Cup [color=blue, style=dashed, label="use"];
    MilkManager -> Milk [color=blue, style=dashed, label="use"];
    
    SugarManager -> Coffee [color=blue, style=dashed, label="use"];
    SugarManager -> Cup [color=blue, style=dashed, label="use"];
    SugarManager -> Sugar [color=blue, style=dashed, label="use"];
    
    HeaterManager -> Water [color=blue, style=dashed, label="use"];
    MessageManager -> Message [color=blue, style=dashed, label="use"];
    
    // PROVIDE RELATIONSHIPS (Red dashed lines) - Controller provides Entity
    HeaterManager -> HotWater [color=red, style=dashed, label="provide"];
    FilterManager -> Filter [color=red, style=dashed, label="provide"];
    AmountManager -> GroundCoffee [color=red, style=dashed, label="provide"];
    CoffeeManager -> Coffee [color=red, style=dashed, label="provide"];
    MilkManager -> Coffee [color=red, style=dashed, label="provide"];
    CupManager -> Cup [color=red, style=dashed, label="provide"];
    MessageManager -> Message [color=red, style=dashed, label="provide"];
    MessageManager -> Error [color=red, style=dashed, label="provide"];
    HMIController -> Message [color=red, style=dashed, label="provide"];
    SugarManager -> Coffee [color=red, style=dashed, label="provide"];
    
    // Supply Controllers provide their entities
    WaterSupplyController -> Water [color=red, style=dashed, label="provide"];
    CoffeeBeansSupplyController -> CoffeeBeans [color=red, style=dashed, label="provide"];
    MilkSupplyController -> Milk [color=red, style=dashed, label="provide"];
    SugarSupplyController -> Sugar [color=red, style=dashed, label="provide"];
    
    // Legend
    subgraph cluster_legend {
        label="Legend";
        style=filled;
        fillcolor=white;
        bgcolor=white;
        
        // Legend nodes
        legend_control [shape=none, label="Control Flow (UC-Methode)", fontsize=10];
        legend_use [shape=none, label="Data Flow: USE", fontsize=10, color=blue];
        legend_provide [shape=none, label="Data Flow: PROVIDE", fontsize=10, color=red];
        
        // Legend edges
        legend_control -> legend_use [color=black, style=solid, label="solid black"];
        legend_use -> legend_provide [color=blue, style=dashed, label="dashed blue"];
        legend_provide -> legend_control [color=red, style=dashed, label="dashed red"];
    }
}
'''
    
    # Write DOT file
    dot_filename = 'output/UC1_RA_Complete_GraphViz.dot'
    os.makedirs('output', exist_ok=True)
    
    with open(dot_filename, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    
    # Generate PNG and SVG
    png_filename = 'output/UC1_RA_Complete_GraphViz.png'
    svg_filename = 'output/UC1_RA_Complete_GraphViz.svg'
    
    try:
        # Generate PNG
        subprocess.run(['dot', '-Tpng', dot_filename, '-o', png_filename], 
                      check=True, capture_output=True, text=True)
        print(f"[OK] PNG generated: {png_filename}")
        
        # Generate SVG
        subprocess.run(['dot', '-Tsvg', dot_filename, '-o', svg_filename], 
                      check=True, capture_output=True, text=True)
        print(f"[OK] SVG generated: {svg_filename}")
        
        print(f"\nComplete UC1 RA Diagram with Control and Data Flows created successfully!")
        print(f"Files saved:")
        print(f"- {png_filename}")
        print(f"- {svg_filename}")
        print(f"- {dot_filename}")
        
        print(f"\nDiagram Legend:")
        print(f"- Black solid lines: Control flow (UC-Methode rules)")
        print(f"- Blue dashed lines (use): Controller uses Entity as input")
        print(f"- Red dashed lines (provide): Controller provides Entity as output")
        print(f"- Gold circle: Domain Orchestrator (central coordinator)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error generating diagram: {e}")
        print(f"Make sure Graphviz is installed: https://graphviz.org/download/")
        return False
    except FileNotFoundError:
        print(f"[ERROR] Graphviz not found. Please install Graphviz: https://graphviz.org/download/")
        return False

if __name__ == "__main__":
    create_uc1_complete_graphviz_diagram()