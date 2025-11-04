#!/usr/bin/env python3
"""
Generate RA diagram for UC3 Rocket Launch with Graphviz
Based on UC3_Rocket_Launch_ra_classes.json analysis results
"""

import os
import subprocess
import json

def create_uc3_rocket_launch_graphviz_diagram():
    """Create UC3 Rocket Launch RA diagram using Graphviz"""
    
    # Load the UC3 analysis results - use latest visualization file
    import glob
    pattern = 'output/UC3_Rocket_Launch_visualization_*.json'
    files = glob.glob(pattern)
    if not files:
        print(f"[ERROR] No UC3 visualization files found matching: {pattern}")
        print("Please run UC3 analysis first using: python src/generic_uc_analyzer.py")
        return False
    
    # Use the latest file
    ra_classes_file = sorted(files)[-1]
    print(f"Using analysis file: {ra_classes_file}")
    
    try:
        with open(ra_classes_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            ra_classes = data.get('graph', {}).get('nodes', [])
    except FileNotFoundError:
        print(f"[ERROR] Analysis file not found: {ra_classes_file}")
        print("Please run UC3 analysis first using: python src/generic_uc_analyzer.py")
        return False
    
    # Organize classes by type
    actors = []
    boundaries = []
    controllers = []
    entities = []
    
    for ra_class in ra_classes:
        class_type = ra_class.get('type', '')
        if class_type == 'actor':
            actors.append(ra_class)
        elif class_type == 'boundary':
            boundaries.append(ra_class)
        elif class_type == 'controller':
            controllers.append(ra_class)
        elif class_type == 'entity':
            entities.append(ra_class)
    
    dot_content = f'''
digraph UC3_Rocket_Launch_RA {{
    // Graph settings
    rankdir=TB;
    bgcolor=white;
    fontname="Arial";
    fontsize=14;
    nodesep=0.8;
    ranksep=1.2;
    
    // Title
    label="UC3: Execute Satellite Launch - RA Diagram (UC-Methode)\\nRobustness Analysis for Rocket Launch System";
    labelloc=t;
    labeljust=c;
    
    // Node styles
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=8];
    
    // Actor nodes ({len(actors)} actors)
    subgraph cluster_actors {{
        label="Actors";
        style=filled;
        fillcolor=lightgray;
        bgcolor=lightgray;
        
'''
    
    # Add actors
    for actor in actors:
        name = actor.get('id', actor.get('label', 'Unknown'))
        desc = actor.get('description', '')
        dot_content += f'        {name} [shape=box, style="filled,rounded", fillcolor=lightgray, label="<<actor>>\\n{name}\\n{desc}"];\n'
    
    dot_content += '''    }
    
    // Boundary nodes'''
    
    dot_content += f''' ({len(boundaries)} boundaries)
    subgraph cluster_boundaries {{
        label="Boundaries";
        style=filled;
        fillcolor=lightblue;
        bgcolor=lightblue;
        
'''
    
    # Add boundaries
    for boundary in boundaries:
        name = boundary.get('id', boundary.get('label', 'Unknown'))
        desc = boundary.get('description', '').replace('Boundary for ', '').replace(' supply monitoring and refill alerts', '')
        dot_content += f'        {name} [shape=circle, style=filled, fillcolor=lightblue, label="<<boundary>>\\n{name}\\n{desc}"];\n'
    
    dot_content += '''    }
    
    // Controller nodes'''
    
    dot_content += f''' ({len(controllers)} controllers)
    subgraph cluster_controllers {{
        label="Controllers";
        style=filled;
        fillcolor=lightgreen;
        bgcolor=lightgreen;
        
'''
    
    # Add controllers (identify domain orchestrator)
    for controller in controllers:
        name = controller.get('id', controller.get('label', 'Unknown'))
        desc = controller.get('description', '').replace('Controls ', '').replace(' operation in ', ' - ')
        
        if 'DomainOrchestrator' in name:
            # Special styling for Domain Orchestrator
            dot_content += f'        {name} [shape=circle, style=filled, fillcolor=gold, label="<<controller>>\\n{name}\\n{desc}", penwidth=3];\n'
        else:
            dot_content += f'        {name} [shape=circle, style=filled, fillcolor=lightgreen, label="<<controller>>\\n{name}\\n{desc}"];\n'
    
    dot_content += '''    }
    
    // Entity nodes'''
    
    dot_content += f''' ({len(entities)} entities)
    subgraph cluster_entities {{
        label="Entities";
        style=filled;
        fillcolor=lightyellow;
        bgcolor=lightyellow;
        
'''
    
    # Add entities
    for entity in entities:
        name = entity.get('id', entity.get('label', 'Unknown'))
        desc = entity.get('description', '').replace('Domain entity: ', '').replace('Resource: ', '').replace('Control data: ', '')
        element_type = entity.get('element_type', 'functional')
        
        # Different colors for different entity types
        if element_type == 'control':
            fillcolor = 'lightcyan'
        elif 'supply' in desc.lower() or 'resource' in desc.lower():
            fillcolor = 'lightpink'
        else:
            fillcolor = 'lightyellow'
            
        dot_content += f'        {name} [shape=circle, style=filled, fillcolor={fillcolor}, label="<<entity>>\\n{name}\\n{desc}"];\n'
    
    dot_content += '''    }
    
    // === CONTROL FLOWS (UC-Methode Rules) ===
    
    // Key control flows based on UC3 analysis
    
    // Rule 1: Actor -> Boundary (Launch triggers)
    Timer -> LaunchWindowSupplyBoundary [color=black, style=solid, label="launch window"];
    ExternalTrigger -> RocketIsFueledAndSupplyBoundary [color=black, style=solid];
    ExternalTrigger -> SatelliteSupplyBoundary [color=black, style=solid];
    ExternalTrigger -> WeatherSupplyBoundary [color=black, style=solid];
    
    // Rule 2: Boundary -> Controller (System activation)
    LaunchWindowSupplyBoundary -> WindowManager [color=black, style=solid];
    RocketIsFueledAndSupplyBoundary -> RocketSupplyController [color=black, style=solid];
    SatelliteSupplyBoundary -> SatelliteSupplyController [color=black, style=solid];
    WeatherSupplyBoundary -> WeatherSupplyController [color=black, style=solid];
    
    // Rule 3: Controller -> Controller (Launch sequence coordination)
    WindowManager -> Rocket_ScienceDomainOrchestrator [color=black, style=solid];
    Rocket_ScienceDomainOrchestrator -> CheckManager [color=black, style=solid, label="systems check"];
    Rocket_ScienceDomainOrchestrator -> SequencerManager [color=black, style=solid, label="activate sequencer"];
    Rocket_ScienceDomainOrchestrator -> IgnitionManager [color=black, style=solid, label="ignition sequence"];
    Rocket_ScienceDomainOrchestrator -> TrajectoryManager [color=black, style=solid, label="monitor trajectory"];
    Rocket_ScienceDomainOrchestrator -> SatelliteManager [color=black, style=solid, label="deploy satellite"];
    Rocket_ScienceDomainOrchestrator -> DeploymentManager [color=black, style=solid, label="confirm deployment"];
    
    // Alternative flows controllers
    CheckManager -> A1ConditionManager [color=red, style=dashed, label="system failure"];
    TrajectoryManager -> A2ConditionManager [color=red, style=dashed, label="trajectory deviation"];
    A1ConditionManager -> SequenceManager [color=red, style=solid, label="abort sequence"];
    A1ConditionManager -> SafetyManager [color=red, style=solid, label="safety protocols"];
    A2ConditionManager -> CorrectionManager [color=red, style=solid, label="guidance correction"];
    CorrectionManager -> SequenceManager [color=red, style=dashed, label="abort if failed"];
    A2ConditionManager -> RecoveryManager [color=red, style=solid, label="recovery procedures"];
    
    // Sequential processing flow
    CheckManager -> SequencerManager [color=black, style=solid];
    SequencerManager -> IgnitionManager [color=black, style=solid];
    IgnitionManager -> TrajectoryManager [color=black, style=solid];
    TrajectoryManager -> SatelliteManager [color=black, style=solid];
    SatelliteManager -> DeploymentManager [color=black, style=solid];
    
    // Legend
    subgraph cluster_legend {{
        label="Legend";
        style=filled;
        fillcolor=white;
        bgcolor=white;
        rankdir=LR;
        
        // Legend nodes
        legend_normal [shape=none, label="Normal Flow", fontsize=10];
        legend_alt [shape=none, label="Alternative Flow", fontsize=10, color=red];
        legend_coord [shape=none, label="Domain Coordination", fontsize=10];
        
        // Legend edges
        legend_normal -> legend_coord [color=black, style=solid, label="solid black"];
        legend_coord -> legend_alt [color=red, style=dashed, label="dashed red"];
    }}
    
    // Ranking for better layout
    {{rank=same; Timer; ExternalTrigger;}}
    {{rank=same; CheckManager; SequencerManager; IgnitionManager;}}
    {{rank=same; TrajectoryManager; SatelliteManager; DeploymentManager;}}
}}
'''
    
    # Write DOT file
    dot_filename = 'output/UC3_Rocket_Launch_RA.dot'
    os.makedirs('output', exist_ok=True)
    
    with open(dot_filename, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    
    # Generate PNG and SVG
    png_filename = 'output/UC3_Rocket_Launch_RA.png'
    svg_filename = 'output/UC3_Rocket_Launch_RA.svg'
    
    try:
        # Generate PNG
        subprocess.run(['dot', '-Tpng', dot_filename, '-o', png_filename], 
                      check=True, capture_output=True, text=True)
        print(f"[OK] PNG generated: {png_filename}")
        
        # Generate SVG
        subprocess.run(['dot', '-Tsvg', dot_filename, '-o', svg_filename], 
                      check=True, capture_output=True, text=True)
        print(f"[OK] SVG generated: {svg_filename}")
        
        print(f"\nUC3 Rocket Launch RA Diagram created successfully!")
        print(f"Files saved:")
        print(f"- {png_filename}")
        print(f"- {svg_filename}")
        print(f"- {dot_filename}")
        
        print(f"\nDiagram Statistics:")
        print(f"- {len(actors)} Actors: Mission Control, Launch Sequencer")
        print(f"- {len(boundaries)} Boundaries: Launch conditions and monitoring")
        print(f"- {len(controllers)} Controllers: Launch sequence management")
        print(f"- {len(entities)} Entities: Satellite, trajectory, systems data")
        
        print(f"\nRocket Launch Key Features:")
        print(f"- Launch window management and scheduling")
        print(f"- Systems check and safety protocols")
        print(f"- Engine ignition sequence control")
        print(f"- Trajectory monitoring and guidance")
        print(f"- Satellite deployment and confirmation")
        print(f"- Abort sequences and recovery procedures")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error generating diagram: {e}")
        print(f"Make sure Graphviz is installed: https://graphviz.org/download/")
        return False
    except FileNotFoundError:
        print(f"[ERROR] Graphviz not found. Please install Graphviz: https://graphviz.org/download/")
        return False

if __name__ == "__main__":
    create_uc3_rocket_launch_graphviz_diagram()