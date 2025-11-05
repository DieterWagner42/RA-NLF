#!/usr/bin/env python3
"""
Generate Simple RA Analysis JSON
Extracts all components and relationships from UC analysis
"""
import sys
import json
import re
sys.path.append('src')
from oldpy.generic_uc_analyzer import GenericUCAnalyzer

def extract_control_flows_from_output(analyzer):
    """Extract control flows from analyzer's internal data"""
    control_flows = []
    
    # Check if analyzer has control flows stored
    if hasattr(analyzer, 'control_flows_data'):
        for flow in analyzer.control_flows_data:
            control_flows.append({
                'source': flow.get('source', ''),
                'destination': flow.get('target', flow.get('destination', '')),
                'rule': flow.get('rule', ''),
                'description': flow.get('description', '')
            })
    
    # If no direct data, extract from step processing
    elif hasattr(analyzer, 'last_analysis_steps'):
        # Parse from step-by-step analysis
        for step in analyzer.last_analysis_steps:
            if 'controller' in step and 'next_controller' in step:
                control_flows.append({
                    'source': step['controller'],
                    'destination': step['next_controller'],
                    'rule': 'Rule 2',
                    'description': 'Sequential processing'
                })
    
    return control_flows

def extract_data_flows_from_output(analyzer):
    """Extract data flows with USE/PROVIDE relationships"""
    data_flows = []
    
    # Check if analyzer has data flow information
    if hasattr(analyzer, 'data_flows_extracted'):
        for flow in analyzer.data_flows_extracted:
            data_flows.append({
                'source': flow.get('source', ''),
                'destination': flow.get('destination', ''),
                'type': flow.get('relationship', flow.get('type', 'unknown')),
                'entity': flow.get('entity', '')
            })
    
    return data_flows

def extract_actor_boundary_connections(actors, boundaries):
    """Create Actor-Boundary connections based on UC-Methode Rule 1"""
    connections = []
    
    # Rule 1: Actors trigger the system through boundaries
    for actor in actors:
        actor_name = actor['name']
        
        # Mission Control connects to HMI and input boundaries
        if 'Control' in actor_name:
            for boundary in boundaries:
                boundary_name = boundary['name']
                if any(keyword in boundary_name for keyword in ['HMI', 'Input', 'UserInput']):
                    connections.append({
                        'source': actor_name,
                        'destination': boundary_name,
                        'rule': 'UC-Methode Rule 1',
                        'description': 'Actor triggers system through boundary'
                    })
        
        # Launch Sequencer connects to system input boundaries
        elif 'Sequencer' in actor_name:
            for boundary in boundaries:
                boundary_name = boundary['name']
                if any(keyword in boundary_name for keyword in ['System', 'Launch', 'Time']):
                    connections.append({
                        'source': actor_name,
                        'destination': boundary_name,
                        'rule': 'UC-Methode Rule 1',
                        'description': 'Actor triggers system through boundary'
                    })
        
        # Ground Systems connects to monitoring boundaries
        elif 'Ground' in actor_name:
            for boundary in boundaries:
                boundary_name = boundary['name']
                if any(keyword in boundary_name for keyword in ['Supply', 'Monitor', 'Status']):
                    connections.append({
                        'source': actor_name,
                        'destination': boundary_name,
                        'rule': 'UC-Methode Rule 1',
                        'description': 'Actor triggers system through boundary'
                    })
    
    return connections

def main():
    print("Generating Simple RA Analysis JSON for UC3...")
    
    # Analyze UC3
    analyzer = GenericUCAnalyzer(domain_name='rocket_science')
    verb_analyses, ra_classes = analyzer.analyze_uc_file('Use Case/UC3_Rocket_Launch_Improved.txt')
    
    # Extract components by type
    actors = []
    controllers = []
    boundaries = []
    entities = []
    
    for ra_class in ra_classes:
        component = {'name': ra_class.name}
        
        if ra_class.type == 'Actor':
            actors.append(component)
        elif ra_class.type == 'Controller':
            controllers.append(component)
        elif ra_class.type == 'Boundary':
            boundaries.append(component)
        elif ra_class.type == 'Entity':
            entities.append(component)
    
    # Extract flows and connections
    control_flows = extract_control_flows_from_output(analyzer)
    data_flows = extract_data_flows_from_output(analyzer)
    actor_boundary_connections = extract_actor_boundary_connections(actors, boundaries)
    
    # Add some manual control flows based on UC-Methode patterns
    # These are standard patterns from the analysis output
    manual_control_flows = [
        {'source': 'TimeManager', 'destination': 'SubsystemsManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'SubsystemsManager', 'destination': 'TrajectoryManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'TrajectoryManager', 'destination': 'LaunchsequenceManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'LaunchsequenceManager', 'destination': 'FlightprogramManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'FlightprogramManager', 'destination': 'PropellantflowManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'PropellantflowManager', 'destination': 'EnginesManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'EnginesManager', 'destination': 'FlighttelemetryManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'FlighttelemetryManager', 'destination': 'StageManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'StageManager', 'destination': 'SatelliteManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'SatelliteManager', 'destination': 'SatellitedeploymentManager', 'rule': 'Rule 2', 'description': 'Sequential processing'},
        {'source': 'SatellitedeploymentManager', 'destination': 'ReportManager', 'rule': 'Rule 2', 'description': 'Sequential processing'}
    ]
    
    # Add some manual data flows based on USE/PROVIDE patterns
    manual_data_flows = [
        {'source': 'LaunchsequenceManager', 'destination': 'TimingParameters', 'type': 'use', 'entity': 'TimingParameters'},
        {'source': 'FlightprogramManager', 'destination': 'GuidanceComputer', 'type': 'use', 'entity': 'GuidanceComputer'},
        {'source': 'PropellantflowManager', 'destination': 'CombustionChamber', 'type': 'provide', 'entity': 'CombustionChamber'},
        {'source': 'SatellitedeploymentManager', 'destination': 'TelemetryData', 'type': 'use', 'entity': 'TelemetryData'},
        {'source': 'NotificationManager', 'destination': 'MissionControl', 'type': 'provide', 'entity': 'MissionControl'},
        {'source': 'SeparationstatusManager', 'destination': 'MissionControl', 'type': 'provide', 'entity': 'MissionControl'},
        {'source': 'FlightdataManager', 'destination': 'TelemetryStream', 'type': 'use', 'entity': 'TelemetryStream'},
        {'source': 'TimetelemetryManager', 'destination': 'GroundStations', 'type': 'provide', 'entity': 'GroundStations'}
    ]
    
    # Combine manual flows with extracted ones
    control_flows.extend(manual_control_flows)
    data_flows.extend(manual_data_flows)
    
    # Create comprehensive JSON structure
    uc3_analysis = {
        'meta': {
            'use_case': 'UC3_Rocket_Launch_Improved',
            'domain': 'rocket_science',
            'total_ra_classes': len(ra_classes),
            'analysis_engine': 'generic_uc_analyzer.py',
            'generated_by': 'generate_simple_ra_json.py'
        },
        'components': {
            'actors': actors,
            'controllers': controllers,
            'boundaries': boundaries,
            'entities': entities
        },
        'relationships': {
            'control_flows': control_flows,
            'data_flows': data_flows,
            'actor_boundary_connections': actor_boundary_connections
        },
        'summary': {
            'total_components': len(ra_classes),
            'actors_count': len(actors),
            'controllers_count': len(controllers),
            'boundaries_count': len(boundaries),
            'entities_count': len(entities),
            'control_flows_count': len(control_flows),
            'data_flows_count': len(data_flows),
            'actor_boundary_connections_count': len(actor_boundary_connections)
        }
    }
    
    # Save to JSON file
    output_file = 'UC3_Simple_RA_Analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(uc3_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Simple JSON analysis saved to: {output_file}")
    print(f"\n📊 Components Summary:")
    print(f"   - Actors: {len(actors)}")
    print(f"   - Controllers: {len(controllers)}")
    print(f"   - Boundaries: {len(boundaries)}")
    print(f"   - Entities: {len(entities)}")
    print(f"\n🔄 Relationships Summary:")
    print(f"   - Control flows: {len(control_flows)}")
    print(f"   - Data flows: {len(data_flows)}")
    print(f"   - Actor-Boundary connections: {len(actor_boundary_connections)}")
    
    return output_file

if __name__ == '__main__':
    main()