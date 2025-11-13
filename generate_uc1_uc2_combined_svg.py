#!/usr/bin/env python3
"""
Generate combined UC1+UC2 SVG diagram using svg_rup_visualizer.py
Loads UC1 and UC2 JSON analyses, merges them, generates combined SVG in new/
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from svg_rup_visualizer import SVGRUPVisualizer

def load_uc_analysis(json_path: str, uc_name: str) -> dict:
    """Load UC analysis JSON"""
    print(f"Loading {uc_name} from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def merge_uc_analyses(uc1_data: dict, uc2_data: dict) -> dict:
    """
    Merge two UC analyses into a combined structure for SVG visualization.
    Preserves component relationships and parallel groups.
    """
    print("\n" + "="*70)
    print("MERGING UC1 AND UC2 ANALYSES")
    print("="*70)

    # Create combined structure based on JSON format 2.0
    combined = {
        "meta": {
            "use_case": "UC1+UC2",
            "domain": "beverage_preparation",
            "capability": f"{uc1_data['meta'].get('capability', 'UC1')} + {uc2_data['meta'].get('capability', 'UC2')}",
            "goal": f"{uc1_data['meta'].get('goal', 'UC1 goal')} AND {uc2_data['meta'].get('goal', 'UC2 goal')}",
            "analysis_engine": "combined_svg_rup_visualizer",
            "generated_at": datetime.now().isoformat(),
            "json_format_version": "2.0_combined"
        },
        "components": {
            "actors": [],
            "controllers": [],
            "boundaries": [],
            "entities": [],
            "control_flow_nodes": []
        },
        "layout": {
            "controller_groups": {},
            "flow_node_groups": {}
        },
        "relationships": {
            "control_flows": [],
            "data_flows": [],
            "parallel_nodes": []
        },
        "context": {},
        "summary": {}
    }

    # Analyze which controllers are shared
    uc1_controllers = {c['name']: c for c in uc1_data['components']['controllers']}
    uc2_controllers = {c['name']: c for c in uc2_data['components']['controllers']}

    shared_controllers = set(uc1_controllers.keys()) & set(uc2_controllers.keys())
    uc1_only = set(uc1_controllers.keys()) - set(uc2_controllers.keys())
    uc2_only = set(uc2_controllers.keys()) - set(uc1_controllers.keys())

    print(f"\nController Classification:")
    print(f"  - Shared: {len(shared_controllers)} {sorted(shared_controllers)}")
    print(f"  - UC1-only: {len(uc1_only)} {sorted(uc1_only)}")
    print(f"  - UC2-only: {len(uc2_only)} {sorted(uc2_only)}")

    # Merge actors (deduplicate)
    seen_actors = set()
    for actor in uc1_data['components']['actors'] + uc2_data['components']['actors']:
        if actor['name'] not in seen_actors:
            combined['components']['actors'].append(actor)
            seen_actors.add(actor['name'])

    # Merge controllers with annotations
    for name, controller in uc1_controllers.items():
        controller_copy = controller.copy()
        if name in shared_controllers:
            controller_copy['shared'] = True
            controller_copy['use_cases'] = ['UC1', 'UC2']
            # Add UC2 functions if different
            uc2_desc = uc2_controllers[name]['description']
            if uc2_desc != controller_copy['description']:
                controller_copy['description'] += f" | UC2: {uc2_desc}"
        else:
            controller_copy['shared'] = False
            controller_copy['use_cases'] = ['UC1']
        combined['components']['controllers'].append(controller_copy)

    # Add UC2-only controllers
    for name, controller in uc2_controllers.items():
        if name not in uc1_controllers:
            controller_copy = controller.copy()
            controller_copy['shared'] = False
            controller_copy['use_cases'] = ['UC2']
            combined['components']['controllers'].append(controller_copy)

    # Merge boundaries (deduplicate by name)
    seen_boundaries = set()
    for boundary in uc1_data['components']['boundaries'] + uc2_data['components']['boundaries']:
        if boundary['name'] not in seen_boundaries:
            combined['components']['boundaries'].append(boundary)
            seen_boundaries.add(boundary['name'])

    # Merge entities (deduplicate by name)
    seen_entities = set()
    for entity in uc1_data['components']['entities'] + uc2_data['components']['entities']:
        if entity['name'] not in seen_entities:
            combined['components']['entities'].append(entity)
            seen_entities.add(entity['name'])

    # Merge control_flow_nodes (parallel nodes like P2_START, P2_END, etc.)
    seen_nodes = set()
    for node in uc1_data['components'].get('control_flow_nodes', []) + uc2_data['components'].get('control_flow_nodes', []):
        node_key = f"{node['name']}_{node.get('type', 'unknown')}"
        if node_key not in seen_nodes:
            combined['components']['control_flow_nodes'].append(node)
            seen_nodes.add(node_key)

    print(f"\nControl Flow Nodes:")
    print(f"  - UC1 nodes: {len(uc1_data['components'].get('control_flow_nodes', []))}")
    print(f"  - UC2 nodes: {len(uc2_data['components'].get('control_flow_nodes', []))}")
    print(f"  - Combined nodes: {len(combined['components']['control_flow_nodes'])}")

    # Merge control flows (from both UCs)
    if 'control_flows' in uc1_data['relationships']:
        combined['relationships']['control_flows'].extend(uc1_data['relationships']['control_flows'])
    if 'control_flows' in uc2_data['relationships']:
        combined['relationships']['control_flows'].extend(uc2_data['relationships']['control_flows'])

    # Merge data flows (from both UCs)
    if 'data_flows' in uc1_data['relationships']:
        combined['relationships']['data_flows'].extend(uc1_data['relationships']['data_flows'])
    if 'data_flows' in uc2_data['relationships']:
        combined['relationships']['data_flows'].extend(uc2_data['relationships']['data_flows'])

    # Merge parallel nodes
    if 'parallel_nodes' in uc1_data['relationships']:
        combined['relationships']['parallel_nodes'].extend(uc1_data['relationships']['parallel_nodes'])
    if 'parallel_nodes' in uc2_data['relationships']:
        combined['relationships']['parallel_nodes'].extend(uc2_data['relationships']['parallel_nodes'])

    # Merge layout information - keep separate groups for UC1 and UC2
    uc1_groups = uc1_data.get('layout', {}).get('controller_groups', {})
    uc2_groups = uc2_data.get('layout', {}).get('controller_groups', {})

    # UC1 groups keep their IDs (0, 2, 3, etc.)
    combined['layout']['controller_groups'] = uc1_groups.copy()

    # UC2 groups get offset IDs to avoid conflicts
    max_group = max([int(k) for k in uc1_groups.keys()] + [0])
    for group_id, controllers in uc2_groups.items():
        new_group_id = str(int(group_id) + max_group + 10)  # Offset by 10 to clearly separate
        combined['layout']['controller_groups'][new_group_id] = controllers

    # Merge flow node groups similarly
    uc1_flow_nodes = uc1_data.get('layout', {}).get('flow_node_groups', {})
    uc2_flow_nodes = uc2_data.get('layout', {}).get('flow_node_groups', {})

    combined['layout']['flow_node_groups'] = uc1_flow_nodes.copy()
    for group_id, nodes in uc2_flow_nodes.items():
        new_group_id = str(int(group_id) + max_group + 10)
        combined['layout']['flow_node_groups'][new_group_id] = nodes

    # Calculate summary
    total_components = (
        len(combined['components']['actors']) +
        len(combined['components']['controllers']) +
        len(combined['components']['boundaries']) +
        len(combined['components']['entities']) +
        len(combined['components']['control_flow_nodes'])
    )

    combined['summary'] = {
        'total_components': total_components,
        'total_actors': len(combined['components']['actors']),
        'total_controllers': len(combined['components']['controllers']),
        'total_boundaries': len(combined['components']['boundaries']),
        'total_entities': len(combined['components']['entities']),
        'total_control_flow_nodes': len(combined['components']['control_flow_nodes']),
        'shared_controllers': len(shared_controllers),
        'uc1_only_controllers': len(uc1_only),
        'uc2_only_controllers': len(uc2_only),
        'total_control_flows': len(combined['relationships']['control_flows']),
        'total_data_flows': len(combined['relationships']['data_flows'])
    }

    combined['meta']['total_ra_classes'] = total_components

    print(f"\nCombined Statistics:")
    print(f"  - Total components: {total_components}")
    print(f"  - Controllers: {len(combined['components']['controllers'])}")
    print(f"  - Boundaries: {len(combined['components']['boundaries'])}")
    print(f"  - Entities: {len(combined['components']['entities'])}")
    print(f"  - Control flow nodes: {len(combined['components']['control_flow_nodes'])}")
    print(f"  - Control flows: {len(combined['relationships']['control_flows'])}")
    print(f"  - Data flows: {len(combined['relationships']['data_flows'])}")

    return combined

def generate_combined_svg(combined_data: dict, output_dir: str = "new") -> tuple:
    """
    Generate combined SVG diagram using svg_rup_visualizer

    Returns:
        Tuple of (json_path, svg_path)
    """
    print("\n" + "="*70)
    print("GENERATING COMBINED SVG DIAGRAM")
    print("="*70)

    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True)

    # Save combined JSON
    json_filename = f"UC1_UC2_Combined_SVG_RA_Analysis.json"
    json_path = f"{output_dir}/{json_filename}"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Combined JSON saved: {json_path}")

    # Generate SVG using svg_rup_visualizer
    svg_filename = f"UC1_UC2_Combined_SVG_RA_Diagram.svg"
    svg_path = f"{output_dir}/{svg_filename}"

    visualizer = SVGRUPVisualizer(canvas_width=1400, canvas_height=2000)
    visualizer.generate_svg(json_path, output_path=svg_path)

    print(f"[OK] Combined SVG diagram generated: {svg_path}")

    return json_path, svg_path

def main():
    """Main execution"""
    print("="*70)
    print("UC1 + UC2 COMBINED SVG RUP DIAGRAM GENERATION")
    print("="*70)

    try:
        # Load UC1 analysis (check both new/multi_uc/ and new/)
        uc1_json = "new/multi_uc/UC1_Structured_RA_Analysis.json"
        if not Path(uc1_json).exists():
            uc1_json = "new/UC1_Structured_RA_Analysis.json"
            if not Path(uc1_json).exists():
                print(f"ERROR: UC1 JSON not found in new/multi_uc/ or new/. Run UC1 analysis first!")
                return 1
        uc1_data = load_uc_analysis(uc1_json, "UC1")

        # Load UC2 analysis (check both new/multi_uc/ and new/)
        uc2_json = "new/multi_uc/UC2_Structured_RA_Analysis.json"
        if not Path(uc2_json).exists():
            uc2_json = "new/UC2_Structured_RA_Analysis.json"
            if not Path(uc2_json).exists():
                print(f"ERROR: UC2 JSON not found in new/multi_uc/ or new/. Run UC2 analysis first!")
                return 1
        uc2_data = load_uc_analysis(uc2_json, "UC2")

        # Merge analyses
        combined_data = merge_uc_analyses(uc1_data, uc2_data)

        # Generate combined SVG in new/multi_uc/ directory
        json_path, svg_path = generate_combined_svg(combined_data, output_dir="new/multi_uc")

        print("\n" + "="*70)
        print("SUCCESS: Combined SVG RUP Diagram Generated")
        print("="*70)
        print(f"Files created in new/:")
        print(f"  - JSON: {json_path}")
        print(f"  - SVG:  {svg_path}")
        print(f"\nFeatures:")
        print(f"  - Shared controllers: {combined_data['summary']['shared_controllers']}")
        print(f"  - Aggregation state-based material controllers")
        print(f"  - Official Wikipedia RUP symbols")
        print(f"  - Complete control and data flows for both UCs")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
