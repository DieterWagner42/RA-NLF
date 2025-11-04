"""
Demo script for RA Visualization Engines
Shows how to use the visualization engines programmatically
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import time

# Import visualization engines
from ra_diagram_engine import RADiagramEngine
from advanced_ra_engine import AdvancedRAEngine
from unified_ra_visualizer import UnifiedRAVisualizer, DiagramStyle, OutputFormat

def demo_basic_engine():
    """Demonstrate basic RA diagram engine"""
    print("\n" + "=" * 60)
    print("BASIC RA DIAGRAM ENGINE DEMO")
    print("=" * 60)
    
    engine = RADiagramEngine(figure_size=(20, 14))
    
    # Find a visualization JSON file
    output_dir = Path("output")
    viz_files = list(output_dir.glob("*_visualization_*.json"))
    
    if not viz_files:
        print("No visualization JSON files found for demo")
        return
    
    json_file = viz_files[0]
    print(f"Using JSON file: {json_file.name}")
    
    # Generate PNG diagram
    start_time = time.time()
    png_path = engine.create_diagram(str(json_file))
    png_time = time.time() - start_time
    
    print(f"✓ PNG diagram created: {Path(png_path).name}")
    print(f"  Generation time: {png_time:.2f} seconds")
    
    # Generate SVG diagram
    start_time = time.time()
    svg_path = engine.create_svg_diagram(str(json_file))
    svg_time = time.time() - start_time
    
    print(f"✓ SVG diagram created: {Path(svg_path).name}")
    print(f"  Generation time: {svg_time:.2f} seconds")
    
    return png_path, svg_path

def demo_advanced_engine():
    """Demonstrate advanced RA diagram engine"""
    print("\n" + "=" * 60)
    print("ADVANCED RA DIAGRAM ENGINE DEMO")
    print("=" * 60)
    
    engine = AdvancedRAEngine(figure_size=(28, 20))
    
    # Find a visualization JSON file
    output_dir = Path("output")
    viz_files = list(output_dir.glob("*_visualization_*.json"))
    
    if not viz_files:
        print("No visualization JSON files found for demo")
        return
    
    json_file = viz_files[0]
    print(f"Using JSON file: {json_file.name}")
    
    # Load and analyze JSON structure
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Parse components and edges for analysis
    components = engine.parse_components(json_data)
    edges = engine.parse_edges(json_data)
    
    print(f"\nJSON Analysis:")
    print(f"  Components: {len(components)}")
    print(f"  Edges: {len(edges)}")
    
    # Count components by type
    type_counts = {}
    element_type_counts = {}
    warning_count = 0
    
    for comp in components:
        comp_type = comp.component_type.value
        elem_type = comp.element_type.value
        
        type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        element_type_counts[elem_type] = element_type_counts.get(elem_type, 0) + 1
        
        if comp.warnings:
            warning_count += 1
    
    print(f"  Component types: {dict(type_counts)}")
    print(f"  Element types: {dict(element_type_counts)}")
    print(f"  Components with warnings: {warning_count}")
    
    # Count edges by type
    edge_type_counts = {}
    control_rule_counts = {}
    
    for edge in edges:
        edge_type = edge.edge_type.value
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        if edge.flow_rule:
            rule = f"Rule {edge.flow_rule}"
            control_rule_counts[rule] = control_rule_counts.get(rule, 0) + 1
    
    print(f"  Edge types: {dict(edge_type_counts)}")
    if control_rule_counts:
        print(f"  UC-Methode rules: {dict(control_rule_counts)}")
    
    # Generate advanced diagram
    start_time = time.time()
    diagram_path = engine.create_advanced_diagram(str(json_file))
    generation_time = time.time() - start_time
    
    print(f"\n✓ Advanced diagram created: {Path(diagram_path).name}")
    print(f"  Generation time: {generation_time:.2f} seconds")
    
    return diagram_path

def demo_unified_interface():
    """Demonstrate unified visualization interface"""
    print("\n" + "=" * 60)
    print("UNIFIED VISUALIZATION INTERFACE DEMO")
    print("=" * 60)
    
    visualizer = UnifiedRAVisualizer()
    
    # Auto-discover and process all visualization files
    print("Auto-discovering visualization JSON files...")
    
    start_time = time.time()
    results = visualizer.auto_discover_and_generate(
        input_dir="output",
        style=DiagramStyle.BOTH,
        format=OutputFormat.PNG
    )
    total_time = time.time() - start_time
    
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    
    # Generate detailed report
    report = visualizer.create_comparison_report(results)
    print("\n" + report)
    
    return results

def demo_json_validation():
    """Demonstrate JSON validation features"""
    print("\n" + "=" * 60)
    print("JSON VALIDATION DEMO")
    print("=" * 60)
    
    visualizer = UnifiedRAVisualizer()
    
    # Find visualization JSON files
    output_dir = Path("output")
    viz_files = list(output_dir.glob("*_visualization_*.json"))
    
    if not viz_files:
        print("No visualization JSON files found for validation demo")
        return
    
    # Validate each file
    validation_results = {}
    
    for json_file in viz_files[:3]:  # Test first 3 files
        print(f"\nValidating: {json_file.name}")
        
        is_valid, issues = visualizer.validate_json_structure(str(json_file))
        validation_results[json_file.name] = (is_valid, issues)
        
        if is_valid:
            print("✓ Valid JSON structure")
        else:
            print("✗ Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
    
    # Summary
    valid_count = sum(1 for is_valid, _ in validation_results.values() if is_valid)
    total_count = len(validation_results)
    
    print(f"\nValidation Summary:")
    print(f"  Valid files: {valid_count}/{total_count}")
    print(f"  Success rate: {(valid_count/total_count*100):.1f}%")
    
    return validation_results

def compare_engines():
    """Compare performance and output of different engines"""
    print("\n" + "=" * 60)
    print("ENGINE COMPARISON")
    print("=" * 60)
    
    # Find a suitable JSON file
    output_dir = Path("output")
    viz_files = list(output_dir.glob("*_visualization_*.json"))
    
    if not viz_files:
        print("No visualization JSON files found for comparison")
        return
    
    json_file = viz_files[0]
    print(f"Comparing engines using: {json_file.name}")
    
    # Basic engine test
    print("\nBasic Engine Performance:")
    basic_engine = RADiagramEngine()
    start_time = time.time()
    basic_png = basic_engine.create_diagram(str(json_file))
    basic_time = time.time() - start_time
    basic_size = Path(basic_png).stat().st_size
    
    print(f"  Generation time: {basic_time:.2f} seconds")
    print(f"  File size: {basic_size:,} bytes")
    print(f"  Output: {Path(basic_png).name}")
    
    # Advanced engine test
    print("\nAdvanced Engine Performance:")
    advanced_engine = AdvancedRAEngine()
    start_time = time.time()
    advanced_png = advanced_engine.create_advanced_diagram(str(json_file))
    advanced_time = time.time() - start_time
    advanced_size = Path(advanced_png).stat().st_size
    
    print(f"  Generation time: {advanced_time:.2f} seconds")
    print(f"  File size: {advanced_size:,} bytes")
    print(f"  Output: {Path(advanced_png).name}")
    
    # Comparison summary
    print(f"\nComparison Summary:")
    print(f"  Speed difference: {abs(advanced_time - basic_time):.2f} seconds")
    print(f"  Size difference: {abs(advanced_size - basic_size):,} bytes")
    
    if advanced_time > basic_time:
        print(f"  Advanced engine is {(advanced_time/basic_time):.1f}x slower (expected due to enhanced features)")
    else:
        print(f"  Advanced engine is {(basic_time/advanced_time):.1f}x faster")
    
    return {
        "basic": {"time": basic_time, "size": basic_size, "path": basic_png},
        "advanced": {"time": advanced_time, "size": advanced_size, "path": advanced_png}
    }

def main():
    """Run complete demonstration of all visualization capabilities"""
    print("RA DIAGRAM VISUALIZATION ENGINES - COMPLETE DEMO")
    print("=" * 80)
    
    # Check if we have visualization JSON files
    output_dir = Path("output")
    if not output_dir.exists():
        print("Error: Output directory not found")
        print("Please run the JSON export functionality first to generate visualization JSON files")
        return
    
    viz_files = list(output_dir.glob("*_visualization_*.json"))
    if not viz_files:
        print("Error: No visualization JSON files found")
        print("Please run the JSON export functionality first to generate visualization JSON files")
        return
    
    print(f"Found {len(viz_files)} visualization JSON files")
    print("Starting comprehensive demonstration...")
    
    # Run all demos
    demos = [
        ("JSON Validation", demo_json_validation),
        ("Basic Engine", demo_basic_engine),
        ("Advanced Engine", demo_advanced_engine),
        ("Engine Comparison", compare_engines),
        ("Unified Interface", demo_unified_interface)
    ]
    
    demo_results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            start_time = time.time()
            result = demo_func()
            demo_time = time.time() - start_time
            demo_results[demo_name] = {"result": result, "time": demo_time}
            print(f"\n{demo_name} completed in {demo_time:.2f} seconds")
        except Exception as e:
            print(f"\nError in {demo_name}: {e}")
            demo_results[demo_name] = {"error": str(e)}
    
    # Final summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    total_time = sum(demo["time"] for demo in demo_results.values() if "time" in demo)
    successful_demos = sum(1 for demo in demo_results.values() if "result" in demo)
    
    print(f"Total demonstration time: {total_time:.2f} seconds")
    print(f"Successful demos: {successful_demos}/{len(demos)}")
    
    print("\nGenerated diagram files can be found in the 'output' directory")
    print("The visualization engines are ready for production use!")


if __name__ == "__main__":
    main()