"""
Unified RA Visualization Interface
Provides easy access to all RA diagram generation capabilities
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import argparse

# Import our visualization engines
from ra_diagram_engine import RADiagramEngine
from advanced_ra_engine import AdvancedRAEngine
from rup_compliant_engine import RUPCompliantEngine
from enhanced_rup_engine import EnhancedRUPEngine
from official_rup_engine import OfficialRUPEngine

class DiagramStyle(Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    RUP_COMPLIANT = "rup"
    ENHANCED_RUP = "enhanced_rup"
    OFFICIAL_RUP = "official_rup"
    ALL = "all"

class OutputFormat(Enum):
    PNG = "png"
    SVG = "svg"
    BOTH = "both"

class UnifiedRAVisualizer:
    """
    Unified interface for all RA diagram generation capabilities
    """
    
    def __init__(self):
        self.basic_engine = RADiagramEngine()
        self.advanced_engine = AdvancedRAEngine()
        self.rup_engine = RUPCompliantEngine()
        self.enhanced_rup_engine = EnhancedRUPEngine()
        self.official_rup_engine = OfficialRUPEngine()
        
    def generate_diagram(self, 
                        json_file_path: str,
                        output_dir: str = "output",
                        style: DiagramStyle = DiagramStyle.ADVANCED,
                        format: OutputFormat = OutputFormat.PNG,
                        custom_output_name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Generate RA diagram(s) from JSON file
        
        Args:
            json_file_path: Path to JSON visualization file
            output_dir: Output directory for generated diagrams
            style: Diagram style (basic, advanced, or both)
            format: Output format (png, svg, or both)
            custom_output_name: Custom output filename (without extension)
            
        Returns:
            Dictionary with generated file paths by style and format
        """
        results = {
            "basic": {"png": [], "svg": []},
            "advanced": {"png": [], "svg": []},
            "rup": {"png": [], "svg": []},
            "enhanced_rup": {"png": [], "svg": []},
            "official_rup": {"png": [], "svg": []}
        }
        
        json_path = Path(json_file_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate base output filename
        if custom_output_name:
            base_name = custom_output_name
        else:
            # Extract UC name from JSON
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                metadata = json_data.get("metadata", {})
                uc_name = metadata.get("uc_name", "Unknown_UC")
                timestamp = metadata.get("analysis_timestamp", "")
                base_name = f"{uc_name}_RA_Diagram_{timestamp}"
            except Exception:
                base_name = json_path.stem
        
        # Generate diagrams based on style preference
        if style in [DiagramStyle.BASIC, DiagramStyle.ALL]:
            try:
                # Basic PNG
                if format in [OutputFormat.PNG, OutputFormat.BOTH]:
                    png_path = str(output_path / f"{base_name}_Basic.png")
                    generated_png = self.basic_engine.create_diagram(json_file_path, png_path)
                    results["basic"]["png"].append(generated_png)
                
                # Basic SVG
                if format in [OutputFormat.SVG, OutputFormat.BOTH]:
                    svg_path = str(output_path / f"{base_name}_Basic.svg")
                    generated_svg = self.basic_engine.create_svg_diagram(json_file_path, svg_path)
                    results["basic"]["svg"].append(generated_svg)
                    
            except Exception as e:
                print(f"Error generating basic diagram: {e}")
        
        if style in [DiagramStyle.ADVANCED, DiagramStyle.ALL]:
            try:
                # Advanced PNG
                png_path = str(output_path / f"{base_name}_Advanced.png")
                generated_png = self.advanced_engine.create_advanced_diagram(json_file_path, png_path)
                results["advanced"]["png"].append(generated_png)
                    
            except Exception as e:
                print(f"Error generating advanced diagram: {e}")
        
        if style in [DiagramStyle.RUP_COMPLIANT, DiagramStyle.ALL]:
            try:
                # RUP-compliant PNG
                png_path = str(output_path / f"{base_name}_RUP.png")
                generated_png = self.rup_engine.create_rup_diagram(json_file_path, png_path)
                results["rup"]["png"].append(generated_png)
                    
            except Exception as e:
                print(f"Error generating RUP diagram: {e}")
        
        if style in [DiagramStyle.ENHANCED_RUP, DiagramStyle.ALL]:
            try:
                # Enhanced RUP PNG
                png_path = str(output_path / f"{base_name}_Enhanced_RUP.png")
                generated_png = self.enhanced_rup_engine.create_enhanced_rup_diagram(json_file_path, png_path)
                results["enhanced_rup"]["png"].append(generated_png)
                    
            except Exception as e:
                print(f"Error generating Enhanced RUP diagram: {e}")
        
        if style in [DiagramStyle.OFFICIAL_RUP, DiagramStyle.ALL]:
            try:
                # Official RUP PNG (Wikipedia standard)
                png_path = str(output_path / f"{base_name}_Official_RUP.png")
                generated_png = self.official_rup_engine.create_official_rup_diagram(json_file_path, png_path)
                results["official_rup"]["png"].append(generated_png)
                    
            except Exception as e:
                print(f"Error generating Official RUP diagram: {e}")
        
        return results

    def generate_from_multiple_json(self,
                                  json_file_paths: List[str],
                                  output_dir: str = "output",
                                  style: DiagramStyle = DiagramStyle.ADVANCED,
                                  format: OutputFormat = OutputFormat.PNG) -> Dict[str, Dict[str, List[str]]]:
        """Generate diagrams from multiple JSON files"""
        all_results = {}
        
        for json_file in json_file_paths:
            json_path = Path(json_file)
            if json_path.exists():
                try:
                    results = self.generate_diagram(json_file, output_dir, style, format)
                    all_results[json_path.name] = results
                    print(f"Generated diagrams for {json_path.name}")
                except Exception as e:
                    print(f"Error processing {json_path.name}: {e}")
                    all_results[json_path.name] = {"error": str(e)}
            else:
                print(f"File not found: {json_file}")
        
        return all_results

    def auto_discover_and_generate(self,
                                 input_dir: str = "output",
                                 style: DiagramStyle = DiagramStyle.ADVANCED,
                                 format: OutputFormat = OutputFormat.PNG) -> Dict[str, Dict[str, List[str]]]:
        """Auto-discover visualization JSON files and generate diagrams"""
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all visualization JSON files
        viz_files = list(input_path.glob("*_visualization_*.json"))
        
        if not viz_files:
            print(f"No visualization JSON files found in {input_dir}")
            return {}
        
        print(f"Found {len(viz_files)} visualization JSON files")
        
        # Generate diagrams for all files
        return self.generate_from_multiple_json([str(f) for f in viz_files], input_dir, style, format)

    def create_comparison_report(self, results: Dict[str, Dict[str, List[str]]]) -> str:
        """Create a summary report of generated diagrams"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("RA DIAGRAM GENERATION REPORT")
        report_lines.append("=" * 60)
        
        total_files = len(results)
        total_diagrams = 0
        successful_files = 0
        
        for filename, file_results in results.items():
            report_lines.append(f"\nFile: {filename}")
            
            if "error" in file_results:
                report_lines.append(f"  Error: {file_results['error']}")
                continue
            
            successful_files += 1
            file_diagram_count = 0
            
            for style_name, style_results in file_results.items():
                if isinstance(style_results, dict):
                    for format_name, file_list in style_results.items():
                        if file_list:
                            count = len(file_list)
                            file_diagram_count += count
                            total_diagrams += count
                            report_lines.append(f"  {style_name.title()} {format_name.upper()}: {count} diagrams")
                            for file_path in file_list:
                                report_lines.append(f"    - {Path(file_path).name}")
            
            if file_diagram_count == 0:
                report_lines.append("  Warning: No diagrams generated")
        
        # Summary
        report_lines.append("\n" + "=" * 60)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 60)
        report_lines.append(f"Total JSON files processed: {total_files}")
        report_lines.append(f"Successful files: {successful_files}")
        report_lines.append(f"Total diagrams generated: {total_diagrams}")
        report_lines.append(f"Success rate: {(successful_files/total_files*100):.1f}%" if total_files > 0 else "N/A")
        
        return "\n".join(report_lines)

    def validate_json_structure(self, json_file_path: str) -> Tuple[bool, List[str]]:
        """Validate that JSON file has required structure for visualization"""
        issues = []
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return False, [f"Cannot load JSON file: {e}"]
        
        # Check for required top-level structures
        if "metadata" not in data:
            issues.append("Missing 'metadata' section")
        
        # Check for component data
        has_graph = "graph" in data and "nodes" in data["graph"]
        has_components = "components" in data and "nodes" in data["components"]
        
        if not (has_graph or has_components):
            issues.append("Missing component data (no 'graph.nodes' or 'components.nodes')")
        
        # Check for edge data
        has_graph_edges = "graph" in data and "edges" in data["graph"]
        has_component_edges = "components" in data and "edges" in data["components"]
        
        if not (has_graph_edges or has_component_edges):
            issues.append("Missing edge data (no 'graph.edges' or 'components.edges')")
        
        # Check component structure
        nodes = []
        if has_graph:
            nodes = data["graph"]["nodes"]
        elif has_components:
            nodes = data["components"]["nodes"]
        
        if not nodes:
            issues.append("No components found in data")
        else:
            required_node_fields = ["id", "label", "type", "stereotype"]
            for i, node in enumerate(nodes[:5]):  # Check first 5 nodes
                for field in required_node_fields:
                    if field not in node:
                        issues.append(f"Node {i} missing required field: {field}")
                        break
        
        return len(issues) == 0, issues


def create_cli_interface():
    """Create command-line interface for the unified visualizer"""
    parser = argparse.ArgumentParser(
        description="Unified RA Diagram Visualization Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_ra_visualizer.py --auto
  python unified_ra_visualizer.py --file output/UC1_visualization.json --style advanced
  python unified_ra_visualizer.py --files output/*.json --format both
  python unified_ra_visualizer.py --validate output/UC1_visualization.json
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--auto", action="store_true",
                           help="Auto-discover and process all visualization JSON files")
    input_group.add_argument("--file", type=str,
                           help="Process single JSON file")
    input_group.add_argument("--files", nargs="+",
                           help="Process multiple JSON files")
    input_group.add_argument("--validate", type=str,
                           help="Validate JSON file structure")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory for generated diagrams")
    parser.add_argument("--style", type=str, choices=["basic", "advanced", "rup", "enhanced_rup", "official_rup", "all"],
                       default="official_rup", help="Diagram style")
    parser.add_argument("--format", type=str, choices=["png", "svg", "both"],
                       default="png", help="Output format")
    parser.add_argument("--custom-name", type=str,
                       help="Custom output filename (without extension)")
    
    # Reporting options
    parser.add_argument("--report", action="store_true",
                       help="Generate detailed report")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    
    return parser


def main():
    """Main CLI interface"""
    parser = create_cli_interface()
    args = parser.parse_args()
    
    visualizer = UnifiedRAVisualizer()
    
    if args.validate:
        # Validate JSON structure
        is_valid, issues = visualizer.validate_json_structure(args.validate)
        print(f"Validation of {args.validate}:")
        if is_valid:
            print("JSON structure is valid for visualization")
        else:
            print("JSON structure issues found:")
            for issue in issues:
                print(f"  - {issue}")
        return
    
    # Convert string enums to enum values
    style = DiagramStyle(args.style)
    format_type = OutputFormat(args.format)
    
    try:
        if args.auto:
            # Auto-discover and process
            if not args.quiet:
                print("Auto-discovering visualization JSON files...")
            results = visualizer.auto_discover_and_generate(
                input_dir=args.output_dir,
                style=style,
                format=format_type
            )
        
        elif args.file:
            # Process single file
            if not args.quiet:
                print(f"Processing single file: {args.file}")
            results = {
                Path(args.file).name: visualizer.generate_diagram(
                    args.file,
                    output_dir=args.output_dir,
                    style=style,
                    format=format_type,
                    custom_output_name=args.custom_name
                )
            }
        
        elif args.files:
            # Process multiple files
            if not args.quiet:
                print(f"Processing {len(args.files)} files...")
            results = visualizer.generate_from_multiple_json(
                args.files,
                output_dir=args.output_dir,
                style=style,
                format=format_type
            )
        
        # Generate and display report
        if args.report or not args.quiet:
            report = visualizer.create_comparison_report(results)
            print("\n" + report)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()