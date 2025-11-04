#!/usr/bin/env python3
"""
Enhanced UC Analyzer with JSON Configuration
Supports 1-n UC analysis with automatic file organization
Command line: python enhanced_uc_analyzer.py [config_file]
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append('src')

from structured_uc_analyzer import StructuredUCAnalyzer
from official_rup_engine import OfficialRUPEngine

class EnhancedUCAnalyzer:
    """Enhanced UC Analyzer with JSON configuration support"""
    
    def __init__(self, config_file="uc_config.json"):
        self.config_file = config_file
        self.config = {}
        self.analysis_results = []
        
    def load_config(self):
        """Load UC configuration from JSON file"""
        if not os.path.exists(self.config_file):
            print(f"ERROR: Config file not found: {self.config_file}")
            return False
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            print(f"Loaded configuration: {self.config_file}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            return False
    
    def backup_and_organize_files(self):
        """Backup current 'new' files to 'old' and clean 'new'"""
        print("\nOrganizing analysis files...")
        
        # Ensure directories exist
        os.makedirs("new", exist_ok=True)
        os.makedirs("old", exist_ok=True)
        
        # Move everything from 'new' to 'old' with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join("old", f"backup_{timestamp}")
        
        new_files = list(Path("new").glob("*"))
        if new_files:
            os.makedirs(backup_dir, exist_ok=True)
            print(f"  Backing up {len(new_files)} files to: {backup_dir}")
            
            for file_path in new_files:
                if file_path.is_file():
                    dest_path = os.path.join(backup_dir, file_path.name)
                    shutil.move(str(file_path), dest_path)
                    print(f"    Moved: {file_path.name}")
        else:
            print("  No files to backup in 'new' folder")
    
    def get_enabled_use_cases(self):
        """Get list of enabled use cases from config"""
        if "use_cases" not in self.config:
            print("ERROR: No use_cases section in config")
            return []
        
        enabled_ucs = [uc for uc in self.config["use_cases"] if uc.get("enabled", False)]
        
        # Sort by priority
        enabled_ucs.sort(key=lambda x: x.get("priority", 999))
        
        return enabled_ucs
    
    def validate_use_case(self, uc_config):
        """Validate that UC file exists and is accessible"""
        uc_file = uc_config.get("file", "")
        
        if not uc_file:
            print(f"  ERROR: No file specified for {uc_config.get('id', 'Unknown')}")
            return False
        
        if not os.path.exists(uc_file):
            print(f"  ERROR: UC file not found: {uc_file}")
            return False
        
        return True
    
    def analyze_single_uc(self, uc_config):
        """Analyze a single UC"""
        uc_id = uc_config.get("id", "Unknown")
        uc_name = uc_config.get("name", "Unknown")
        uc_file = uc_config.get("file", "")
        
        print(f"\n{'='*60}")
        print(f"Analyzing {uc_id}: {uc_name}")
        print(f"File: {uc_file}")
        print(f"{'='*60}")
        
        if not self.validate_use_case(uc_config):
            return None
        
        try:
            # Get domain from config
            domain = self.config.get("analysis_config", {}).get("domain", "beverage_preparation")
            
            # Initialize analyzer
            analyzer = StructuredUCAnalyzer(domain)
            
            # Perform analysis
            line_analyses, output_json = analyzer.analyze_uc_file(uc_file)
            
            print(f"\n{uc_id} Analysis completed!")
            print(f"  JSON: {output_json}")
            
            # Generate diagram if enabled
            diagram_path = None
            if self.config.get("analysis_options", {}).get("generate_diagrams", True):
                diagram_path = self.generate_uc_diagram(output_json, uc_id)
            
            # Store results
            result = {
                "uc_id": uc_id,
                "uc_name": uc_name,
                "uc_file": uc_file,
                "json_file": output_json,
                "diagram_file": diagram_path,
                "line_analyses": line_analyses,
                "analyzer": analyzer
            }
            
            return result
            
        except Exception as e:
            print(f"ERROR: Failed to analyze {uc_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_uc_diagram(self, json_file, uc_id):
        """Generate RA diagram for UC"""
        if not os.path.exists(json_file):
            print(f"  WARNING: JSON file not found for diagram: {json_file}")
            return None
        
        try:
            print(f"  Generating {uc_id} RA diagram...")
            
            # Initialize RUP engine
            engine = OfficialRUPEngine(figure_size=(24, 20))
            
            # Generate diagram
            diagram_path = engine.create_official_rup_diagram_from_json(json_file)
            
            print(f"  Diagram: {diagram_path}")
            return diagram_path
            
        except Exception as e:
            print(f"  ERROR: Failed to generate diagram for {uc_id}: {e}")
            return None
    
    def move_results_to_new(self):
        """Move all analysis results to 'new' folder"""
        print(f"\nMoving analysis results to 'new' folder...")
        
        # Define patterns for analysis files
        analysis_patterns = [
            "*_Structured_RA_Analysis.json",
            "*_Structured_UC_Steps_RA_Classes.csv",
            "*_RA_Diagram_Official_RUP.png"
        ]
        
        moved_count = 0
        for pattern in analysis_patterns:
            files = list(Path(".").glob(pattern))
            for file_path in files:
                if file_path.is_file():
                    dest_path = Path("new") / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    print(f"  Moved: {file_path.name}")
                    moved_count += 1
        
        print(f"  Total files moved: {moved_count}")
        return moved_count
    
    def generate_analysis_summary(self):
        """Generate summary of all analyzed UCs"""
        if not self.analysis_results:
            return
        
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        successful_analyses = [r for r in self.analysis_results if r is not None]
        failed_count = len(self.analysis_results) - len(successful_analyses)
        
        print(f"Total UCs processed: {len(self.analysis_results)}")
        print(f"Successful analyses: {len(successful_analyses)}")
        print(f"Failed analyses: {failed_count}")
        
        if successful_analyses:
            print(f"\nSuccessful UCs:")
            for result in successful_analyses:
                print(f"  ✓ {result['uc_id']}: {result['uc_name']}")
                print(f"    JSON: {result['json_file']}")
                if result['diagram_file']:
                    print(f"    Diagram: {result['diagram_file']}")
        
        print(f"\nGenerated Features:")
        print(f"  ✓ Enhanced generative context system")
        print(f"  ✓ Operational materials with safety/hygiene classification")
        print(f"  ✓ Universal operational materials framework")
        print(f"  ✓ No hardcoded contexts - all from domain JSON")
        print(f"  ✓ Automatic file organization")
        
        # Generate combined summary file
        self.save_analysis_summary(successful_analyses)
    
    def save_analysis_summary(self, successful_analyses):
        """Save analysis summary to JSON file"""
        summary = {
            "analysis_session": {
                "timestamp": datetime.now().isoformat(),
                "config_file": self.config_file,
                "domain": self.config.get("analysis_config", {}).get("domain", "unknown"),
                "total_ucs": len(successful_analyses)
            },
            "analyzed_ucs": []
        }
        
        for result in successful_analyses:
            uc_summary = {
                "uc_id": result['uc_id'],
                "uc_name": result['uc_name'],
                "uc_file": result['uc_file'],
                "output_files": {
                    "json": result['json_file'],
                    "diagram": result['diagram_file']
                }
            }
            
            # Add RA class counts if available
            if result['analyzer'] and hasattr(result['analyzer'], 'generated_contexts'):
                contexts = result['analyzer'].generated_contexts
                uc_summary["generative_contexts"] = {
                    "total_contexts": sum(len(c) for c in contexts.values()),
                    "steps_with_contexts": len(contexts)
                }
            
            summary["analyzed_ucs"].append(uc_summary)
        
        summary_file = "new/analysis_session_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnalysis summary saved: {summary_file}")
    
    def run_analysis(self):
        """Run the complete analysis process"""
        print("Enhanced UC Analyzer with JSON Configuration")
        print("=" * 60)
        
        # Step 1: Load configuration
        if not self.load_config():
            return False
        
        # Step 2: Show configuration
        enabled_ucs = self.get_enabled_use_cases()
        if not enabled_ucs:
            print("ERROR: No enabled use cases found in configuration")
            return False
        
        print(f"\nEnabled UCs for analysis: {len(enabled_ucs)}")
        for uc in enabled_ucs:
            print(f"  {uc['id']}: {uc['name']} ({uc['file']})")
        
        # Step 3: Backup and organize files
        if self.config.get("analysis_options", {}).get("auto_organize_files", True):
            self.backup_and_organize_files()
        
        # Step 4: Analyze each enabled UC
        print(f"\nStarting analysis of {len(enabled_ucs)} use cases...")
        
        for uc_config in enabled_ucs:
            result = self.analyze_single_uc(uc_config)
            self.analysis_results.append(result)
        
        # Step 5: Move results to 'new' folder
        if self.config.get("analysis_options", {}).get("auto_organize_files", True):
            self.move_results_to_new()
        
        # Step 6: Generate summary
        self.generate_analysis_summary()
        
        return True

def create_sample_configs():
    """Create sample configuration files for different scenarios"""
    
    # Single UC config (UC1 only)
    uc1_config = {
        "analysis_config": {
            "version": "1.0",
            "description": "Single UC Analysis - UC1 only",
            "domain": "beverage_preparation",
            "output_format": ["json", "csv", "diagram"]
        },
        "use_cases": [
            {
                "id": "UC1",
                "name": "Prepare Milk Coffee",
                "file": "Use Case/UC1.txt",
                "enabled": True,
                "priority": 1
            }
        ],
        "analysis_options": {
            "generate_diagrams": True,
            "generate_csv": True,
            "auto_organize_files": True,
            "backup_previous": True,
            "verbose_output": True
        }
    }
    
    with open("uc1_only_config.json", "w", encoding="utf-8") as f:
        json.dump(uc1_config, f, indent=2, ensure_ascii=False)
    
    # Multiple UC config (UC1 + UC2)
    multi_config = {
        "analysis_config": {
            "version": "1.0", 
            "description": "Multi UC Analysis - UC1 and UC2",
            "domain": "beverage_preparation",
            "output_format": ["json", "csv", "diagram"]
        },
        "use_cases": [
            {
                "id": "UC1",
                "name": "Prepare Milk Coffee", 
                "file": "Use Case/UC1.txt",
                "enabled": True,
                "priority": 1
            },
            {
                "id": "UC2",
                "name": "Prepare Espresso",
                "file": "Use Case/UC2.txt", 
                "enabled": True,
                "priority": 2
            }
        ],
        "analysis_options": {
            "generate_diagrams": True,
            "generate_csv": True,
            "auto_organize_files": True,
            "backup_previous": True,
            "verbose_output": True
        }
    }
    
    with open("uc1_uc2_config.json", "w", encoding="utf-8") as f:
        json.dump(multi_config, f, indent=2, ensure_ascii=False)
    
    print("Sample configuration files created:")
    print("  - uc1_only_config.json (UC1 only)")
    print("  - uc1_uc2_config.json (UC1 + UC2)")

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description="Enhanced UC Analyzer with JSON Configuration")
    parser.add_argument("config", nargs="?", default="uc_config.json", 
                       help="JSON configuration file (default: uc_config.json)")
    parser.add_argument("--create-samples", action="store_true",
                       help="Create sample configuration files")
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_configs()
        return
    
    # Initialize and run analyzer
    analyzer = EnhancedUCAnalyzer(args.config)
    success = analyzer.run_analysis()
    
    if success:
        print(f"\n🎯 Analysis completed successfully!")
    else:
        print(f"\n❌ Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()