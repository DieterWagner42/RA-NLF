#!/usr/bin/env python3
"""
Test Domain Orchestrator integration in generic UC analyzer
"""

from oldpy.generic_uc_analyzer import GenericUCAnalyzer
from pathlib import Path

def test_domain_orchestrator():
    """Test that Domain Orchestrator is included in analysis"""
    print("TESTING DOMAIN ORCHESTRATOR INTEGRATION")
    print("=" * 50)
    
    uc1_file = "D:\\KI\\RA-NLF\\Use Case\\UC1.txt"
    
    if Path(uc1_file).exists():
        print(f"\nAnalyzing UC1 with Domain Orchestrator...")
        analyzer = GenericUCAnalyzer(domain_name="beverage_preparation")
        verb_analyses, ra_classes = analyzer.analyze_uc_file(uc1_file)
        
        # Check for Domain Orchestrator
        orchestrators = [ra for ra in ra_classes if "DomainOrchestrator" in ra.name]
        
        print(f"\nResults: {len(verb_analyses)} verb analyses, {len(ra_classes)} RA classes")
        print(f"Domain Orchestrators found: {len(orchestrators)}")
        
        for orchestrator in orchestrators:
            print(f"\nDomain Orchestrator:")
            print(f"  Name: {orchestrator.name}")
            print(f"  Type: {orchestrator.type}")
            print(f"  Description: {orchestrator.description}")
            print(f"  Source: {orchestrator.source}")
    else:
        print(f"ERROR: UC1 file not found")

if __name__ == "__main__":
    test_domain_orchestrator()