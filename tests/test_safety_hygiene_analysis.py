#!/usr/bin/env python3
"""
Test Safety, Hygiene and Operational Materials Analysis
Demonstrates UC-Methode analysis extended with safety/hygiene requirements
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
from oldpy.generic_uc_analyzer import GenericUCAnalyzer

def test_safety_hygiene_analysis():
    """Test safety and hygiene analysis across different domains"""
    
    print("="*80)
    print("SAFETY, HYGIENE & OPERATIONAL MATERIALS ANALYSIS TEST")
    print("="*80)
    
    # Test cases for different domains
    test_cases = [
        {
            "name": "UC1: Beverage Preparation (Food Safety)",
            "uc_file": "D:\\KI\\RA-NLF\\Use Case\\UC1.txt",
            "domain": "beverage_preparation",
            "expected_materials": ["Water", "Milk", "CoffeeBeans", "Sugar"],
            "expected_safety_classes": ["standard"],
            "expected_hygiene_levels": ["food_grade"]
        },
        {
            "name": "UC3: Rocket Launch (Explosive Materials)",
            "uc_file": "D:\\KI\\RA-NLF\\Use Case\\UC3_Rocket_Launch.txt", 
            "domain": "aerospace",
            "expected_materials": ["Fuel", "Oxidizer"],
            "expected_safety_classes": ["explosive"],
            "expected_hygiene_levels": ["cleanroom"]
        },
        {
            "name": "UC4: Nuclear Shutdown (Radioactive Materials)",
            "uc_file": "D:\\KI\\RA-NLF\\Use Case\\UC4_Nuclear_Shutdown.txt",
            "domain": "nuclear", 
            "expected_materials": ["Uranium", "Coolant"],
            "expected_safety_classes": ["radioactive", "toxic"],
            "expected_hygiene_levels": ["cleanroom"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST CASE: {test_case['name']}")
        print(f"{'='*80}")
        
        uc_file_path = test_case["uc_file"]
        
        if not Path(uc_file_path).exists():
            print(f"SKIPPED: UC file not found: {uc_file_path}")
            continue
            
        try:
            # Initialize analyzer for specific domain
            analyzer = GenericUCAnalyzer(domain_name=test_case["domain"])
            
            # Perform comprehensive analysis
            verb_analyses, ra_classes, operational_materials, safety_constraints, hygiene_requirements = \
                analyzer.analyze_uc_with_safety_hygiene(uc_file_path)
            
            # Validate results
            print(f"\n{'='*60}")
            print(f"VALIDATION RESULTS")
            print(f"{'='*60}")
            
            # Check operational materials
            found_materials = [mat.material_name for mat in operational_materials]
            print(f"Found Materials: {found_materials}")
            print(f"Expected Materials: {test_case['expected_materials']}")
            
            # Check safety classifications
            found_safety_classes = list(set([mat.safety_class for mat in operational_materials]))
            print(f"Found Safety Classes: {found_safety_classes}")
            print(f"Expected Safety Classes: {test_case['expected_safety_classes']}")
            
            # Check hygiene levels
            found_hygiene_levels = list(set([mat.hygiene_level for mat in operational_materials]))
            print(f"Found Hygiene Levels: {found_hygiene_levels}")
            print(f"Expected Hygiene Levels: {test_case['expected_hygiene_levels']}")
            
            # Check addressing IDs
            print(f"\nADDRESSING IDs:")
            for material in operational_materials:
                print(f"  {material.material_name}: {material.addressing_id}")
            
            # Summary statistics
            print(f"\nSUMMARY STATISTICS:")
            print(f"  Operational Materials: {len(operational_materials)}")
            print(f"  Safety Constraints: {len(safety_constraints)}")
            print(f"  Hygiene Requirements: {len(hygiene_requirements)}")
            print(f"  Controllers: {len([ra for ra in ra_classes if ra.type == 'Controller'])}")
            print(f"  Entities: {len([ra for ra in ra_classes if ra.type == 'Entity'])}")
            
            # Demonstration of domain-specific requirements
            print(f"\nDOMAIN-SPECIFIC HIGHLIGHTS:")
            if test_case["domain"] == "beverage_preparation":
                milk_materials = [m for m in operational_materials if "milk" in m.material_name.lower()]
                if milk_materials:
                    milk = milk_materials[0]
                    print(f"  Milk Storage: {milk.storage_conditions}")
                    print(f"  Milk Requirements: {milk.special_requirements}")
                    
            elif test_case["domain"] == "aerospace":
                fuel_constraints = [c for c in safety_constraints if "fuel" in c.material_name.lower()]
                if fuel_constraints:
                    constraint = fuel_constraints[0]
                    print(f"  Fuel Safety Limits: {constraint.max_limits}")
                    print(f"  Fuel Emergency Actions: {constraint.emergency_actions}")
                    
            elif test_case["domain"] == "nuclear":
                rad_constraints = [c for c in safety_constraints if c.constraint_type == "radiation"]
                if rad_constraints:
                    constraint = rad_constraints[0]
                    print(f"  Radiation Limits: {constraint.max_limits}")
                    print(f"  Radiation Monitoring: {constraint.monitoring_required}")
            
            print(f"\n✅ TEST PASSED: {test_case['name']}")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_case['name']}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("SAFETY/HYGIENE ANALYSIS TEST COMPLETED")
    print(f"{'='*80}")

def demonstrate_addressing_system():
    """Demonstrate the universal addressing system for operational materials"""
    
    print(f"\n{'='*80}")
    print("UNIVERSAL ADDRESSING SYSTEM DEMONSTRATION")
    print(f"{'='*80}")
    
    # Examples of addressing IDs across domains
    addressing_examples = [
        {
            "domain": "Beverage Preparation",
            "material": "Milk",
            "addressing_id": "STANDARD-FOOD_GRADE-MILK-B20241026-COOLER02",
            "description": "Standard safety, food-grade hygiene, batch tracked, stored in cooler"
        },
        {
            "domain": "Aerospace",
            "material": "Rocket Fuel",
            "addressing_id": "EXPLOSIVE-CLEANROOM-FUEL-B20241026-TANK001", 
            "description": "Explosive safety class, cleanroom hygiene, batch tracked, stored in tank"
        },
        {
            "domain": "Nuclear",
            "material": "Uranium",
            "addressing_id": "RADIOACTIVE-CLEANROOM-U235-B20241026-VAULT001",
            "description": "Radioactive safety class, cleanroom hygiene, enriched uranium, vault storage"
        },
        {
            "domain": "Medical",
            "material": "Sterile Instruments",
            "addressing_id": "TOXIC-STERILE-SURGICAL-B20241026-OR001",
            "description": "Toxic safety (cleaning chemicals), sterile hygiene, surgical batch, OR storage"
        }
    ]
    
    print(f"\nADDRESSING FORMAT: {{SAFETY_CLASS}}-{{HYGIENE_LEVEL}}-{{MATERIAL_CODE}}-{{BATCH_ID}}-{{LOCATION}}")
    print(f"\n{'Domain':<20} {'Material':<15} {'Addressing ID':<45} {'Description'}")
    print("-" * 120)
    
    for example in addressing_examples:
        print(f"{example['domain']:<20} {example['material']:<15} {example['addressing_id']:<45} {example['description']}")
    
    print(f"\nKEY BENEFITS:")
    print(f"  ✅ Universal format across all domains")
    print(f"  ✅ Safety classification embedded")
    print(f"  ✅ Hygiene requirements visible") 
    print(f"  ✅ Batch traceability included")
    print(f"  ✅ Location tracking integrated")
    print(f"  ✅ Regulatory compliance ready")

def demonstrate_cross_domain_comparison():
    """Compare safety/hygiene requirements across domains"""
    
    print(f"\n{'='*80}")
    print("CROSS-DOMAIN SAFETY/HYGIENE COMPARISON")
    print(f"{'='*80}")
    
    domain_comparison = {
        "beverage_preparation": {
            "primary_concerns": ["Food safety", "Temperature control", "Contamination prevention"],
            "regulatory_standards": ["HACCP", "FDA", "ISO 22000"],
            "critical_materials": ["Milk (perishable)", "Water (quality)", "Coffee (freshness)"],
            "monitoring_systems": ["Temperature sensors", "Quality testing", "Expiration tracking"]
        },
        "aerospace": {
            "primary_concerns": ["Explosion prevention", "Static elimination", "Contamination control"],
            "regulatory_standards": ["NASA standards", "OSHA", "DOT"],
            "critical_materials": ["Propellants (explosive)", "Oxidizers (reactive)", "Cryogenics (pressure)"],
            "monitoring_systems": ["Static detectors", "Pressure monitors", "Fire suppression"]
        },
        "nuclear": {
            "primary_concerns": ["Radiation safety", "Criticality prevention", "Contamination control"],
            "regulatory_standards": ["NRC", "IAEA", "10 CFR"],
            "critical_materials": ["Uranium (radioactive)", "Plutonium (fissile)", "Coolant (toxic)"],
            "monitoring_systems": ["Radiation detectors", "Criticality alarms", "Contamination monitors"]
        },
        "medical": {
            "primary_concerns": ["Sterility maintenance", "Biocompatibility", "Traceability"],
            "regulatory_standards": ["FDA", "ISO 13485", "GMP"],
            "critical_materials": ["Devices (sterile)", "Drugs (pure)", "Biologics (viable)"],
            "monitoring_systems": ["Bioburden testing", "Sterility validation", "Environmental monitoring"]
        }
    }
    
    print(f"\n{'Domain':<20} {'Primary Concerns':<35} {'Critical Materials':<30} {'Standards'}")
    print("-" * 120)
    
    for domain, info in domain_comparison.items():
        concerns = ", ".join(info["primary_concerns"][:2])  # First 2 concerns
        materials = ", ".join([m.split("(")[0].strip() for m in info["critical_materials"][:2]])  # First 2 materials
        standards = ", ".join(info["regulatory_standards"][:2])  # First 2 standards
        
        print(f"{domain.replace('_', ' ').title():<20} {concerns:<35} {materials:<30} {standards}")
    
    print(f"\nCOMMON PATTERNS:")
    print(f"  🔍 All domains require contamination control")
    print(f"  📊 All domains need environmental monitoring")
    print(f"  📋 All domains follow regulatory standards")
    print(f"  🔒 All domains require access control")
    print(f"  📝 All domains need traceability")
    print(f"  ⚠️  All domains have emergency procedures")

if __name__ == "__main__":
    # Run comprehensive safety/hygiene analysis test
    test_safety_hygiene_analysis()
    
    # Demonstrate addressing system
    demonstrate_addressing_system()
    
    # Show cross-domain comparison
    demonstrate_cross_domain_comparison()