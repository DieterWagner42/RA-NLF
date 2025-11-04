"""
Test RUP-compliant fix directly with fresh Phase 2 data
"""

from domain_config_analyzer import DomainConfigurableAnalyzer
from phase2_betriebsmittel_analyzer import BetriebsmittelAnalyzer
from phase5_datenfluss_analyzer import DatenflussAnalyzer
from phase3_interaktion_analyzer import Phase3Result
from phase4_kontrollfluss_analyzer import Phase4Result

def test_rup_compliant_fix():
    """Test the RUP-compliant fix with fresh data"""
    
    print("=== Testing RUP-Compliant Fix ===")
    
    # Initialize analyzers
    domain_analyzer = DomainConfigurableAnalyzer()
    phase2_analyzer = BetriebsmittelAnalyzer(domain_analyzer)
    phase5_analyzer = DatenflussAnalyzer(domain_analyzer)
    
    # Phase 1 analysis
    phase1_result = domain_analyzer.perform_phase1_analysis(
        capability_name="Coffee Preparation",
        uc_title="Prepare Milk Coffee", 
        goal_text="User can drink their milk coffee every morning at 7am",
        actors_text="User, Timer"
    )
    
    # Phase 2 analysis with RUP-compliant transformations
    preconditions = [
        "Coffee beans are available in the system",
        "Water is available in the system", 
        "Milk is available in the system"
    ]
    
    phase2_result = phase2_analyzer.perform_phase2_analysis(phase1_result, preconditions)
    
    print(f"Phase 2 completed. Resources: {len(phase2_result.resource_analyses)}")
    
    # Print transformations to verify they have "validated" prefix
    for analysis in phase2_result.resource_analyses:
        if analysis.transformations:
            print(f"Resource: {analysis.resource_name}")
            for transformation in analysis.transformations:
                print(f"  Transformation: {transformation}")
    
    # Create mock Phase 3 and Phase 4 results
    phase3_result = Phase3Result(
        phase1_result=phase1_result,
        phase2_result=phase2_result,
        uc_steps=[],
        interactions=[],
        controller_functions=[],
        entity_flows=[],
        orchestration_pattern={},
        missing_preconditions=[],
        summary="Mock Phase 3"
    )
    
    phase4_result = Phase4Result(
        phase3_result=phase3_result,
        control_flow_patterns=[],
        decision_points=[],
        coordination_rules={},
        flow_violations=[],
        summary="Mock Phase 4"
    )
    
    # Phase 5 analysis with fresh data (not from JSON)
    print(f"\n=== Phase 5 Analysis ===")
    phase5_result = phase5_analyzer.perform_phase5_analysis(phase4_result)
    
    print(f"Phase 5 completed!")
    print(f"Summary: {phase5_result.summary}")
    print(f"Data Entities: {len(phase5_result.data_entities)}")
    print(f"Data Violations: {len(phase5_result.data_violations)}")
    
    if phase5_result.data_violations:
        print("\nRemaining violations:")
        for violation in phase5_result.data_violations:
            print(f"  - {violation}")
    else:
        print("\nSUCCESS: No violations found with RUP-compliant fix!")
    
    # Check dependencies to verify RUP compliance
    print(f"\n=== Dependency Analysis ===")
    for entity in phase5_result.data_entities:
        if entity.dependencies:
            print(f"{entity.entity_id} depends on: {entity.dependencies}")
    
    return phase5_result

if __name__ == "__main__":
    test_rup_compliant_fix()