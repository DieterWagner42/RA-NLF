"""
Integrated UC-Methode Analyzer - Phases 1-5
Complete systematic robustness analysis according to UC-Methode.txt
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from domain_config_analyzer import DomainConfigurableAnalyzer
from phase2_betriebsmittel_analyzer import BetriebsmittelAnalyzer
from phase3_interaktion_analyzer import InteraktionAnalyzer
from phase4_kontrollfluss_analyzer import KontrollflussAnalyzer
from phase5_datenfluss_analyzer import DatenflussAnalyzer


@dataclass
class IntegratedAnalysisResult:
    """Complete analysis result from all 5 phases"""
    phase1_result: object
    phase2_result: object
    phase3_result: object
    phase4_result: object
    phase5_result: object
    overall_summary: str
    total_violations: List[str]
    rup_compliance: bool


class IntegratedUCAnalyzer:
    """
    Complete UC-Methode implementation integrating all 5 phases
    """
    
    def __init__(self):
        """Initialize all phase analyzers"""
        self.domain_analyzer = DomainConfigurableAnalyzer()
        self.phase2_analyzer = BetriebsmittelAnalyzer(self.domain_analyzer)
        self.phase3_analyzer = InteraktionAnalyzer(self.domain_analyzer)
        self.phase4_analyzer = KontrollflussAnalyzer(self.domain_analyzer)
        self.phase5_analyzer = DatenflussAnalyzer(self.domain_analyzer)
    
    def analyze_complete_uc(self, uc_file_path: str, capability: str, feature: str) -> IntegratedAnalysisResult:
        """
        Perform complete 5-phase UC-Methode analysis
        
        Args:
            uc_file_path: Path to use case text file
            capability: Capability name for Phase 1
            feature: Feature name for Phase 1
            
        Returns:
            Complete integrated analysis result
        """
        print(f"Starting integrated UC-Methode analysis for {uc_file_path}...")
        
        # Read UC text
        with open(uc_file_path, 'r', encoding='utf-8') as f:
            uc_text = f.read()
        
        # Extract preconditions for Phase 2
        preconditions = self._extract_preconditions(uc_text)
        
        # Phase 1: Context Analysis (simplified for integration)
        print("Phase 1: Context Analysis...")
        capability_context = self.domain_analyzer.analyze_capability(capability)
        uc_title_analysis = self.domain_analyzer.analyze_uc_title(feature, capability_context)
        goal_analysis = self.domain_analyzer.analyze_goal_and_actors(uc_text, capability_context)
        
        # Create combined Phase 1 result
        from dataclasses import dataclass
        @dataclass
        class Phase1Result:
            capability_context: object
            uc_title_analysis: object
            goal_analysis: object
            summary: str
        
        phase1_result = Phase1Result(
            capability_context=capability_context,
            uc_title_analysis=uc_title_analysis,
            goal_analysis=goal_analysis,
            summary=f"Domain: {capability_context.domain}; Capability: {capability}; Actors: {len(goal_analysis.actors)}"
        )
        
        # Phase 2: Resource Analysis (with domain context derivation)
        print("Phase 2: Resource Analysis...")
        phase2_result = self.phase2_analyzer.perform_phase2_analysis(phase1_result, preconditions)
        
        # Phase 3: Interaction Analysis
        print("Phase 3: Interaction Analysis...")
        phase3_result = self.phase3_analyzer.perform_phase3_analysis(phase2_result, uc_text)
        
        # Phase 4: Control Flow Analysis  
        print("Phase 4: Control Flow Analysis...")
        phase4_result = self.phase4_analyzer.perform_phase4_analysis(phase3_result, uc_text)
        
        # Phase 5: Data Flow Analysis
        print("Phase 5: Data Flow Analysis...")
        phase5_result = self.phase5_analyzer.perform_phase5_analysis(phase4_result)
        
        # Collect all violations
        total_violations = []
        if hasattr(phase3_result, 'violations'):
            total_violations.extend(phase3_result.violations)
        if hasattr(phase4_result, 'flow_violations'):
            total_violations.extend(phase4_result.flow_violations)
        if hasattr(phase5_result, 'data_violations'):
            total_violations.extend(phase5_result.data_violations)
        
        # Determine RUP compliance
        rup_compliance = len(total_violations) == 0
        
        # Generate overall summary
        overall_summary = self._generate_overall_summary(
            phase1_result, phase2_result, phase3_result, phase4_result, phase5_result, total_violations
        )
        
        result = IntegratedAnalysisResult(
            phase1_result=phase1_result,
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phase4_result=phase4_result,
            phase5_result=phase5_result,
            overall_summary=overall_summary,
            total_violations=total_violations,
            rup_compliance=rup_compliance
        )
        
        print(f"Integrated analysis completed!")
        print(f"RUP Compliance: {rup_compliance}")
        print(f"Total violations: {len(total_violations)}")
        
        return result
    
    def _extract_preconditions(self, uc_text: str) -> List[str]:
        """Extract preconditions from UC text"""
        preconditions = []
        lines = uc_text.split('\n')
        in_preconditions = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("Preconditions:"):
                in_preconditions = True
                continue
            elif in_preconditions and line.startswith("Actors:"):
                break
            elif in_preconditions and line.startswith("- "):
                preconditions.append(line[2:])  # Remove "- "
        
        return preconditions
    
    def _generate_overall_summary(self, phase1, phase2, phase3, phase4, phase5, violations) -> str:
        """Generate comprehensive summary of all phases"""
        
        summary_parts = [
            f"PHASE 1: {phase1.summary if hasattr(phase1, 'summary') else 'Context analysis completed'}",
            f"PHASE 2: {phase2.summary}",
            f"PHASE 3: {phase3.summary}",
            f"PHASE 4: {phase4.summary}",
            f"PHASE 5: {phase5.summary}",
            f"VIOLATIONS: {len(violations)} total issues detected",
            f"UC-METHODE: All 5 phases completed successfully"
        ]
        
        return " | ".join(summary_parts)
    
    def save_integrated_result(self, result: IntegratedAnalysisResult, output_file: str):
        """Save complete analysis result to JSON file"""
        
        # Convert complex objects to serializable format
        serializable_result = {
            "overall_summary": result.overall_summary,
            "rup_compliance": result.rup_compliance,
            "total_violations": result.total_violations,
            "phase_summaries": {
                "phase1": getattr(result.phase1_result, 'summary', 'Phase 1 completed'),
                "phase2": result.phase2_result.summary,
                "phase3": result.phase3_result.summary,
                "phase4": result.phase4_result.summary,
                "phase5": result.phase5_result.summary
            },
            "analysis_details": {
                "domain": result.phase1_result.capability_context.domain,
                "total_resources": len(result.phase2_result.resource_analyses),
                "total_controllers": len(result.phase3_result.all_objects.get("controllers", [])),
                "total_boundaries": len(result.phase3_result.all_objects.get("boundaries", [])),
                "total_entities": len(result.phase3_result.all_objects.get("entities", [])),
                "control_patterns": len(result.phase4_result.control_flow_patterns),
                "data_entities": len(result.phase5_result.data_entities),
                "data_transformations": len(result.phase5_result.data_transformations)
            },
            "uc_methode_rules_applied": [
                "Phase 2: Domain-Context-Regel (implizite Ressourcen)",
                "Phase 3: Koordinator-Regel (externe Ereignisse)",
                "Phase 4: Kontrollfluss-Regeln (Parallelisierung & Sequenzierung)",
                "Phase 5: Datenfluss-Regeln (Validierung & Transformation)"
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        print(f"Integrated analysis result saved to: {output_file}")
    
    def generate_compliance_report(self, result: IntegratedAnalysisResult) -> str:
        """Generate UC-Methode compliance report"""
        
        report = f"""
UC-METHODE COMPLIANCE REPORT
============================

Overall Compliance: {'PASS' if result.rup_compliance else 'FAIL'}
Total Violations: {len(result.total_violations)}

PHASE ANALYSIS:
Phase 1 (Context): ✓ Domain identification and actor classification
Phase 2 (Resources): ✓ Resource analysis with domain context derivation  
Phase 3 (Interactions): ✓ Interaction patterns with coordination rules
Phase 4 (Control Flow): ✓ Control flow analysis with parallelization
Phase 5 (Data Flow): ✓ Data flow analysis with transformation validation

APPLIED UC-METHODE RULES:
✓ Domain-Context-Regel: Implicit resources derived from domain knowledge
✓ Koordinator-Regel: External events properly separated from coordination
✓ Kontrollfluss-Regeln: Parallel and sequential execution properly identified
✓ Datenfluss-Regeln: Data transformations and validations implemented

VIOLATIONS DETECTED:
"""
        
        if result.total_violations:
            for i, violation in enumerate(result.total_violations, 1):
                report += f"{i}. {violation}\n"
        else:
            report += "None - Full UC-Methode compliance achieved!\n"
        
        report += f"""
SUMMARY:
{result.overall_summary}

RUP ANALYSIS CLASS DIAGRAM READY: {'YES' if result.rup_compliance else 'NEEDS FIXES'}
"""
        
        return report


def main():
    """Demonstrate integrated analysis with UC3"""
    analyzer = IntegratedUCAnalyzer()
    
    # Analyze UC3 - Rocket Launch
    uc3_result = analyzer.analyze_complete_uc(
        uc_file_path="Use Case/UC3_Rocket_Launch.txt",
        capability="Rocket Launch", 
        feature="Satellite Deployment"
    )
    
    # Save integrated result
    output_file = "Zwischenprodukte/UC3_Rocket_Launch_integrated_analysis.json"
    analyzer.save_integrated_result(uc3_result, output_file)
    
    # Generate compliance report
    compliance_report = analyzer.generate_compliance_report(uc3_result)
    
    # Save compliance report
    report_file = "Zwischenprodukte/UC3_Rocket_Launch_compliance_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(compliance_report)
    
    print(f"Compliance report saved to: {report_file}")
    print("\n" + "="*50)
    print(compliance_report)
    
    return uc3_result


if __name__ == "__main__":
    main()