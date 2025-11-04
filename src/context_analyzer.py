#!/usr/bin/env python3
"""
Context Analyzer - Automatic domain detection from UC capabilities
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ContextAnalyzer:
    """Analyzes UC context to automatically determine domain"""
    
    def __init__(self):
        self.domains = {}
        self.load_domain_configurations()
    
    def load_domain_configurations(self):
        """Load all domain configurations"""
        domains_dir = Path("domains")
        
        if not domains_dir.exists():
            domains_dir = Path("../domains")  # If running from src/
        
        if not domains_dir.exists():
            print(f"[WARNING] Domains directory not found!")
            return
        
        for domain_file in domains_dir.glob("*.json"):
            try:
                with open(domain_file, 'r', encoding='utf-8') as f:
                    domain_config = json.load(f)
                    domain_name = domain_config.get('domain_name', domain_file.stem)
                    self.domains[domain_name] = domain_config
                    
            except Exception as e:
                print(f"[WARNING] Could not load domain {domain_file}: {e}")
        
        print(f"Loaded {len(self.domains)} domain configurations")
    
    def extract_capability_from_uc_file(self, uc_file_path: str) -> Optional[str]:
        """Extract capability from UC file"""
        try:
            with open(uc_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Look for Capability line
                    if line.lower().startswith('capability:'):
                        capability = line.split(':', 1)[1].strip()
                        print(f"[CONTEXT] Found capability: '{capability}'")
                        return capability
                    
                    # Stop looking after first few lines
                    if line_num > 10:
                        break
            
            print(f"[WARNING] No capability found in {uc_file_path}")
            return None
            
        except Exception as e:
            print(f"[ERROR] Could not read UC file {uc_file_path}: {e}")
            return None
    
    def detect_domain_from_capability(self, capability: str) -> Tuple[str, float]:
        """
        Detect domain from capability text
        
        Returns:
            Tuple of (domain_name, confidence_score)
        """
        if not capability:
            return 'common_domain', 0.0
        
        capability_lower = capability.lower()
        domain_scores = {}
        
        # Score each domain based on keyword matches
        for domain_name, domain_config in self.domains.items():
            score = 0.0
            keywords = domain_config.get('keywords', [])
            
            # Check keyword matches
            for keyword in keywords:
                if keyword.lower() in capability_lower:
                    # Exact word match gets higher score
                    if re.search(r'\\b' + re.escape(keyword.lower()) + r'\\b', capability_lower):
                        score += 2.0
                    else:
                        score += 1.0
            
            # Boost score for domain-specific terms
            if domain_name == 'beverage_preparation':
                beverage_terms = ['coffee', 'tea', 'drink', 'beverage', 'brewing', 'preparation']
                for term in beverage_terms:
                    if term in capability_lower:
                        score += 3.0
            
            elif domain_name == 'rocket_science':
                space_terms = ['rocket', 'launch', 'satellite', 'space', 'orbital', 'deployment', 'mission']
                for term in space_terms:
                    if term in capability_lower:
                        score += 3.0
            
            elif domain_name == 'nuclear':
                nuclear_terms = ['nuclear', 'reactor', 'power plant', 'radioactive', 'shutdown']
                for term in nuclear_terms:
                    if term in capability_lower:
                        score += 3.0
            
            elif domain_name == 'robotics':
                robot_terms = ['robot', 'assembly', 'automation', 'manufacturing', 'industrial']
                for term in robot_terms:
                    if term in capability_lower:
                        score += 3.0
            
            elif domain_name == 'automotive':
                auto_terms = ['vehicle', 'car', 'automotive', 'driving', 'navigation']
                for term in auto_terms:
                    if term in capability_lower:
                        score += 3.0
            
            elif domain_name == 'aerospace':
                aero_terms = ['aircraft', 'flight', 'aviation', 'navigation', 'air traffic']
                for term in aero_terms:
                    if term in capability_lower:
                        score += 3.0
            
            domain_scores[domain_name] = score
        
        # Find best match
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            domain_name, score = best_domain
            
            # Normalize confidence (simple approach)
            confidence = min(score / 5.0, 1.0)  # Max confidence of 1.0
            
            print(f"[CONTEXT] Domain detection results:")
            for domain, domain_score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  - {domain}: {domain_score:.1f}")
            
            print(f"[CONTEXT] Selected domain: {domain_name} (confidence: {confidence:.2f})")
            
            if confidence < 0.3:
                print(f"[WARNING] Low confidence in domain detection, using common_domain")
                return 'common_domain', confidence
            
            return domain_name, confidence
        
        return 'common_domain', 0.0
    
    def analyze_uc_context(self, uc_file_path: str) -> Dict:
        """
        Complete context analysis of UC file
        
        Returns:
            Dict with domain, capability, confidence, etc.
        """
        # Extract capability
        capability = self.extract_capability_from_uc_file(uc_file_path)
        
        # Detect domain
        domain, confidence = self.detect_domain_from_capability(capability)
        
        # Extract additional context
        feature = self.extract_feature_from_uc_file(uc_file_path)
        goal = self.extract_goal_from_uc_file(uc_file_path)
        
        context = {
            'uc_file': uc_file_path,
            'capability': capability,
            'feature': feature,
            'goal': goal,
            'detected_domain': domain,
            'confidence': confidence,
            'available_domains': list(self.domains.keys())
        }
        
        return context
    
    def extract_feature_from_uc_file(self, uc_file_path: str) -> Optional[str]:
        """Extract feature from UC file"""
        try:
            with open(uc_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if line.lower().startswith('feature:'):
                        feature = line.split(':', 1)[1].strip()
                        return feature
                    
                    if line_num > 10:
                        break
            
            return None
            
        except Exception as e:
            return None
    
    def extract_goal_from_uc_file(self, uc_file_path: str) -> Optional[str]:
        """Extract goal from UC file"""
        try:
            with open(uc_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if line.lower().startswith('goal:'):
                        goal = line.split(':', 1)[1].strip()
                        return goal
                    
                    if line_num > 15:
                        break
            
            return None
            
        except Exception as e:
            return None

def test_context_analyzer():
    """Test the context analyzer with available UC files"""
    analyzer = ContextAnalyzer()
    
    # Test with different UC files
    test_files = [
        'Use Case/UC1.txt',
        'Use Case/UC3_Rocket_Launch_Improved.txt',
        'Use Case/UC4_Nuclear_Shutdown.txt',
        'Use Case/UC5_Robot_Assembly.txt'
    ]
    
    for uc_file in test_files:
        if os.path.exists(uc_file):
            print(f"\\n{'='*60}")
            print(f"Testing: {uc_file}")
            print('='*60)
            
            context = analyzer.analyze_uc_context(uc_file)
            
            print(f"Capability: {context['capability']}")
            print(f"Feature: {context['feature']}")
            print(f"Goal: {context['goal']}")
            print(f"Detected Domain: {context['detected_domain']} (confidence: {context['confidence']:.2f})")

if __name__ == "__main__":
    test_context_analyzer()