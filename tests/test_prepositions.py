"""
Test comprehensive preposition detection
"""
import spacy
from uc1_advanced_verb_analyzer import UC1AdvancedVerbAnalyzer

def test_preposition_coverage():
    """Test various prepositions that might appear in UC steps"""
    
    analyzer = UC1AdvancedVerbAnalyzer()
    
    test_sentences = [
        # Location prepositions
        "The system places the cup under the filter",
        "The system moves coffee from storage to container", 
        "The system pours liquid into the cup",
        "The system extracts coffee through the filter",
        "The system heats water inside the tank",
        "The system grinds beans within the chamber",
        
        # Time prepositions  
        "The system activates heating after 5 minutes",
        "The system stops brewing before overflow",
        "The system maintains temperature during brewing",
        
        # Method/instrument prepositions
        "The system controls flow with a valve",
        "The system measures temperature by sensor",
        "The system operates via remote control",
        
        # Purpose prepositions
        "The system prepares filter for brewing",
        "The system heats water for extraction",
        
        # Compound prepositions
        "The system operates according to settings",
        "The system functions because of pressure",
        "The system responds instead of manual control"
    ]
    
    print("="*70)
    print("COMPREHENSIVE PREPOSITION DETECTION TEST")
    print("="*70)
    
    all_prepositions = set()
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i:2d}. {sentence}")
        
        # Analyze with spaCy
        doc = analyzer.nlp(sentence)
        
        # Find main verb
        main_verb = None
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                main_verb = token
                break
        
        if main_verb:
            # Get prepositional objects
            prep_objs = analyzer._find_prepositional_objects(main_verb)
            
            if prep_objs:
                for prep, obj in prep_objs:
                    print(f"    -> {prep}: {obj}")
                    all_prepositions.add(prep)
            else:
                print(f"    (No prepositional objects detected)")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(all_prepositions)} UNIQUE PREPOSITIONS DETECTED")
    print("="*70)
    
    for prep in sorted(all_prepositions):
        print(f"  - {prep}")
    
    print(f"\nTotal prepositions captured: {len(all_prepositions)}")

if __name__ == "__main__":
    test_preposition_coverage()