#!/usr/bin/env python3
"""
Debug script to analyze how spaCy parses direct objects in the sentence:
"System clock reaches the user defined time"
"""

import spacy
import sys

def debug_sentence_parsing():
    """Debug the parsing of a specific sentence"""
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_lg")
        print("[OK] spaCy model loaded successfully")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_md")
            print("[OK] spaCy model loaded successfully")
        except OSError:
            print("ERROR: spaCy model not found")
            print("Please install: python -m spacy download en_core_web_md")
            return

    # Test sentence
    sentence = "System clock reaches the user defined time"
    print(f"\nAnalyzing sentence: '{sentence}'")
    
    # Process with spaCy
    doc = nlp(sentence)
    
    print("\n=== Token Analysis ===")
    for i, token in enumerate(doc):
        print(f"{i}: {token.text:15} | {token.pos_:8} | {token.dep_:12} | {token.lemma_:15} | Head: {token.head.text}")
    
    print("\n=== Noun Chunks ===")
    for chunk in doc.noun_chunks:
        print(f"Chunk: '{chunk.text}' | Root: {chunk.root.text} | Start: {chunk.start} | End: {chunk.end}")
    
    # Find the main verb
    print("\n=== Finding Main Verb ===")
    main_verb = None
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            main_verb = token
            print(f"Main verb: {main_verb.text} (lemma: {main_verb.lemma_})")
            break
    
    if not main_verb:
        for token in doc:
            if token.pos_ == "VERB":
                main_verb = token
                print(f"Verb found: {main_verb.text} (lemma: {main_verb.lemma_})")
                break
    
    # Find direct object
    print("\n=== Finding Direct Object ===")
    if main_verb:
        print(f"Looking for direct object of verb: {main_verb.text}")
        for child in main_verb.children:
            print(f"  Child: {child.text} | Dep: {child.dep_} | POS: {child.pos_}")
            if child.dep_ == "dobj":
                print(f"  -> DIRECT OBJECT FOUND: {child.text}")
                
                # Expand to full noun phrase
                print(f"  -> Expanding noun phrase...")
                for chunk in doc.noun_chunks:
                    if child in chunk:
                        print(f"  -> EXPANDED NOUN PHRASE: '{chunk.text}'")
                        return chunk.text
                
                # If not in a chunk, return the token itself
                print(f"  -> NO CHUNK FOUND, using token: '{child.text}'")
                return child.text
    
    print("No direct object found!")
    return ""

def test_alternative_parsing():
    """Test alternative approaches to finding the direct object"""
    
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            print("ERROR: spaCy model not found")
            return

    sentence = "System clock reaches the user defined time"
    doc = nlp(sentence)
    
    print(f"\n=== Alternative Analysis for: '{sentence}' ===")
    
    # Method 1: Look for noun chunks that come after the verb
    print("\n--- Method 1: Noun chunks after verb ---")
    verb_found = False
    for token in doc:
        if token.pos_ == "VERB":
            verb_found = True
            print(f"Verb found: {token.text}")
            break
    
    if verb_found:
        for chunk in doc.noun_chunks:
            # Check if chunk comes after the verb and is likely the object
            chunk_start = chunk.start
            verb_pos = None
            for i, token in enumerate(doc):
                if token.pos_ == "VERB":
                    verb_pos = i
                    break
            
            if verb_pos and chunk_start > verb_pos:
                print(f"Potential object chunk: '{chunk.text}'")
    
    # Method 2: Look for the rightmost noun phrase
    print("\n--- Method 2: Rightmost substantial noun phrase ---")
    rightmost_chunk = None
    for chunk in doc.noun_chunks:
        # Skip very short chunks and determiners only
        words = chunk.text.split()
        substantial_words = [w for w in words if w.lower() not in ['the', 'a', 'an']]
        if len(substantial_words) >= 2:  # Must have at least 2 substantial words
            rightmost_chunk = chunk
    
    if rightmost_chunk:
        print(f"Rightmost substantial chunk: '{rightmost_chunk.text}'")

if __name__ == "__main__":
    result = debug_sentence_parsing()
    test_alternative_parsing()