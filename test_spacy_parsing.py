#!/usr/bin/env python3
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Test the problematic phrase
text = "Launch window opens at scheduled time"
doc = nlp(text)

print("=== spaCy Analysis for 'Launch window opens at scheduled time' ===")
for token in doc:
    print(f"'{token.text}' -> POS: {token.pos_}, DEP: {token.dep_}, HEAD: '{token.head.text}'")

print("\n=== Testing just 'scheduled time' ===")
text2 = "scheduled time"
doc2 = nlp(text2)
for token in doc2:
    print(f"'{token.text}' -> POS: {token.pos_}, DEP: {token.dep_}, HEAD: '{token.head.text}'")

print("\n=== Testing compound noun detection logic ===")
def test_compound_detection(text):
    doc = nlp(text)
    processed_tokens = []
    i = 0
    
    while i < len(doc):
        token = doc[i]
        
        # Look for compound patterns
        if i + 1 < len(doc):
            next_token = doc[i + 1]
            
            # Pattern 1: NOUN + NOUN
            if token.pos_ == "NOUN" and next_token.pos_ == "NOUN":
                compound = token.text.capitalize() + next_token.text.capitalize()
                processed_tokens.append(compound)
                print(f"  NOUN+NOUN: '{token.text}' + '{next_token.text}' -> {compound}")
                i += 2
                continue
            
            # Pattern 2: ADJ + NOUN
            elif token.pos_ == "ADJ" and next_token.pos_ == "NOUN":
                compound = token.text.capitalize() + next_token.text.capitalize()
                processed_tokens.append(compound)
                print(f"  ADJ+NOUN: '{token.text}' + '{next_token.text}' -> {compound}")
                i += 2
                continue
            
            # Pattern 2b: VERB + NOUN (for participles like "scheduled time")
            elif token.pos_ == "VERB" and next_token.pos_ == "NOUN" and token.dep_ == "amod":
                compound = token.text.capitalize() + next_token.text.capitalize()
                processed_tokens.append(compound)
                print(f"  VERB+NOUN (participle): '{token.text}' + '{next_token.text}' -> {compound}")
                i += 2
                continue
        
        processed_tokens.append(token.text)
        i += 1
    
    return " ".join(processed_tokens)

result = test_compound_detection("Launch window opens at scheduled time")
print(f"\nResult: '{result}'")