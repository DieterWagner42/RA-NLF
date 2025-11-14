import json

# Load UC1 analysis
with open('new/multi_uc/UC1_Structured_RA_Analysis.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

# Load UC2 analysis
with open('new/multi_uc/UC2_Structured_RA_Analysis.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

print("=" * 80)
print("UC1 Controllers with Implicit Protection Functions")
print("=" * 80)

for c in data1['components']['controllers']:
    print(f"\n{c['name']}:")
    print(f"  {c['description']}")

print("\n\n" + "=" * 80)
print("UC2 Controllers with Implicit Protection Functions")
print("=" * 80)

for c in data2['components']['controllers']:
    print(f"\n{c['name']}:")
    print(f"  {c['description']}")
