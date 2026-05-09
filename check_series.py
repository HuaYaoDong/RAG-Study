import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/knowledge_graph.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== All concept keys ===')
for c in data['concepts']:
    print(f"  - {c['key']}: {c['name']}")

print('\n=== Search for "series" or "series" ===')
found = [c for c in data['concepts'] if 'series' in c.get('key', '').lower() or 'series' in c.get('name', '').lower()]
for c in found:
    print(f"  OK: {c['key']}: {c['name']}")

if not found:
    print('  NOT FOUND')
