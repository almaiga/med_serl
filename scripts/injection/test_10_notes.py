import json
from pathlib import Path
from inject_benign_changes import BenignChangeGenerator
from inject_benign_changes import parse_extraction

def print_change(result: dict, index: int):
    """Pretty print a single change result."""
    print(f"\n{'='*80}")
    print(f"CHANGE #{index}")
    print(f"{'='*80}")
    print(f"Change Type: {result['change_type']}")
    print(f"Note ID: {result.get('note_id', 'N/A')}")
    print(f"Change Made: {result['change_made']}")
    
    if result.get('change_made'):
        print(f"Verified: {result.get('verified', False)}")
        print(f"Source: {result.get('verification_source', 'N/A')}")
        
        # Show term replacement if available
        if result.get('original_term') and result.get('replacement_term'):
            print(f"\n🔄 TERM REPLACEMENT:")
            print(f"   '{result['original_term']}' → '{result['replacement_term']}'")
        
        # Show sentence change
        if result.get('original_sentence') and result.get('modified_sentence'):
            print(f"\n📝 SENTENCE CHANGE:")
            print(f"\n   ORIGINAL:")
            print(f"   {result['original_sentence']}")
            print(f"\n   MODIFIED:")
            print(f"   {result['modified_sentence']}")
        
        # Show position
        if result.get('sentence_position'):
            pos = result['sentence_position']
            print(f"\n📍 Position: chars {pos['start']}-{pos['end']}")
    else:
        print(f"\n❌ FAILED:")
        error = result.get('error', 'Unknown')
        # Try to pretty-print JSON error messages
        if isinstance(error, str):
            error_str = error.strip()
            # If error looks like a JSON string, pretty-print it
            if (error_str.startswith("{") and error_str.endswith("}")) or (error_str.startswith("[") and error_str.endswith("]")):
                try:
                    import json
                    parsed = json.loads(error_str)
                    print("   Error (parsed JSON):")
                    print(json.dumps(parsed, indent=4))
                except Exception:
                    print(f"   Reason: {error}")
            else:
                # Clean up escaped newlines/quotes for readability
                cleaned = error_str.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                print(f"   Reason: {cleaned}")
        else:
            print(f"   Reason: {error}")
        # Optionally, print LLM raw response if available
        if 'llm_response' in result:
            print("\n   LLM raw response:")
            print(result['llm_response'])


# Load just 10 notes
data_path = Path("data_processed/parsed_medical_note/extractions.jsonl")
notes = []
with open(data_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        notes.append(json.loads(line))

print(f"Testing with {len(notes)} notes...\n")

# Initialize generator
config_path = Path("configs/prompts/benign_change_prompt.json")
generator = BenignChangeGenerator(config_path, model="gpt-4o-mini")

print(f"Change types: {generator.change_types}\n")

# Generate changes but DON'T save to file
from tqdm import tqdm
import random

assignments = generator.assign_change_types(notes)
random.shuffle(assignments)

results = []

print("Generating benign changes...\n")
for item in tqdm(assignments):
    note_obj = item["note"]
    change_type = item["change_type"]
    note_text = note_obj.get("correct_note", "")
    
    extraction = parse_extraction(note_obj.get("extraction", {}))
    
    # Enable fallback for each note
    result = generator.generate_change(note_text, change_type, extraction, allow_fallback=True)
    result["note_id"] = note_obj.get("note_id")
    
    results.append(result)

# Now print all results
print("\n" + "="*80)
print("ALL CHANGES (NOT SAVED)")
print("="*80)

successful = [r for r in results if r.get('change_made')]
failed = [r for r in results if not r.get('change_made')]

print(f"\nTotal: {len(results)} changes")
print(f"✓ Successful: {len(successful)}")
print(f"✗ Failed: {len(failed)}")

# Show all successful
print("\n" + "="*80)
print("SUCCESSFUL CHANGES")
print("="*80)

for i, result in enumerate(successful, 1):
    print_change(result, i)

# Show all failed
if failed:
    print("\n" + "="*80)
    print("FAILED CHANGES")
    print("="*80)
    
    for i, result in enumerate(failed, 1):
        print_change(result, i)

# Summary
print("\n" + "="*80)
print("SUMMARY BY TYPE")
print("="*80)

from collections import defaultdict
by_type = defaultdict(lambda: {"success": 0, "failed": 0, "verified": 0})

for result in results:
    ct = result['change_type']
    if result.get('change_made'):
        by_type[ct]["success"] += 1
        if result.get('verified'):
            by_type[ct]["verified"] += 1
    else:
        by_type[ct]["failed"] += 1

for change_type in sorted(by_type.keys()):
    counts = by_type[change_type]
    print(f"\n{change_type}:")
    print(f"  ✓ Success: {counts['success']} (verified: {counts['verified']})")
    print(f"  ✗ Failed: {counts['failed']}")