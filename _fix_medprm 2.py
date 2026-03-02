"""One-shot script to update medprm_prompts.json. Delete after running."""
import json

with open('configs/prompts/sft/medprm_prompts.json') as f:
    data = json.load(f)

# === assessor_incorrect: final_answer: "INCORRECT" → final_answer: <error_sentence_id> ===
s = data['assessor_incorrect']['system']
s = s.replace(
    'final_answer: "INCORRECT"\nExplanation: One sentence explaining the mechanistic contradiction.',
    'final_answer: <error_sentence_id>\nExplanation: One sentence explaining the mechanistic contradiction.\n\nIMPORTANT: Replace <error_sentence_id> with the actual 1-indexed sentence number containing the error.'
)
s = s.replace(
    'final_answer: "INCORRECT"\nExplanation: Chorea indicates striatal degeneration',
    'final_answer: 4\nExplanation: Chorea indicates striatal degeneration'
)
data['assessor_incorrect']['system'] = s

# === injector_error: final_answer → sentence format ===
s2 = data['injector_error']['system']
s2 = s2.replace(
    'final_answer: "INCORRECT"\nExplanation: One sentence stating the mechanistic distinction',
    'final_answer: <sentence_id>. <modified sentence with injected error>\nExplanation: One sentence stating the mechanistic distinction'
)
s2 = s2.replace(
    'final_answer: "INCORRECT"\nExplanation: Error exploits availability heuristic',
    'final_answer: 8. Oral erythromycin is administered to all family members and Tdap vaccination is administered to the father and mother.\nExplanation: Error exploits availability heuristic'
)
data['injector_error']['system'] = s2

# === metadata ===
data['metadata']['version'] = '2.0.0'
data['metadata']['updated'] = '2026-02-26'
data['metadata']['description'] = 'Med-PRM adapted prompts with sentence-level input/output. Notes are pre-numbered (1-indexed). Assessor outputs CORRECT or sentence number. Injector outputs N. <modified sentence>.'

with open('configs/prompts/sft/medprm_prompts.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print('Done - verify with: python3 -c "import json; d=json.load(open(\'configs/prompts/sft/medprm_prompts.json\')); print(d[\'metadata\'][\'version\'])"')
