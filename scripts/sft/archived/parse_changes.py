import json
import re
import difflib
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform colored output
try:
    colorama_init()
    HAS_COLORAMA = True
except Exception:
    HAS_COLORAMA = False


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_raw_output(raw_output: str) -> Dict[str, Optional[str]]:
    """
    Parses the raw output from the model to extract the generated note,
    the final answer, and the structured `changes_made` block.

    Args:
        raw_output: The full string output from the language model.

    Returns:
        A dictionary containing:
        - 'generated_note': The text of the modified note.
        - 'final_answer': The final answer ('CORRECT' or 'INCORRECT').
        - 'changes_made': The JSON object describing the changes as a dict.
        - 'error': A string describing any parsing error, if one occurred.
    """
    output = {
        "generated_note": None,
        "final_answer": None,
        "changes_made": None,
        "error": None,
    }

    try:
        # Strip think blocks first
        clean_output = strip_think_blocks(raw_output)
        
        # Extract generated_note - try multiple patterns
        note_match = re.search(
            r"generated_note:\s*(.*?)(?=final_answer:|changes_made:|$)", 
            clean_output, re.DOTALL | re.IGNORECASE
        )
        if note_match:
            note_text = note_match.group(1).strip()
            # Remove any trailing markers
            note_text = re.sub(r'\s*final_answer:.*$', '', note_text, flags=re.DOTALL | re.IGNORECASE)
            note_text = re.sub(r'\s*changes_made:.*$', '', note_text, flags=re.DOTALL | re.IGNORECASE)
            output["generated_note"] = note_text.strip()
        else:
            output["error"] = "Could not find 'generated_note:' section."
            return output

        # Extract final_answer - try multiple patterns
        answer_patterns = [
            r'final_answer:\s*"(CORRECT|INCORRECT)"',
            r'final_answer:\s*(CORRECT|INCORRECT)',
            r'Answer:\s*(CORRECT|INCORRECT)',
        ]
        for pattern in answer_patterns:
            answer_match = re.search(pattern, clean_output, re.IGNORECASE)
            if answer_match:
                output["final_answer"] = answer_match.group(1).upper()
                break
        
        if not output["final_answer"]:
            output["error"] = "Could not find 'final_answer:' section."
            # Don't return early - we may still have useful data

        # Extract changes_made JSON block
        changes_match = re.search(
            r"changes_made:\s*(.*?)(?=$)", clean_output, re.DOTALL | re.IGNORECASE
        )
        if changes_match:
            json_str = changes_match.group(1).strip()
            try:
                # Clean up potential code block markers
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                elif json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                
                # Find the JSON object
                brace_start = json_str.find('{')
                if brace_start >= 0:
                    # Find matching closing brace
                    depth = 0
                    end_idx = brace_start
                    for i, char in enumerate(json_str[brace_start:], brace_start):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                end_idx = i + 1
                                break
                    json_str = json_str[brace_start:end_idx]
                
                output["changes_made"] = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to extract useful info even if JSON is malformed
                output["error"] = f"Failed to decode JSON from changes_made block: {e}"
        # changes_made is optional - don't set error if not found

    except Exception as e:
        output["error"] = f"An unexpected error occurred during parsing: {e}"

    return output


def compute_word_level_diff(original: str, modified: str) -> Dict:
    """
    Compute word-level differences between two texts.
    
    Returns:
        Dict with 'added', 'removed', 'changed' word counts and details.
    """
    orig_words = original.split()
    mod_words = modified.split()
    
    matcher = difflib.SequenceMatcher(None, orig_words, mod_words)
    
    added = []
    removed = []
    changed = []
    
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'replace':
            removed.extend(orig_words[i1:i2])
            added.extend(mod_words[j1:j2])
            changed.append({
                'original': ' '.join(orig_words[i1:i2]),
                'modified': ' '.join(mod_words[j1:j2]),
            })
        elif opcode == 'delete':
            removed.extend(orig_words[i1:i2])
        elif opcode == 'insert':
            added.extend(mod_words[j1:j2])
    
    return {
        'added_count': len(added),
        'removed_count': len(removed),
        'changed_count': len(changed),
        'total_word_edits': len(added) + len(removed),
        'added_words': added,
        'removed_words': removed,
        'changes': changed,
    }


def find_changed_sentences(original: str, modified: str) -> List[Dict]:
    """
    Find sentences that differ between original and modified text.
    
    Returns:
        List of dicts with 'original_sentence', 'modified_sentence', 'word_diff'.
    """
    # Split into sentences (simple approach)
    def split_sentences(text: str) -> List[str]:
        # Split on period followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    orig_sentences = split_sentences(original)
    mod_sentences = split_sentences(modified)
    
    changed = []
    
    # Use sequence matcher to align sentences
    matcher = difflib.SequenceMatcher(None, orig_sentences, mod_sentences)
    
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'replace':
            for orig_sent, mod_sent in zip(orig_sentences[i1:i2], mod_sentences[j1:j2]):
                word_diff = compute_word_level_diff(orig_sent, mod_sent)
                changed.append({
                    'original_sentence': orig_sent,
                    'modified_sentence': mod_sent,
                    'word_diff': word_diff,
                })
        elif opcode == 'delete':
            for sent in orig_sentences[i1:i2]:
                changed.append({
                    'original_sentence': sent,
                    'modified_sentence': None,
                    'word_diff': {'removed_count': len(sent.split())},
                })
        elif opcode == 'insert':
            for sent in mod_sentences[j1:j2]:
                changed.append({
                    'original_sentence': None,
                    'modified_sentence': sent,
                    'word_diff': {'added_count': len(sent.split())},
                })
    
    return changed

def get_change_diff(
    original_note: str,
    generated_note: str,
    changes_made: Optional[Dict] = None,
    context_lines: int = 2,
    colorize: bool = True
) -> str:
    """
    Generates a unified diff to visualize the changes between the original and
    generated notes. Also provides word-level change analysis.

    Args:
        original_note: The original note text.
        generated_note: The generated note text.
        changes_made: The parsed `changes_made` object from the model output.
        context_lines: Number of context lines to show around the change.
        colorize: Whether to add ANSI color codes for terminal output.

    Returns:
        A string containing the formatted diff and change summary.
    """
    output_lines = []
    
    # Use model-provided change info if available
    if changes_made and "original_sentence" in changes_made and "modified_sentence" in changes_made:
        original_sentence = changes_made["original_sentence"]
        modified_sentence = changes_made["modified_sentence"]
        words_changed = changes_made.get("words_changed", "unknown")
        
        output_lines.append("=" * 60)
        output_lines.append("MODEL-REPORTED CHANGE:")
        output_lines.append("=" * 60)
        
        if colorize and HAS_COLORAMA:
            output_lines.append(f"{Fore.RED}- {original_sentence}{Style.RESET_ALL}")
            output_lines.append(f"{Fore.GREEN}+ {modified_sentence}{Style.RESET_ALL}")
            output_lines.append(f"{Fore.CYAN}  Words changed: {words_changed}{Style.RESET_ALL}")
        else:
            output_lines.append(f"- {original_sentence}")
            output_lines.append(f"+ {modified_sentence}")
            output_lines.append(f"  Words changed: {words_changed}")
        
        output_lines.append("")
    
    # Always compute and show programmatic diff for verification
    output_lines.append("=" * 60)
    output_lines.append("PROGRAMMATIC DIFF ANALYSIS:")
    output_lines.append("=" * 60)
    
    # Compute word-level stats
    word_diff = compute_word_level_diff(original_note, generated_note)
    output_lines.append(f"Total word edits: {word_diff['total_word_edits']}")
    output_lines.append(f"  Added: {word_diff['added_count']}, Removed: {word_diff['removed_count']}")
    
    if word_diff['changes']:
        output_lines.append("\nWord-level changes:")
        for change in word_diff['changes'][:5]:  # Show first 5
            if colorize and HAS_COLORAMA:
                output_lines.append(f"  {Fore.RED}{change['original']}{Style.RESET_ALL} → {Fore.GREEN}{change['modified']}{Style.RESET_ALL}")
            else:
                output_lines.append(f"  '{change['original']}' → '{change['modified']}'")
    
    # Find changed sentences
    changed_sentences = find_changed_sentences(original_note, generated_note)
    output_lines.append(f"\nSentences modified: {len(changed_sentences)}")
    
    if changed_sentences:
        output_lines.append("\nSentence-level changes:")
        for i, sent_change in enumerate(changed_sentences[:3], 1):  # Show first 3
            output_lines.append(f"\n  Change {i}:")
            orig = sent_change.get('original_sentence', '(none)')
            mod = sent_change.get('modified_sentence', '(none)')
            if colorize and HAS_COLORAMA:
                output_lines.append(f"    {Fore.RED}- {orig[:100]}...{Style.RESET_ALL}" if len(orig) > 100 else f"    {Fore.RED}- {orig}{Style.RESET_ALL}")
                output_lines.append(f"    {Fore.GREEN}+ {mod[:100]}...{Style.RESET_ALL}" if len(mod) > 100 else f"    {Fore.GREEN}+ {mod}{Style.RESET_ALL}")
            else:
                output_lines.append(f"    - {orig[:100]}..." if len(orig) > 100 else f"    - {orig}")
                output_lines.append(f"    + {mod[:100]}..." if len(mod) > 100 else f"    + {mod}")
    
    # Quality assessment
    output_lines.append("\n" + "=" * 60)
    output_lines.append("QUALITY ASSESSMENT:")
    output_lines.append("=" * 60)
    
    is_minimal = word_diff['total_word_edits'] <= 6 and len(changed_sentences) <= 1
    if is_minimal:
        if colorize and HAS_COLORAMA:
            output_lines.append(f"{Fore.GREEN}✓ PASS: Minimal edit (≤6 words in ≤1 sentence){Style.RESET_ALL}")
        else:
            output_lines.append("✓ PASS: Minimal edit (≤6 words in ≤1 sentence)")
    else:
        if colorize and HAS_COLORAMA:
            output_lines.append(f"{Fore.YELLOW}⚠ WARNING: Edit may be too extensive{Style.RESET_ALL}")
            output_lines.append(f"  Expected: ≤6 word edits in 1 sentence")
            output_lines.append(f"  Got: {word_diff['total_word_edits']} word edits in {len(changed_sentences)} sentences")
        else:
            output_lines.append("⚠ WARNING: Edit may be too extensive")
            output_lines.append(f"  Expected: ≤6 word edits in 1 sentence")
            output_lines.append(f"  Got: {word_diff['total_word_edits']} word edits in {len(changed_sentences)} sentences")
    
    return "\n".join(output_lines)


def format_change_log(
    note_id: str,
    original_note: str,
    generated_note: str,
    changes_made: Optional[Dict] = None,
    filter_passed: bool = True,
    filter_reason: Optional[str] = None,
) -> Dict:
    """
    Create a structured log entry for a change, including programmatic analysis.
    
    Returns:
        Dict suitable for JSONL logging with change analysis.
    """
    word_diff = compute_word_level_diff(original_note, generated_note)
    changed_sentences = find_changed_sentences(original_note, generated_note)
    
    return {
        "note_id": note_id,
        "filter_passed": filter_passed,
        "filter_reason": filter_reason,
        "model_reported_changes": changes_made,
        "programmatic_analysis": {
            "total_word_edits": word_diff["total_word_edits"],
            "words_added": word_diff["added_count"],
            "words_removed": word_diff["removed_count"],
            "sentences_changed": len(changed_sentences),
            "is_minimal_edit": word_diff["total_word_edits"] <= 6 and len(changed_sentences) <= 1,
            "word_changes": word_diff["changes"][:5],  # First 5 changes
            "sentence_changes": [
                {
                    "original": s.get("original_sentence", "")[:200],
                    "modified": s.get("modified_sentence", "")[:200],
                }
                for s in changed_sentences[:3]  # First 3 sentence changes
            ],
        },
    }


if __name__ == "__main__":
    # Example usage demonstrating how to use the parsing and diffing functions.
    # This sample output is what we expect from the newly prompted model.
    sample_raw_output = r"""
<think>
The user wants me to make a minimal edit. I need to:
1. Select one sentence
2. Change 1-3 words only
3. Preserve meaning

I'll change "started" to "initiated" - single word, same meaning.
</think>
generated_note:
A previously healthy 18-year-old army recruit is brought to a military treatment facility because of a 3-week history of right foot pain. He recently initiated basic infantry training and has been running several kilometers daily. Initially, the pain only occurred while running, but now it is also present at rest. The pain is located diffusely over the right forefoot. Vital signs are within normal range. Examination shows mild swelling over the distal right forefoot. Pressing the metatarsal of the third right toe results in pain. He walks with an antalgic gait. The remainder of the examination shows no abnormalities. An x-ray of the right foot shows a slight loss of cortical density and callus formation at the third metatarsal shaft. Rest is advised and acetaminophen is ordered.

final_answer: "CORRECT"

changes_made:
{"original_sentence": "He recently started basic infantry training and has been running several kilometers daily.", "modified_sentence": "He recently initiated basic infantry training and has been running several kilometers daily.", "words_changed": "started → initiated"}
"""

    original_note = "A previously healthy 18-year-old army recruit is brought to a military treatment facility because of a 3-week history of right foot pain. He recently started basic infantry training and has been running several kilometers daily. Initially, the pain only occurred while running, but now it is also present at rest. The pain is located diffusely over the right forefoot. Vital signs are within normal range. Examination shows mild swelling over the distal right forefoot. Pressing the metatarsal of the third right toe results in pain. He walks with an antalgic gait. The remainder of the examination shows no abnormalities. An x-ray of the right foot shows a slight loss of cortical density and callus formation at the third metatarsal shaft. Rest is advised and acetaminophen is ordered."

    print("=" * 70)
    print("DEMO: PARSING AND ANALYZING MODEL OUTPUT")
    print("=" * 70)
    
    print("\n--- 1. PARSING RAW MODEL OUTPUT ---")
    parsed_data = parse_raw_output(sample_raw_output)
    print(f"Generated note found: {bool(parsed_data['generated_note'])}")
    print(f"Final answer: {parsed_data['final_answer']}")
    print(f"Changes made: {json.dumps(parsed_data['changes_made'], indent=2) if parsed_data['changes_made'] else 'None'}")
    print(f"Parsing errors: {parsed_data['error']}")
    
    print("\n" + "=" * 70)

    if parsed_data["generated_note"]:
        print("\n--- 2. CHANGE DIFF ANALYSIS ---")
        diff_output = get_change_diff(
            original_note,
            parsed_data["generated_note"],
            parsed_data["changes_made"],
            colorize=True
        )
        print(diff_output)
        
        print("\n--- 3. STRUCTURED LOG ENTRY ---")
        log_entry = format_change_log(
            note_id="demo-001",
            original_note=original_note,
            generated_note=parsed_data["generated_note"],
            changes_made=parsed_data["changes_made"],
            filter_passed=True,
        )
        print(json.dumps(log_entry, indent=2))

    # Now test with a BAD example (too many changes)
    print("\n" + "=" * 70)
    print("DEMO: DETECTING OVER-EDITING")
    print("=" * 70)
    
    bad_generated_note = "A previously healthy 18-year-old army recruit is admitted to a military medical facility due to a 3-week history of right foot pain. He began basic infantry training and has been running multiple kilometers daily. Initially, the pain was limited to running, but it is now also present at rest."
    
    bad_parsed = {"generated_note": bad_generated_note, "changes_made": None}
    
    print("\n--- OVER-EDITED NOTE ANALYSIS ---")
    diff_output = get_change_diff(
        original_note[:300],  # Just first part for demo
        bad_generated_note,
        None,
        colorize=True
    )
    print(diff_output)

    print("\n" + "=" * 70)
    print("INTEGRATION GUIDE")
    print("=" * 70)
    integration_guide = """
To integrate this into `quick_infer_qwen3_4b_lora.py`:

1. Import the functions:
   from parse_changes import parse_raw_output, get_change_diff, format_change_log

2. After getting model output in run_selfplay_loop():
   
   parsed = parse_raw_output(generated)
   candidate_note = parsed.get("generated_note")
   
3. Log with change analysis:
   
   change_log = format_change_log(
       note_id=state["note_id"],
       original_note=state["original_note"],
       generated_note=candidate_note,
       changes_made=parsed.get("changes_made"),
       filter_passed=filter_meta["passed"],
       filter_reason=filter_meta.get("reason"),
   )
   
4. Print diff for debugging:
   
   if candidate_note:
       diff = get_change_diff(
           state["original_note"], 
           candidate_note, 
           parsed.get("changes_made")
       )
       print(diff)

5. Use the new prompts file:
   --injector-prompt-file configs/prompts/error_injection_prompts_v2.json
   
6. IMPORTANT: For minimal edits, consider disabling thinking mode:
   --thinking-budget 0
   
   Or use the /no_think soft switch in the prompt for Qwen3.
"""
    print(integration_guide)
