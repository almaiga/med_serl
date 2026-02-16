"""
Process ALL notes and generate benign changes using rule-based methods only.
Saves successful changes and logs failures for later processing.

Usage:
  python process_all_benign_changes.py                    # Start fresh
  python process_all_benign_changes.py --resume           # Resume from existing files
  python process_all_benign_changes.py --skip 452         # Skip first N notes
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import random
import argparse

from inject_benign_changes import BenignChangeGenerator, parse_extraction


def get_processed_note_ids(success_path, failed_path):
    """Get set of note_ids that have already been processed."""
    processed = set()
    
    # Read successful notes
    if success_path.exists():
        with open(success_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'note_id' in record:
                        processed.add(record['note_id'])
                except:
                    continue
    
    # Read failed notes
    if failed_path.exists():
        with open(failed_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'note_id' in record:
                        processed.add(record['note_id'])
                except:
                    continue
    
    return processed


def main(resume=False, skip_count=0, resume_files=None):
    # Paths
    data_path = Path("data_processed/parsed_medical_note/extractions.jsonl")
    output_dir = Path("data_processed/benign_changes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle resume mode
    if resume and resume_files:
        success_path = Path(resume_files['success'])
        failed_path = Path(resume_files['failed'])
        summary_path = Path(resume_files['summary'])
        print(f"RESUME MODE: Appending to existing files")
        print(f"  Success: {success_path}")
        print(f"  Failed: {failed_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        success_path = output_dir / f"benign_changes_{timestamp}.jsonl"
        failed_path = output_dir / f"failed_notes_{timestamp}.jsonl"
        summary_path = output_dir / f"processing_summary_{timestamp}.json"
    
    # Load all notes
    print("Loading notes...")
    notes = []
    with open(data_path, 'r') as f:
        for line in f:
            notes.append(json.loads(line))
    
    print(f"Total notes in dataset: {len(notes)}")
    
    # Get already processed note_ids if resuming
    processed_ids = set()
    if resume:
        processed_ids = get_processed_note_ids(success_path, failed_path)
        print(f"Already processed: {len(processed_ids)} notes")
        # Filter out already processed notes
        notes = [n for n in notes if n.get('note_id') not in processed_ids]
    
    # Apply skip if specified
    if skip_count > 0:
        notes = notes[skip_count:]
        print(f"Skipping first {skip_count} notes")
    
    print(f"Notes to process: {len(notes)}")
    
    # Initialize generator
    config_path = Path("configs/prompts/benign_change_prompt.json")
    generator = BenignChangeGenerator(config_path, model="gpt-4o-mini")
    
    # Exclude LLM-dependent types
    EXCLUDE_TYPES = {"logical_restatement", "irrelevant_correlation"}
    generator.change_types = [ct for ct in generator.change_types if ct not in EXCLUDE_TYPES]
    
    print(f"Change types (rule-based only): {generator.change_types}\n")
    
    # Assign change types
    assignments = generator.assign_change_types(notes)
    random.shuffle(assignments)
    
    # Open output files for incremental writing (append mode if resuming)
    file_mode = 'a' if resume else 'w'
    success_file = open(success_path, file_mode)
    failed_file = open(failed_path, file_mode)
    
    # Track stats
    stats = defaultdict(lambda: {"success": 0, "failed": 0, "verified": 0})
    success_count = 0
    failed_count = 0
    
    print("Processing all notes (saving incrementally)...")
    try:
        for item in tqdm(assignments, desc="Generating benign changes"):
            note_obj = item["note"]
            change_type = item["change_type"]
            note_text = note_obj.get("correct_note", "")
            extraction = parse_extraction(note_obj.get("extraction", {}))
            note_type = note_obj.get("error_type", "")
            note_id = note_obj.get("note_id", "unknown")
            
            # Find target term
            target_term = generator.find_target_term(note_text, extraction, change_type, note_type=note_type)
            
            # Fallback strategy: try alternative change types if no target found
            original_change_type = change_type
            if not target_term:
                allowed_types = generator.get_allowed_change_types(note_type)
                for alt_type in allowed_types:
                    if alt_type != change_type:
                        alt_target = generator.find_target_term(note_text, extraction, alt_type, note_type=note_type)
                        if alt_target:
                            change_type = alt_type
                            target_term = alt_target
                            break
            
            if not target_term:
                # No target term found - write immediately
                failed_record = {
                    "note_id": note_id,
                    "original_note": note_text,
                    "extraction": note_obj.get("extraction", {}),
                    "error_type": note_type,
                    "attempted_change_type": original_change_type,
                    "failure_reason": "no_target_term",
                    "error": f"No target term found for change type '{original_change_type}' or fallbacks"
                }
                failed_file.write(json.dumps(failed_record) + '\n')
                failed_file.flush()  # Ensure it's written to disk
                stats[original_change_type]["failed"] += 1
                failed_count += 1
                continue
            
            # Get verified replacement
            verified = generator.get_verified_replacement(target_term, change_type)
            
            if not verified or not verified.get("verified"):
                # No verified replacement available - write immediately
                failed_record = {
                    "note_id": note_id,
                    "original_note": note_text,
                    "extraction": note_obj.get("extraction", {}),
                    "error_type": note_type,
                    "attempted_change_type": change_type,
                    "target_term": target_term,
                    "failure_reason": "no_verified_replacement",
                    "error": f"No verified replacement for '{target_term}' with change type '{change_type}'"
                }
                failed_file.write(json.dumps(failed_record) + '\n')
                failed_file.flush()
                stats[change_type]["failed"] += 1
                failed_count += 1
                continue
            
            # Generate the change
            result = generator.generate_change(note_text, change_type, extraction, allow_fallback=False)
            result["note_id"] = note_id
            result["error_type"] = note_type
            
            if result.get("change_made"):
                # Success - write immediately
                success_file.write(json.dumps(result) + '\n')
                success_file.flush()
                stats[change_type]["success"] += 1
                success_count += 1
                if result.get("verified"):
                    stats[change_type]["verified"] += 1
            else:
                # Change generation failed - write immediately
                failed_record = {
                    "note_id": note_id,
                    "original_note": note_text,
                    "extraction": note_obj.get("extraction", {}),
                    "error_type": note_type,
                    "attempted_change_type": change_type,
                    "target_term": target_term,
                    "failure_reason": "generation_failed",
                    "error": result.get("error", "Unknown generation error")
                }
                failed_file.write(json.dumps(failed_record) + '\n')
                failed_file.flush()
                stats[change_type]["failed"] += 1
                failed_count += 1
    
    finally:
        # Always close files even if interrupted
        success_file.close()
        failed_file.close()
        print(f"\nFiles closed. Saved {success_count} successful and {failed_count} failed notes.")
    
    # Create summary
    summary = {
        "timestamp": timestamp,
        "total_notes": len(notes),
        "successful": success_count,
        "failed": failed_count,
        "success_rate": f"{success_count / len(notes) * 100:.1f}%",
        "output_files": {
            "successful_changes": str(success_path),
            "failed_notes": str(failed_path)
        },
        "stats_by_type": dict(stats)
    }
    
    # Breakdown failures by reason - read from failed file
    failure_reasons = defaultdict(int)
    with open(failed_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            failure_reasons[record.get("failure_reason", "unknown")] += 1
    summary["failure_breakdown"] = dict(failure_reasons)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"\nTotal notes processed: {len(notes)}")
    print(f"✓ Successful: {success_count} ({success_count/len(notes)*100:.1f}%)")
    print(f"✗ Failed: {failed_count} ({failed_count/len(notes)*100:.1f}%)")
    
    print("\n" + "-"*40)
    print("STATS BY CHANGE TYPE")
    print("-"*40)
    for change_type in sorted(stats.keys()):
        counts = stats[change_type]
        total = counts["success"] + counts["failed"]
        rate = counts["success"] / total * 100 if total > 0 else 0
        print(f"\n{change_type}:")
        print(f"  ✓ Success: {counts['success']} (verified: {counts['verified']})")
        print(f"  ✗ Failed: {counts['failed']}")
        print(f"  Rate: {rate:.1f}%")
    
    print("\n" + "-"*40)
    print("FAILURE BREAKDOWN")
    print("-"*40)
    for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    print("\n" + "-"*40)
    print("OUTPUT FILES")
    print("-"*40)
    print(f"  Successful changes: {success_path}")
    print(f"  Failed notes: {failed_path}")
    print(f"  Summary: {summary_path}")
    
    return success_count, failed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process medical notes and generate benign changes")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output files (finds most recent)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N notes (alternative to resume)")
    parser.add_argument("--success-file", type=str, help="Path to existing success file (for resume)")
    parser.add_argument("--failed-file", type=str, help="Path to existing failed file (for resume)")
    
    args = parser.parse_args()
    
    resume_files = None
    if args.resume:
        # Find most recent files if not specified
        if args.success_file and args.failed_file:
            resume_files = {
                'success': args.success_file,
                'failed': args.failed_file,
                'summary': args.success_file.replace('benign_changes_', 'processing_summary_').replace('.jsonl', '.json')
            }
        else:
            # Auto-detect most recent files
            output_dir = Path("data_processed/benign_changes")
            success_files = sorted(output_dir.glob("benign_changes_*.jsonl"))
            if success_files:
                latest_success = success_files[-1]
                timestamp = latest_success.stem.replace("benign_changes_", "")
                resume_files = {
                    'success': str(latest_success),
                    'failed': str(output_dir / f"failed_notes_{timestamp}.jsonl"),
                    'summary': str(output_dir / f"processing_summary_{timestamp}.json")
                }
                print(f"Auto-detected files to resume from:")
                print(f"  {resume_files['success']}")
                print(f"  {resume_files['failed']}")
            else:
                print("ERROR: No existing files found to resume from.")
                print("Run without --resume to start fresh.")
                exit(1)
    
    main(resume=args.resume, skip_count=args.skip, resume_files=resume_files)

