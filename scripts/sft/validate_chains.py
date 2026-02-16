"""
Validate Med-PRM chains from generated JSONL files.
Provides detailed statistics and identifies common issues.

Usage:
    python scripts/sft/validate_chains.py data_processed/medprm_chains/medprm_chains_*.jsonl
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class ChainAnalyzer:
    """Analyze and validate Med-PRM chains."""
    
    REQUIRED_SECTIONS = [
        "[CLINICAL FINDINGS]",
        "[CLINICAL PRINCIPLE]",
        "[REASONING]",
        "[ERROR_TYPE]",
        "[LOCATION]",
        "[CORRECTION]",
        "final_answer:"
    ]
    
    BANNED_PHRASES = [
        "might", "possibly", "could be", "suggests", "may be",
        "perhaps", "let me consider", "i think", "it seems",
        "appears to", "seems to", "could suggest", "may suggest"
    ]
    
    def __init__(self):
        self.chains = []
        self.validation_results = []
    
    def load_chains(self, filepath: str) -> int:
        """Load chains from JSONL file."""
        count = 0
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    self.chains.append(json.loads(line))
                    count += 1
        return count
    
    def validate_chain(self, chain: Dict) -> Tuple[bool, List[str]]:
        """Validate a single chain."""
        errors = []
        text = chain.get('reasoning_chain', '')
        expected_label = chain.get('label', '')
        scenario = chain.get('scenario', '')
        
        if not text:
            return False, ["Empty reasoning_chain"]
        
        # Check sections present and in order
        last_pos = -1
        for section in self.REQUIRED_SECTIONS:
            pos = text.find(section)
            if pos == -1:
                errors.append(f"Missing: {section}")
            elif pos < last_pos:
                errors.append(f"Out of order: {section}")
            else:
                last_pos = pos
        
        # Check banned phrases in REASONING
        reasoning_match = re.search(r'\[REASONING\](.*?)\[ERROR_TYPE\]', text, re.DOTALL)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).lower()
            for phrase in self.BANNED_PHRASES:
                if phrase in reasoning_text:
                    errors.append(f"Banned phrase: '{phrase}'")
                    break
        
        # Check final_answer
        final_match = re.search(r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?', text, re.IGNORECASE)
        if final_match:
            found = final_match.group(1).upper()
            if found != expected_label:
                errors.append(f"Label mismatch: expected {expected_label}, got {found}")
        else:
            errors.append("Cannot parse final_answer")
        
        # Check token count
        token_est = len(text) // 4
        max_tokens = 768 if scenario == 'error' else 512
        if token_est > max_tokens:
            errors.append(f"Too long: ~{token_est} tokens (max {max_tokens})")
        
        # Check bullet points in findings
        findings_match = re.search(r'\[CLINICAL FINDINGS\](.*?)\[CLINICAL PRINCIPLE\]', text, re.DOTALL)
        if findings_match:
            bullets = findings_match.group(1).count('•')
            if bullets == 0:
                errors.append("No bullets in FINDINGS")
            elif bullets > 4:
                errors.append(f"Too many findings: {bullets}")
        
        return len(errors) == 0, errors
    
    def analyze_all(self) -> Dict:
        """Analyze all loaded chains."""
        results = {
            'total': len(self.chains),
            'valid': 0,
            'invalid': 0,
            'by_scenario': {},
            'by_role': {},
            'by_error_type': {},
            'common_issues': Counter(),
            'token_stats': {'min': float('inf'), 'max': 0, 'total': 0}
        }
        
        for chain in self.chains:
            scenario = chain.get('scenario', 'unknown')
            role = chain.get('role', 'unknown')
            error_type = chain.get('error_type', 'none')
            
            # Initialize counters
            if scenario not in results['by_scenario']:
                results['by_scenario'][scenario] = {'valid': 0, 'invalid': 0}
            if role not in results['by_role']:
                results['by_role'][role] = {'valid': 0, 'invalid': 0}
            if error_type not in results['by_error_type']:
                results['by_error_type'][error_type] = {'valid': 0, 'invalid': 0}
            
            # Validate
            is_valid, errors = self.validate_chain(chain)
            
            if is_valid:
                results['valid'] += 1
                results['by_scenario'][scenario]['valid'] += 1
                results['by_role'][role]['valid'] += 1
                results['by_error_type'][error_type]['valid'] += 1
            else:
                results['invalid'] += 1
                results['by_scenario'][scenario]['invalid'] += 1
                results['by_role'][role]['invalid'] += 1
                results['by_error_type'][error_type]['invalid'] += 1
                for err in errors:
                    # Simplify error for counting
                    err_type = err.split(':')[0] if ':' in err else err
                    results['common_issues'][err_type] += 1
            
            # Token stats
            text = chain.get('reasoning_chain', '')
            tokens = len(text) // 4
            results['token_stats']['min'] = min(results['token_stats']['min'], tokens)
            results['token_stats']['max'] = max(results['token_stats']['max'], tokens)
            results['token_stats']['total'] += tokens
            
            self.validation_results.append({
                'note_id': chain.get('note_id'),
                'is_valid': is_valid,
                'errors': errors
            })
        
        if results['total'] > 0:
            results['token_stats']['avg'] = results['token_stats']['total'] / results['total']
            results['validation_rate'] = results['valid'] / results['total'] * 100
        
        return results
    
    def print_report(self, results: Dict):
        """Print formatted analysis report."""
        print("\n" + "=" * 60)
        print("  MED-PRM CHAIN VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Total chains:    {results['total']}")
        print(f"   Valid chains:    {results['valid']} ({results.get('validation_rate', 0):.1f}%)")
        print(f"   Invalid chains:  {results['invalid']}")
        
        print(f"\n📏 TOKEN STATISTICS")
        print(f"   Min tokens:  ~{results['token_stats']['min']}")
        print(f"   Max tokens:  ~{results['token_stats']['max']}")
        print(f"   Avg tokens:  ~{results['token_stats'].get('avg', 0):.0f}")
        
        print(f"\n🎭 BY SCENARIO")
        for scenario, counts in results['by_scenario'].items():
            total = counts['valid'] + counts['invalid']
            rate = counts['valid'] / total * 100 if total > 0 else 0
            print(f"   {scenario:15} {counts['valid']:4}/{total:4} valid ({rate:.1f}%)")
        
        print(f"\n👤 BY ROLE")
        for role, counts in results['by_role'].items():
            total = counts['valid'] + counts['invalid']
            rate = counts['valid'] / total * 100 if total > 0 else 0
            print(f"   {role:15} {counts['valid']:4}/{total:4} valid ({rate:.1f}%)")
        
        print(f"\n🏷️  BY ERROR TYPE")
        for etype, counts in sorted(results['by_error_type'].items()):
            total = counts['valid'] + counts['invalid']
            rate = counts['valid'] / total * 100 if total > 0 else 0
            print(f"   {etype:15} {counts['valid']:4}/{total:4} valid ({rate:.1f}%)")
        
        if results['common_issues']:
            print(f"\n⚠️  COMMON ISSUES (top 10)")
            for issue, count in results['common_issues'].most_common(10):
                print(f"   {count:4}x  {issue}")
        
        print("\n" + "=" * 60)
    
    def export_invalid(self, output_path: str):
        """Export invalid chains for review."""
        invalid = [r for r in self.validation_results if not r['is_valid']]
        
        with open(output_path, 'w') as f:
            for item in invalid:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Exported {len(invalid)} invalid chains to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate Med-PRM chains")
    parser.add_argument('files', nargs='+', help='JSONL files to validate')
    parser.add_argument('--export-invalid', help='Export invalid chains to this file')
    args = parser.parse_args()
    
    analyzer = ChainAnalyzer()
    
    # Load all files
    total_loaded = 0
    for filepath in args.files:
        count = analyzer.load_chains(filepath)
        total_loaded += count
        logger.info(f"Loaded {count} chains from {filepath}")
    
    logger.info(f"Total chains loaded: {total_loaded}")
    
    # Analyze
    results = analyzer.analyze_all()
    
    # Print report
    analyzer.print_report(results)
    
    # Export invalid if requested
    if args.export_invalid:
        analyzer.export_invalid(args.export_invalid)


if __name__ == '__main__':
    main()
