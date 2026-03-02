#!/usr/bin/env python3
import json

print("=== Jan 14 2026 (Post-SFT Qwen3-4B LoRA) ===")
results = [json.loads(l) for l in open("results/Qwen3-4B Lora Results Jan 14 2026.jsonl")]
total = len(results)
correct = sum(1 for r in results if r["correct"])
tp = sum(1 for r in results if r["predicted_label"] == "INCORRECT" and r["ground_truth_label"] == "INCORRECT")
fp = sum(1 for r in results if r["predicted_label"] == "INCORRECT" and r["ground_truth_label"] == "CORRECT")
tn = sum(1 for r in results if r["predicted_label"] == "CORRECT" and r["ground_truth_label"] == "CORRECT")
fn = sum(1 for r in results if r["predicted_label"] == "CORRECT" and r["ground_truth_label"] == "INCORRECT")
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
print(f"Total: {total}, Acc: {correct/total:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"Predicted CORRECT: {sum(1 for r in results if r['predicted_label']=='CORRECT')}")
print(f"Predicted INCORRECT: {sum(1 for r in results if r['predicted_label']=='INCORRECT')}")

print()
print("=== Dec 10 2025 (Pre-SFT baseline) ===")
results2 = [json.loads(l) for l in open("results/inference/Final MS Results Dec 10 2025 (1)_cleaned.jsonl")]
total2 = len(results2)
labels = set(r["predicted_label"] for r in results2[:20])
gt_labels = set(r["ground_truth_label"] for r in results2[:20])
print(f"Total: {total2}")
print(f"Predicted labels (sample): {labels}")
print(f"GT labels (sample): {gt_labels}")
correct2 = sum(1 for r in results2 if r["correct"])
print(f"Correct: {correct2}/{total2} = {correct2/total2:.3f}")
