import glob
import json
import os

log_files = glob.glob("results/self_play/interactions/*.jsonl")
if not log_files:
    print("No log files found.")
else:
    latest_file = max(log_files, key=os.path.getmtime)
    print(f"Latest log file: {latest_file}")
    with open(latest_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                print("-----")
                print(f"Mode: {data.get('mode')}")
                print(f"Judge Verdict: {data.get('judge_verdict')}")
                print(f"Assessor Reward: {data.get('assessor_reward')}")
                spans = data.get('turn_reward_spans')
                if spans:
                    print(f"Turn Spans:")
                    for span in spans:
                        print(f"  - Role: {span.get('role')}, Reward: {span.get('reward', span.get('raw_reward'))}, Raw: {span.get('raw_reward', 'N/A')}")
            except Exception as e:
                pass
