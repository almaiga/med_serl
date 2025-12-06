Project Specification: Medical SeRL (MedSeRL)

1. Executive Summary

Goal: Build a self-improving "Medical Doctor" agent capable of reasoning through clinical notes to detect, classify, and correct medical errors.
Core Methodology: Adapt the SeRL (Self-Play Reinforcement Learning) framework to the MEDEC benchmark, using a single small-scale medical LLM (MedGemma-4B) to play both "Scribe" (Generator) and "Doctor" (Verifier).

2. Key Resources & Assets

A. Literature & Baselines

SeRL Paper: arXiv:2505.20347

Core Concept: Bootstrapping LLM training with limited data via self-instruction and self-rewarding.

Algorithm: Reinforce++ (Hu et al., 2025).

SeRL Codebase: GitHub: wantbook-book/SeRL

MEDEC Paper: arXiv:2412.19260

Core Concept: A benchmark for medical error detection covering 5 types (Diagnosis, Management, Treatment, Pharmacotherapy, Causal Organism).

MEDEC Dataset: GitHub: abachaa/MEDEC

Target Data: MEDEC-MS subset (Contrastive pairs of Correct vs. Error notes).

B. Model Choice

Base Model: google/medgemma-4b-it (HuggingFace)

Rationale: * 4B Parameters: Fits on consumer GPUs (e.g., RTX 3090/4090) for fast self-play iteration.

Medical Pre-training: Fine-tuned on EHRs and PubMed; possesses the necessary "Medical Common Sense" priors that RL cannot teach from scratch.

3. Architecture: The Single-Model Self-Play Loop

Unlike standard distillation, we use one model switching between two personas.

Persona A: The Adversarial Scribe (Generator)

Role: Generates training batches by mixing static ground truth with dynamic synthetic challenges.

Input Data Pool: MEDEC Dataset (containing both Error=1 and Error=0 samples).

Sampling Policy (The "Balanced Quadrant"):

Augmented Ground Truth (25%): Select an existing MEDEC note with Error=1.

Action: Reword the patient demographics and narrative flow surrounding the error, but keep the error logic identical.

Goal (Generalization): Prevent the model from memorizing specific text patterns (e.g., "The note about Mr. Smith always has a dosage error").

Ground Truth: Error (Persistent).

Augmented Safe (25%): Select an existing MEDEC note with Error=0 (or corrected_text).

Action: Paraphrase the clinical narrative (e.g., change "Patient denies chest pain" to "No reports of angina").

Goal (Stability): Ensure the model recognizes valid medical logic even when phrased differently.

Ground Truth: Clean (Persistent).

Synthetic Decoy (25%): Select a Clean Note (Error=0) -> Apply Cosmetic Noise.

Action: Inject typos, weird date formats, or synonyms ("Tylenol" vs "Acetaminophen") without changing clinical logic.

Goal (Robustness): Teach the model that messy text does not equal medically incorrect text (Hard Negative).

Ground Truth: Clean (Despite looks).

Synthetic Injection (25%): Select a Clean Note (Error=0) -> Apply Lethal Injection.

Action: LLM rewrites note to insert a new, specific error (e.g., "Change dosage to toxic level") not found in the original dataset.

Goal (Expansion): Expose model to new error types and subtle logic traps.

Ground Truth: Error (New).

Output: (Note_Text, Ground_Truth_Label, Error_Type)

Persona B: The Doctor (Policy)

Role: The agent being trained.

Input: Note_Text (blind to label).

Mechanism: Uses Chain of Thought (CoT) reasoning.

Format: <thinking> ... analysis ... </thinking> <verdict> ... </verdict>

Objective: Correctly classify the note as "Error" or "Clean" and localize the issue.

4. Implementation Plan

Phase 1: SFT Warm-Up (Format Alignment)

Objective: Teach MedGemma the <thinking> structure and the definition of the 5 MEDEC error types.

Data: MEDEC Training Set (Error Flag == 1).

Target Output:

<thinking>
Scanning Pharmacotherapy... Metformin dosage at 5000mg is toxic. Max daily is usually 2000-2550mg.
</thinking>
<verdict>Error: Pharmacotherapy</verdict>




Action: 1 Epoch SFT using openrlhf.trainer.SFTTrainer (or standard HF Trainer).

Phase 2: The SeRL Loop (Reinforce++)

Objective: Optimize the "Doctor" policy using self-generated data.

A. Deterministic Reward Function

Since the Scribe knows what it injected (or what the dataset label is), we do not need Majority Voting.
$$ R = R_{outcome} + R_{structure} + R_{penalty} $$

def calculate_reward(model_output, ground_truth):
    reward = 0.0
    
    # 1. Structural Reward (Did it think?)
    if "<thinking>" in model_output: 
        reward += 0.1

    # 2. Outcome Reward
    if ground_truth['has_error']:
        # Case: Sick Note (Real or Synthetic)
        if ground_truth['error_metadata'] in model_output:
            reward += 1.0   # Found the specific error
        elif "No Error" in model_output:
            reward -= 1.0   # False Negative (Missed Diagnosis)
    else:
        # Case: Healthy Note (Real or Decoy)
        if "No Clinical Error" in model_output:
            reward += 1.0   # Correctly ignored noise
        elif "Error" in model_output:
            reward -= 1.5   # False Positive (Alert Fatigue Penalty)

    return reward




B. Optimization

Algorithm: Reinforce++ (As per SeRL/OpenRLHF implementation).

Library: OpenRLHF (Recommended for SeRL reproduction).

Cycle:

Generate Batch (Mix of 4 types defined in Persona A).

Rollout Doctor policy on these notes.

Compute Rewards.

Update Weights using openrlhf PPO/Reinforce engine.

Repeat.

5. Coding Specs (For AI Assistant)

File Structure

src/data_processor.py: Loads MEDEC, creates the "Balanced Quadrant" batches.

src/scribe_agent.py: The generation logic (injecting errors vs. noise vs. augmentation).

src/reward_engine.py: The deterministic scoring logic.

src/train_serl.py: The main training loop using OpenRLHF.

Key Libraries

transformers, peft (LoRA)

openrlhf (Primary RL Library)

Note: SeRL uses Reinforce++, often supported in OpenRLHF via custom configuration or PPO with specific parameters.

deepspeed (for efficient training)

ray (OpenRLHF dependency for distributed rollout)