You are a medical educator teaching a model HOW realistic medical errors occur. Given an existing note pair (CORRECT and INCORRECT versions), you will reverse-engineer the adversarial reasoning that produced the error.

**Task**: Demonstrate the thought process that led to injecting a plausible but dangerous error into a correct note.

**Input Format**:
- Correct Note: [The original clinically sound note]
- Incorrect Note: [The version with an injected error]
- Error Metadata: [error_type, error_sentence, corrected_sentence]

**Output Format**:
<error_injection_reasoning>
1. Analysis of Correct Note:
   - Clinical situation: [Brief summary of case]
   - Current correct management: [What's being done right]
   - Key patient factors: [Demographics, comorbidities, exposures that matter]
   - **Vulnerability points identified**: [Where errors commonly happen in similar cases]

2. Error Strategy Selection:
   - Target error type: [Contraindication/Dosage/Wrong organism/Wrong diagnosis]
   - Why this error is "sneaky": [What cognitive trap makes this error plausible]
     * Cognitive mechanism: [e.g., "Availability heuristic - provider thinks of common drug first"]
     * Plausibility factor: [Why it looks reasonable on surface]
     * Detection difficulty: [What makes it hard to catch]
   - Expected difficulty level: [Easy/Medium/Hard for critic to detect]
   - Clinical risk level: [Minor/Moderate/Severe if undetected]
   - **Strategic choice**: [Why this particular error teaches the model something valuable]

3. Error Injection Mechanism:
   - Original correct statement: "[Quote exact text]"
   - Adversarial modification: "[Quote exact modification]"
   - Preservation of plausibility: [What was kept the same to maintain realism]
   - Critical clue planted: [Subtle detail that reveals the error - e.g., "Asthma mentioned in line 1"]
   - **Trap design**: [How this error tests clinical reasoning]

4. Adversarial Reasoning (Why a clinician might make this mistake):
   - Cognitive shortcuts: [Mental heuristics that lead to this error]
   - Time pressure scenario: [How this error happens in busy clinical settings]
   - Knowledge gap: [What specialized knowledge is needed to catch this]
   - Pattern mismatch: [How it violates expected clinical pattern]
   - **Teaching value**: [What medical knowledge this error tests]

5. Modified Note with Error:
   [The full incorrect note from your dataset]

6. Ground Truth Documentation:
   - Error location: "[Exact sentence containing error]"
   - Error category: [Specific classification]
   - Why it's wrong: [Medical explanation of the problem]
   - Clinical consequence if uncaught: [Harm that would result]
   - Correct alternative: "[What should have been written]"
   - Key for detection: [The specific detail critic should notice]
</error_injection_reasoning>

---

**Example**:

**Input**:
- Correct Note: "58F with moderate persistent asthma (on fluticasone/salmeterol) and hypertension presents to ED with palpitations × 2 hours. ECG confirms atrial fibrillation with RVR (HR 142). Hemodynamically stable. No chest pain. Plan: Metoprolol 25mg PO BID for rate control."
- Incorrect Note: "58F with moderate persistent asthma (on fluticasone/salmeterol) and hypertension presents to ED with palpitations × 2 hours. ECG confirms atrial fibrillation with RVR (HR 142). Hemodynamically stable. No chest pain. Plan: Propranolol 40mg PO BID for rate control."
- Error Metadata: {error_type: "contraindication", error_sentence: "Plan: Propranolol 40mg PO BID for rate control.", corrected_sentence: "Plan: Metoprolol 25mg PO BID for rate control."}

**Output**:
<error_injection_reasoning>
1. Analysis of Correct Note:
   - Clinical situation: Middle-aged woman with new-onset A-fib needing rate control
   - Current correct management: Metoprolol (cardioselective β-blocker) chosen
   - Key patient factors: 
     * **Asthma** (moderate persistent, on controller meds)
     * Hypertension (makes β-blocker reasonable)
     * Hemodynamically stable (allows oral agent)
   - **Vulnerability points identified**: 
     * β-blockers are common first-line for A-fib → easy to default to non-selective
     * Asthma mentioned early but may be overlooked when focus is on cardiac issue
     * Multiple conditions require integration of conflicting constraints

2. Error Strategy Selection:
   - Target error type: Drug-disease contraindication (respiratory)
   - Why this error is "sneaky":
     * Cognitive mechanism: Anchoring bias - provider focuses on "need β-blocker for A-fib" and grabs familiar drug (propranolol)
     * Plausibility factor: Propranolol is indeed used for A-fib and HTN - it's a valid drug, just wrong patient
     * Detection difficulty: Requires connecting TWO facts: (1) asthma in line 1, (2) drug choice in plan
   - Expected difficulty level: Medium (catchable but requires active integration of history)
   - Clinical risk level: Severe (can trigger acute bronchospasm → respiratory failure)
   - **Strategic choice**: Tests whether model integrates past medical history into treatment decisions (not just pattern matching diagnoses)

3. Error Injection Mechanism:
   - Original correct statement: "Plan: Metoprolol 25mg PO BID for rate control."
   - Adversarial modification: "Plan: Propranolol 40mg PO BID for rate control."
   - Preservation of plausibility:
     * Kept same indication (A-fib rate control) ✓
     * Used realistic dosing (40mg BID is standard) ✓
     * Maintained same route and frequency ✓
     * Did NOT add any obvious red flags (no mention of wheezing) ✓
   - Critical clue planted: Asthma documented in first sentence with specific medications (fluticasone/salmeterol)
   - **Trap design**: Asthma is mentioned but not highlighted - tests if model truly reads full clinical picture before evaluating plan

4. Adversarial Reasoning (Why a clinician might make this mistake):
   - Cognitive shortcuts:
     * "A-fib → need β-blocker" (correct)
     * "Propranolol = β-blocker" (correct)
     * "Therefore propranolol works" (WRONG - skipped selectivity check)
   - Time pressure scenario: Busy ED, multiple patients, provider focuses on acute cardiac issue and forgets respiratory history
   - Knowledge gap: Understanding difference between cardioselective (metoprolol, atenolol) vs non-selective (propranolol, carvedilol) β-blockers
   - Pattern mismatch: Non-selective β₂-blockade → bronchoconstriction in asthma patients
   - **Teaching value**: Tests understanding of β-blocker pharmacology + ability to integrate multiple organ systems

5. Modified Note with Error:
   "58F with moderate persistent asthma (on fluticasone/salmeterol) and hypertension presents to ED with palpitations × 2 hours. ECG confirms atrial fibrillation with RVR (HR 142). Hemodynamically stable. No chest pain. Plan: Propranolol 40mg PO BID for rate control."

6. Ground Truth Documentation:
   - Error location: "Plan: Propranolol 40mg PO BID for rate control."
   - Error category: Drug-disease contraindication (respiratory system)
   - Why it's wrong: Propranolol is non-selective β-blocker → blocks β₂-receptors in lungs → bronchoconstriction in asthma patients → can precipitate respiratory failure
   - Clinical consequence if uncaught: Acute asthma exacerbation, potential respiratory failure requiring intubation
   - Correct alternative: "Plan: Metoprolol 25mg PO BID for rate control" (cardioselective, safer in asthma)
   - Key for detection: Must connect asthma in line 1 with drug choice in plan - tests integration of full clinical picture
</error_injection_reasoning>

---

**Training Goal**:
By seeing HOW errors are strategically injected (with explicit adversarial reasoning), the model learns to:
1. Recognize common cognitive traps that lead to medical errors
2. Understand what makes errors "plausible but dangerous"
3. Generate similar challenging errors for self-play training
4. Appreciate the subtlety required for realistic error injection (not obvious mistakes)
5. Create errors that test specific medical knowledge domains
