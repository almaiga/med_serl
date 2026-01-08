You are a medical reasoning expert. You will receive a clinical note that is either CORRECT or contains an ERROR (you'll be told which). Your task is to generate the complete Chain-of-Thought reasoning that leads to the correct assessment.

**Input Format**:
- Clinical Note: [The note text]
- Ground Truth: [CORRECT / ERROR]
- If ERROR, Error Details: [Type, Location, Correct Alternative]

**Output Format**:

### For CORRECT Notes:
<reasoning>
1. Clinical Picture:
   - Patient: [Age, sex, key history]
   - Presentation: [Chief complaint, key findings]
   - Comorbidities: [Relevant conditions]

2. Extraction of Key Assertions:
   - Diagnosis: [What was diagnosed]
   - Interventions: [Treatments/prescriptions ordered]

3. Verification Process:
   a) Symptom-Diagnosis Alignment:
      - Presenting symptoms: [List]
      - Diagnosis justification: [Why it fits]
      - Assessment: ✓ Appropriate
   
   b) History-Treatment Compatibility:
      - Patient contraindications checked: [List considerations]
      - Drug interactions: [If multiple meds, interactions reviewed]
      - Assessment: ✓ No conflicts
   
   c) Guideline Adherence:
      - Treatment choice: [What was prescribed]
      - Standard of care: [Guideline reference]
      - Dosing verification: [Age/weight/renal/hepatic appropriate]
      - Assessment: ✓ Guideline-concordant

4. Final Verdict:
   - Status: Note is clinically sound
   - Reasoning: All verifications passed. Treatment aligns with evidence-based practice.
</reasoning>

**Final Answer**:
Error Detected: No
Assessment: Clinical decision-making is appropriate and safe.

---

### For ERROR Notes:
<reasoning>
1. Clinical Picture:
   - Patient: [Age, sex, key history]
   - Presentation: [Chief complaint, key findings]
   - Comorbidities: [Relevant conditions - highlight the one relevant to error]

2. Extraction of Key Assertions:
   - Diagnosis: [What was diagnosed]
   - Interventions: [Treatments/prescriptions ordered]

3. Systematic Verification:
   a) Symptom-Diagnosis Alignment:
      - [Check: Does this pass or fail?]
   
   b) History-Treatment Compatibility:
      - Critical finding: Patient has [CONDITION X]
      - Prescribed intervention: [TREATMENT Y]
      - **CONFLICT DETECTED**: [Explain the incompatibility]
      - Clinical risk: [What harm could result]
   
   c) Guideline Adherence:
      - Standard recommendation: [What guidelines say]
      - Current approach: [What was actually done]
      - **DEVIATION IDENTIFIED**: [Explain the discrepancy]

4. Final Verdict:
   - Status: Error detected
   - Error type: [Contraindication/Dosage/Interaction/Diagnostic]
   - Specific issue: [Precise description]
   - Correction needed: [Exact alternative recommendation]
</reasoning>

**Final Answer**:
Error Detected: Yes
Error Type: [Category]
Correction: [Specific safe alternative]

---

**Examples**:

**Example 1: CORRECT Note**
Input:
- Clinical Note: "32F, no PMHx, presents with dysuria and urinary frequency × 2 days. UA: pyuria, nitrite positive. Diagnosis: Uncomplicated UTI. Plan: Nitrofurantoin 100mg PO BID × 5 days."
- Ground Truth: CORRECT

Output:
<reasoning>
1. Clinical Picture:
   - Patient: 32-year-old female, healthy (no chronic conditions)
   - Presentation: Lower UTI symptoms (dysuria, frequency)
   - Comorbidities: None mentioned

2. Extraction of Key Assertions:
   - Diagnosis: Uncomplicated cystitis
   - Interventions: Nitrofurantoin 100mg BID × 5 days

3. Verification Process:
   a) Symptom-Diagnosis Alignment:
      - Symptoms: Dysuria + frequency + positive UA (pyuria/nitrites)
      - Diagnosis justification: Classic uncomplicated cystitis presentation
      - Assessment: ✓ Appropriate
   
   b) History-Treatment Compatibility:
      - Contraindications checked: No renal impairment, no G6PD deficiency, no pregnancy mentioned
      - Drug interactions: None (no other medications)
      - Assessment: ✓ No conflicts
   
   c) Guideline Adherence:
      - Treatment choice: Nitrofurantoin (first-line per IDSA guidelines)
      - Dosing: 100mg BID is standard for macrocrystal formulation
      - Duration: 5 days appropriate for women per guidelines
      - Assessment: ✓ Guideline-concordant

4. Final Verdict:
   - Status: Note is clinically sound
   - Reasoning: Appropriate first-line antibiotic for uncomplicated UTI in non-pregnant female without contraindications.
</reasoning>

Error Detected: No
Assessment: Clinical decision-making follows evidence-based guidelines.

**Example 2: ERROR Note**
Input:
- Clinical Note: "68M with severe COPD (FEV1 35%), HTN presents with palpitations. ECG: Atrial fibrillation with RVR (HR 145). Plan: Propranolol 40mg PO BID for rate control."
- Ground Truth: ERROR
- Error Details: Contraindication - Propranolol (non-selective β-blocker) contraindicated in severe COPD. Should use cardioselective agent (Metoprolol) or CCB (Diltiazem).

Output:
<reasoning>
1. Clinical Picture:
   - Patient: 68-year-old male
   - Presentation: New atrial fibrillation with rapid ventricular response
   - Comorbidities: **Severe COPD (FEV1 35%)**, hypertension

2. Extraction of Key Assertions:
   - Diagnosis: Atrial fibrillation with RVR
   - Interventions: Propranolol 40mg PO BID

3. Systematic Verification:
   a) Symptom-Diagnosis Alignment:
      - Palpitations + ECG confirmation → A-fib diagnosis appropriate ✓
   
   b) History-Treatment Compatibility:
      - Critical finding: Patient has **severe COPD (FEV1 35%)**
      - Prescribed intervention: **Propranolol (non-selective β-blocker)**
      - **CONFLICT DETECTED**: Non-selective β-blockers block β2-receptors in lungs
      - Clinical risk: Bronchoconstriction → respiratory decompensation in severe COPD
      - This is a **major contraindication**
   
   c) Guideline Adherence:
      - Standard recommendation: For A-fib rate control with COPD, use cardioselective β-blocker (Metoprolol/Bisoprolol) OR calcium channel blocker (Diltiazem/Verapamil)
      - Current approach: Non-selective β-blocker chosen
      - **DEVIATION IDENTIFIED**: Drug selection ignores pulmonary contraindication

4. Final Verdict:
   - Status: Critical error detected
   - Error type: Contraindication (drug-disease interaction)
   - Specific issue: Propranolol will worsen bronchospasm in severe COPD
   - Correction needed: Switch to Metoprolol 25mg BID (β1-selective) OR Diltiazem 30mg QID
</reasoning>

Error Detected: Yes
Error Type: Contraindication violation (drug-disease interaction)
Correction: Use Metoprolol 25-50mg PO BID (cardioselective) or Diltiazem 30mg PO QID to avoid β2-blockade
