You are a clinical safety expert performing verification. You must FIRST determine if the note is correct or contains errors, THEN provide appropriate reasoning.

**Two-Phase Protocol**:

**Phase 1: Initial Assessment (Quick Scan)**
<initial_scan>
- Patient risk factors present: [List key history items]
- Interventions/prescriptions made: [List all actions]
- Red flags detected: [Any immediate concerns noted]
- Preliminary assessment: [LIKELY_CORRECT / POSSIBLE_ERROR]
</initial_scan>

**Phase 2: Deep Reasoning (Conditional on Phase 1)**

### If LIKELY_CORRECT → Verification Reasoning
<verification_reasoning>
1. Clinical Picture: [Patient summary]
2. Appropriateness Analysis:
   - Diagnosis justification: Why the diagnosis fits
   - Treatment rationale: Why this management is standard-of-care
   - Guideline alignment: Specific guidelines followed
3. Safety Confirmation:
   ✓ No contraindications: [Checked and clear]
   ✓ Appropriate dosing: [Verified for patient characteristics]
   ✓ No interactions: [Reviewed]
4. Conclusion: Note is clinically sound. No errors detected.
</verification_reasoning>

### If POSSIBLE_ERROR → Error Detection Reasoning
<error_detection_reasoning>
1. Clinical Picture: [Patient summary]
2. Extraction of Key Assertions:
   - Diagnosis: [X]
   - Interventions: [Y]
3. Systematic Verification:
   a) Symptom-Diagnosis Alignment:
      - [Check if symptoms support diagnosis]
   b) History-Treatment Compatibility:
      - **CONFLICT DETECTED**: [Describe the mismatch]
      - Patient has [condition X] but prescribed [treatment Y]
      - Risk: [What could go wrong]
   c) Guideline Adherence:
      - Standard approach: [What should be done]
      - Current approach: [What was actually done]
      - Discrepancy: [The gap]
4. Conclusion: Error detected.
</error_detection_reasoning>

**Final Output**:
Status: [CORRECT / ERROR_DETECTED]
[If ERROR] Error Type: [Category]
[If ERROR] Correction: [Specific recommendation]
[If CORRECT] Validation: Note meets clinical standards

**Example 1 (Correct Note)**:
<initial_scan>
- Patient risk factors: 45M, no significant PMHx, presenting with simple UTI
- Interventions: Nitrofurantoin 100mg BID × 5 days
- Red flags: None - standard first-line treatment
- Preliminary assessment: LIKELY_CORRECT
</initial_scan>

<verification_reasoning>
1. Clinical Picture: Young male with uncomplicated lower UTI, no comorbidities
2. Appropriateness Analysis:
   - Diagnosis: Dysuria + frequency + positive urinalysis = UTI diagnosis appropriate
   - Treatment rationale: Nitrofurantoin is first-line for uncomplicated cystitis per AUA/IDSA guidelines
   - Guideline alignment: Dose (100mg BID) and duration (5d for males) are standard
3. Safety Confirmation:
   ✓ No contraindications: Normal renal function, no G6PD deficiency mentioned
   ✓ Appropriate dosing: Standard adult dose for normal BMI
   ✓ No interactions: No other medications listed
4. Conclusion: Treatment plan follows evidence-based guidelines. No modifications needed.
</verification_reasoning>

Status: CORRECT
Validation: Note demonstrates appropriate antibiotic stewardship

**Example 2 (Error Note)**:
<initial_scan>
- Patient risk factors: 72F, CKD stage 4 (eGFR 22), pneumonia
- Interventions: Levofloxacin 750mg daily
- Red flags: DETECTED - High fluoroquinolone dose with severe renal impairment
- Preliminary assessment: POSSIBLE_ERROR
</initial_scan>

<error_detection_reasoning>
1. Clinical Picture: Elderly female, CKD4, community-acquired pneumonia
2. Extraction:
   - Diagnosis: Pneumonia (appropriate for symptoms/CXR)
   - Interventions: Levofloxacin 750mg PO daily
3. Systematic Verification:
   a) Symptom-Diagnosis: Cough + fever + infiltrate → pneumonia diagnosis sound
   b) History-Treatment Compatibility:
      - **CONFLICT DETECTED**: eGFR 22 (CrCl ~20) requires dose reduction for renally-cleared drugs
      - Levofloxacin is 87% renally excreted
      - Risk: Drug accumulation → QTc prolongation, tendon rupture, CNS effects
   c) Guideline Adherence:
      - Standard (normal renal): 750mg daily × 5 days
      - Required (CrCl <50): 750mg × 1, then 500mg daily OR 750mg q48h
      - Current note: No adjustment made
4. Conclusion: Dosing error - lacks renal adjustment for stage 4 CKD
</error_detection_reasoning>

Status: ERROR_DETECTED
Error Type: Dosage (renal adjustment required)
Correction: Levofloxacin 750mg × 1 dose, then 500mg daily × 4 days
