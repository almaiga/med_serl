You are a medical educator teaching a model HOW correct clinical notes are created. Given an existing CORRECT clinical note, you will reverse-engineer the reasoning that would have produced it.

**Task**: Demonstrate the thought process that led to creating this clinically sound note.

**Input Format**:
- Clinical Note: [A correct note from your dataset]

**Output Format**:
<generation_reasoning>
1. Clinical Picture Analysis:
   - Patient demographics: [Age, sex, relevant history]
   - Presenting complaint: [Chief complaint, timeline]
   - Key findings: [Physical exam, labs, imaging highlights]
   - Clinical context: [Risk factors, exposures, comorbidities]
   - **Synthesis**: [1-2 sentence summary of the clinical situation]

2. Key Clinical Assertions to Establish:
   - Primary diagnosis needed: [What condition explains this presentation]
   - Causative organism (if applicable): [Which pathogen fits the pattern]
   - Treatment plan required: [What intervention is needed]
   - **Decision point**: [Why these assertions are appropriate]

3. Systematic Safety Verification (Pre-writing checklist):
   
   a) Clinical Coherence Check:
      - Symptom pattern: [List key features]
      - Diagnosis fit: [Why this diagnosis matches]
      - Alternative diagnoses considered: [What else could this be, and why ruled out]
      - ✓ Decision: [Chosen diagnosis is most consistent]
   
   b) Safety Compatibility Screen:
      - Patient contraindications reviewed: [Any conditions that affect treatment choice]
        * [Condition 1]: [How it affects management - e.g., "Asthma → avoid non-selective β-blockers"]
        * [Condition 2]: [Impact on drug choice]
      - Drug-drug interactions: [If multiple meds, check interactions]
      - Drug-disease interactions: [Verify no conflicts]
      - Dosing adjustments needed: [Age/weight/renal/hepatic considerations]
      - ✓ Decision: [Chosen treatment is safe for this patient]
   
   c) Guideline Alignment Verification:
      - Standard approach: [What do guidelines recommend]
      - Treatment choice justification: [Why this specific intervention]
      - Dosing rationale: [Standard vs adjusted, and why]
      - Monitoring plan: [What follow-up is needed]
      - ✓ Decision: [Plan follows evidence-based practice]

4. Note Construction Plan:
   - Structure: [How to organize the note - e.g., SOAP format]
   - Key elements to include:
     * Patient identifiers and context
     * Clinical findings supporting diagnosis
     * Diagnostic reasoning (briefly)
     * Clear management plan
     * Follow-up instructions
   - Language: [Professional, precise, complete]

5. Final Note Composition:
   [This is where the actual clinical note appears - the one from your dataset]

6. Post-Writing Verification:
   - Completeness check: All critical information included ✓
   - Safety confirmation: No contradictions or risks ✓
   - Clarity assessment: Note is unambiguous ✓
   - Confidence: This note represents sound clinical practice ✓
</generation_reasoning>

---

**Example**:

**Input**:
- Clinical Note: "A 32-year-old woman with no significant medical history presents with 3 days of dysuria, urinary frequency, and suprapubic discomfort. Temperature 37.2°C. Urinalysis shows pyuria, positive nitrites, positive leukocyte esterase. Urine culture grows >100,000 CFU/mL Escherichia coli. Diagnosis: Uncomplicated urinary tract infection. Plan: Nitrofurantoin 100mg PO BID × 5 days."

**Output**:
<generation_reasoning>
1. Clinical Picture Analysis:
   - Patient demographics: 32-year-old female, no significant PMH
   - Presenting complaint: 3-day history of urinary symptoms (dysuria, frequency, suprapubic pain)
   - Key findings: Afebrile (37.2°C), UA positive for infection markers, E. coli on culture
   - Clinical context: Young, healthy female with isolated urinary symptoms - low risk for complications
   - **Synthesis**: Classic presentation of uncomplicated lower urinary tract infection in a healthy young woman

2. Key Clinical Assertions to Establish:
   - Primary diagnosis needed: Uncomplicated cystitis (UTI limited to bladder)
   - Causative organism: E. coli (confirmed by culture - most common uropathogen)
   - Treatment plan required: First-line oral antibiotic for uncomplicated UTI
   - **Decision point**: This is a straightforward case requiring standard first-line therapy per guidelines

3. Systematic Safety Verification (Pre-writing checklist):
   
   a) Clinical Coherence Check:
      - Symptom pattern: Dysuria, frequency, suprapubic pain (classic UTI triad)
      - Diagnosis fit: Symptoms + positive UA + E. coli culture = definitive UTI diagnosis
      - Alternative diagnoses considered:
        * Pyelonephritis: Ruled out (afebrile, no flank pain, no systemic symptoms)
        * STI: Less likely (no discharge, no partner symptoms mentioned)
        * Interstitial cystitis: Acute onset favors infection over chronic condition
      - ✓ Decision: Uncomplicated cystitis is the correct diagnosis
   
   b) Safety Compatibility Screen:
      - Patient contraindications reviewed:
        * No pregnancy mentioned (but should confirm - nitrofurantoin avoid in 3rd trimester)
        * No renal impairment (nitrofurantoin requires adequate renal function)
        * No G6PD deficiency mentioned (nitrofurantoin can cause hemolysis)
        * No liver disease (generally safe)
      - Drug-drug interactions: None (no current medications listed)
      - Drug-disease interactions: None identified
      - Dosing adjustments needed: None (normal adult dosing appropriate)
      - ✓ Decision: Nitrofurantoin is safe choice - no contraindications present
   
   c) Guideline Alignment Verification:
      - Standard approach: IDSA 2019 Guidelines recommend nitrofurantoin as first-line for uncomplicated cystitis
      - Treatment choice justification: 
        * High efficacy against E. coli
        * Low resistance rates
        * Minimal collateral damage to microbiome
        * Cost-effective
      - Dosing rationale: 100mg BID is standard macrocrystal formulation dose
      - Duration: 5 days is guideline-recommended for women with uncomplicated UTI
      - Monitoring plan: Symptom improvement expected in 24-48h; follow-up if no improvement
      - ✓ Decision: Treatment plan is fully guideline-concordant

4. Note Construction Plan:
   - Structure: Use clear narrative format with diagnosis and plan sections
   - Key elements to include:
     * Patient age, sex, medical history (none significant)
     * Duration and nature of symptoms
     * Vital signs (confirm not septic)
     * UA results (support diagnosis)
     * Culture results (confirm pathogen)
     * Clear diagnosis statement
     * Specific treatment with dose, route, frequency, duration
   - Language: Concise, professional, complete

5. Final Note Composition:
   "A 32-year-old woman with no significant medical history presents with 3 days of dysuria, urinary frequency, and suprapubic discomfort. Temperature 37.2°C. Urinalysis shows pyuria, positive nitrites, positive leukocyte esterase. Urine culture grows >100,000 CFU/mL Escherichia coli. Diagnosis: Uncomplicated urinary tract infection. Plan: Nitrofurantoin 100mg PO BID × 5 days."

6. Post-Writing Verification:
   - Completeness check: All key information included (history, exam, labs, diagnosis, plan) ✓
   - Safety confirmation: No contraindications, appropriate drug choice and dosing ✓
   - Clarity assessment: Diagnosis and plan are explicit and actionable ✓
   - Confidence: This note represents standard-of-care management for uncomplicated UTI ✓
</generation_reasoning>

---

**Training Goal**: 
By seeing HOW correct notes are constructed (with explicit safety reasoning), the model learns to:
1. Recognize what makes a note clinically sound
2. Apply systematic safety checks before committing to a plan
3. Generate similar correct notes when given new scenarios
4. Understand the relationship between clinical reasoning and written documentation
