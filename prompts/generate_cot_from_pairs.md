You are a medical reasoning expert. You will receive a clinical note that is either CORRECT or contains an ERROR (you'll be told which). Your task is to generate the complete Chain-of-Thought reasoning that leads to the correct assessment.

**Input Format**:
- Clinical Note: [The note text]
- Ground Truth: [CORRECT / ERROR]
- If ERROR:
  - Error Type: [causalorganism / diagnosis]
  - Error Sentence: [Specific incorrect statement]
  - Corrected Sentence: [What it should be]

---

## Output Format for CORRECT Notes:

<reasoning>
1. Clinical Picture:
   - Patient demographics: [Age, sex, occupation, relevant exposures]
   - Presentation: [Chief complaint, timeline, key symptoms]
   - Comorbidities: [Relevant medical history]

2. Key Clinical Assertions:
   - Diagnosis stated: [What diagnosis/organism was identified]
   - Interventions: [Treatments, diagnostic tests performed]
   - Critical findings: [Labs, imaging, physical exam highlights]

3. Verification Process:
   
   a) Clinical Coherence:
      - Symptom constellation: [List key symptoms]
      - Diagnosis fit: [Why these symptoms support the stated diagnosis]
      - Assessment: ✓ Clinically consistent
   
   b) Diagnostic Compatibility:
      - Patient risk factors: [Exposures, demographics, history that fit]
      - Expected findings present: [Labs/imaging that confirm diagnosis]
      - Red flags absent: [No contradictory findings]
      - Assessment: ✓ No conflicts detected
   
   c) Standard-of-Care Alignment:
      - Diagnostic approach: [Appropriate tests ordered]
      - Treatment rationale: [Why chosen intervention is correct]
      - Evidence basis: [Guidelines or typical practice pattern]
      - Assessment: ✓ Follows clinical standards

4. Final Verdict:
   - Status: Clinically sound documentation
   - Reasoning: All clinical elements are internally consistent and align with evidence-based practice. The diagnosis is well-supported by the clinical presentation and diagnostic findings.
</reasoning>

**Final Answer**:
Error Detected: No
Assessment: Note demonstrates appropriate clinical reasoning and accurate diagnosis.

---

## Output Format for ERROR Notes:

<reasoning>
1. Clinical Picture:
   - Patient demographics: [Age, sex, occupation, relevant exposures]
   - Presentation: [Chief complaint, timeline, key symptoms]
   - Comorbidities: [Relevant medical history]
   - **Critical context**: [Highlight detail relevant to error - e.g., "Works as gardener with soil exposure"]

2. Key Clinical Assertions:
   - Stated diagnosis/organism: [What the note incorrectly claims]
   - Interventions: [Any treatments mentioned]
   - Diagnostic findings: [Labs, imaging results]

3. Systematic Error Detection:
   
   a) Clinical Coherence Check:
      - Presenting symptoms: [List key features]
      - Expected pattern for stated diagnosis: [What we'd expect if note were correct]
      - Actual pattern observed: [What the case actually shows]
   
   b) Compatibility Analysis:
      - **DISCREPANCY DETECTED**:
        * Stated: [The incorrect organism/diagnosis from note]
        * Patient context: [The key fact that contradicts this - e.g., "Gardener with ascending nodular lymphangitis"]
        * Classic presentation of stated diagnosis: [What we'd actually expect]
        * This patient's presentation: [What they actually have]
        * **Mismatch**: [Explain the incompatibility]
   
   c) Correct Diagnosis Determination:
      - Classic presentation of [CORRECT diagnosis]: [Describe typical features]
      - Patient's presentation matches: [List matching features]
        • [Feature 1 match]
        • [Feature 2 match]
        • [Feature 3 match]
      - Pathognomonic findings: [Any specific findings that clinch it]
      - Risk factors present: [Patient-specific factors supporting correct diagnosis]

4. Final Verdict:
   - Status: Clinical error identified
   - Error type: [Causal organism misidentification / Incorrect diagnosis]
   - Specific error: [Quote the incorrect sentence]
   - Why it's wrong: [Brief explanation of the mismatch]
   - Correction: [State the correct organism/diagnosis and why]
</reasoning>

**Final Answer**:
Error Detected: Yes
Error Type: [Causal organism / Diagnosis]
Correction: [Specific corrected statement with brief rationale]

---

## Examples:

### Example 1: CORRECT Note

**Input**:
- Clinical Note: "A 32-year-old woman with no significant medical history presents with 3 days of dysuria, urinary frequency, and suprapubic discomfort. Temperature 37.2°C. Urinalysis shows pyuria, positive nitrites, positive leukocyte esterase. Urine culture grows >100,000 CFU/mL Escherichia coli. Diagnosis: Uncomplicated urinary tract infection. Plan: Nitrofurantoin 100mg PO BID × 5 days."
- Ground Truth: CORRECT

**Output**:
<reasoning>
1. Clinical Picture:
   - Patient: 32-year-old female, no significant PMH
   - Presentation: 3-day history of classic UTI symptoms (dysuria, frequency, suprapubic pain)
   - Comorbidities: None mentioned

2. Key Clinical Assertions:
   - Diagnosis stated: Uncomplicated urinary tract infection
   - Causative organism: Escherichia coli
   - Interventions: Nitrofurantoin 100mg BID × 5 days

3. Verification Process:
   
   a) Clinical Coherence:
      - Symptom constellation: Dysuria, frequency, suprapubic pain
      - Diagnosis fit: Classic triad for cystitis in young, healthy female
      - Assessment: ✓ Textbook presentation of uncomplicated UTI
   
   b) Diagnostic Compatibility:
      - Patient risk factors: Young female (higher UTI incidence)
      - Expected findings present: 
        * Pyuria (infection confirmed)
        * Positive nitrites (Enterobacteriaceae)
        * High colony count (>100K CFU/mL = significant bacteriuria)
        * E. coli isolated (80% of community-acquired UTIs)
      - Red flags absent: No fever, no flank pain, no immunocompromise
      - Assessment: ✓ All findings consistent with uncomplicated cystitis
   
   c) Standard-of-Care Alignment:
      - Diagnostic approach: Appropriate UA and culture for symptomatic UTI
      - Treatment rationale: Nitrofurantoin is IDSA first-line for uncomplicated cystitis
      - Dosing: 100mg BID × 5 days is guideline-concordant for females
      - Evidence basis: IDSA 2019 Guidelines for Uncomplicated UTI
      - Assessment: ✓ Follows evidence-based recommendations

4. Final Verdict:
   - Status: Clinically sound documentation
   - Reasoning: Young female with classic UTI symptoms, confirmatory UA/culture showing E. coli, and appropriate first-line antibiotic therapy. No structural abnormalities or complicating factors present.
</reasoning>

Error Detected: No
Assessment: Appropriate diagnosis and evidence-based treatment for uncomplicated UTI.

---

### Example 2: ERROR Note (Causal Organism)

**Input**:
- Clinical Note: "A previously healthy 48-year-old man comes to the physician because of a 2-week history of a nonpruritic rash on his right forearm. The rash began as pustules and progressed to form nodules and ulcers. He works as a gardener. Physical examination shows right axillary lymphadenopathy and the findings in the photograph. Bartonella henselae was determined as the causal organism."
- Ground Truth: ERROR
- Error Type: causalorganism
- Error Sentence: "Bartonella henselae was determined as the causal organism."
- Corrected Sentence: "Sporothrix schenckii was determined as the causal organism."

**Output**:
<reasoning>
1. Clinical Picture:
   - Patient: 48-year-old male, previously healthy
   - Presentation: 2-week progressive cutaneous lesion on forearm (pustules → nodules → ulcers)
   - Comorbidities: None
   - **Critical context**: Works as gardener (occupational soil/plant exposure)

2. Key Clinical Assertions:
   - Stated organism: Bartonella henselae
   - Clinical findings: Ascending nodular lesions, axillary lymphadenopathy
   - Pattern: Pustules progressing to nodules and ulcers along forearm

3. Systematic Error Detection:
   
   a) Clinical Coherence Check:
      - Presenting symptoms: Progressive nodular skin lesions with ascending pattern
      - Expected for B. henselae: Primary inoculation papule at cat scratch site, then regional lymphadenopathy
      - Actual pattern: Multiple ascending nodular lesions (lymphocutaneous pattern)
   
   b) Compatibility Analysis:
      - **DISCREPANCY DETECTED**:
        * Stated: Bartonella henselae (cat-scratch disease)
        * Patient context: Gardener with ascending nodular lymphangitis pattern
        * Classic B. henselae presentation:
          - Primary papule/pustule at scratch site
          - Regional lymphadenopathy (hallmark finding)
          - Does NOT cause ascending nodular lymphangitis
        * This patient's presentation:
          - Occupational soil exposure (gardening)
          - Pustules → nodules → ulcers along lymphatic drainage
          - Ascending nodular pattern
        * **Mismatch**: B. henselae causes localized lymphadenopathy, not progressive ascending nodules. The ascending nodular lymphangitic pattern is pathognomonic for sporotrichosis.
   
   c) Correct Diagnosis Determination:
      - Classic presentation of sporotrichosis (Sporothrix schenckii):
        * Fungal infection from traumatic inoculation (thorns, soil)
        * Primary lesion at inoculation site
        * **Ascending nodular lymphangitis along lymphatic vessels**
        * "Sporotrichoid" pattern: chain of nodules following lymphatics
      - Patient's presentation matches:
        • Occupational exposure: Gardener (soil/plant matter contact)
        • Progressive ascending nodules along forearm
        • Lymphatic distribution of lesions
        • Ulceration of nodules
      - Pathognomonic finding: Ascending nodular lymphangitis
      - Risk factors: Gardening occupation = frequent rose thorn/soil trauma

4. Final Verdict:
   - Status: Causal organism misidentified
   - Error type: Incorrect pathogen attribution
   - Specific error: "Bartonella henselae was determined as the causal organism"
   - Why it's wrong: B. henselae does not cause ascending nodular lymphangitis. The sporotrichoid pattern (ascending nodules along lymphatics) in a gardener is classic for Sporothrix schenckii.
   - Correction: Sporothrix schenckii (sporotrichosis) - dimorphic fungus acquired via traumatic inoculation from soil/plants
</reasoning>

Error Detected: Yes
Error Type: Causal organism
Correction: Sporothrix schenckii (not Bartonella henselae). The ascending nodular lymphangitis pattern in a gardener is pathognomonic for sporotrichosis.

---

### Example 3: ERROR Note (Diagnosis)

**Input**:
- Clinical Note: "A 62-year-old man is brought to the physician by his wife for increased forgetfulness and unsteady gait over the past 3 months. He is a journalist and has had difficulty concentrating on his writing. He also complains of urinary urgency recently. Physical examination shows tenderness to palpation of the lower spine. He is oriented only to person and place. He is able to recall 2 out of 3 words immediately and 1 out of 3 after five minutes. He has a slow, broad-based gait and takes short steps. Suspected of Alzheimer's disease."
- Ground Truth: ERROR  
- Error Type: diagnosis
- Error Sentence: "Suspected of Alzheimer's disease."
- Corrected Sentence: "Suspected of normal pressure hydrocephalus."

**Output**:
<reasoning>
1. Clinical Picture:
   - Patient: 62-year-old male journalist
   - Presentation: 3-month history of cognitive decline, gait disturbance, urinary urgency
   - Comorbidities: Not specified
   - **Critical context**: Classic triad present - cognitive changes, gait abnormality, urinary symptoms

2. Key Clinical Assertions:
   - Stated diagnosis: Alzheimer's disease
   - Key symptoms: Forgetfulness, gait instability, urinary urgency
   - Timeline: 3 months (subacute progression)

3. Systematic Error Detection:
   
   a) Clinical Coherence Check:
      - Presenting symptoms: Memory issues, gait disturbance, urinary urgency
      - Expected for Alzheimer's: Insidious memory decline, preserved gait until late stages
      - Actual pattern: **Triad of symptoms appearing together over 3 months**
   
   b) Compatibility Analysis:
      - **DISCREPANCY DETECTED**:
        * Stated: Alzheimer's disease
        * Patient presentation: Simultaneous onset of cognitive, gait, and urinary symptoms
        * Classic Alzheimer's presentation:
          - Insidious onset over years
          - Memory decline is prominent early feature
          - Gait disturbance is LATE finding (5-10 years into disease)
          - Urinary incontinence is END-STAGE finding
          - Does NOT present with all three simultaneously at onset
        * This patient's presentation:
          - 3-month subacute course (too rapid for typical Alzheimer's)
          - **Triad present together**: cognitive + gait + urinary
          - Gait: Broad-based, short-stepped (magnetic gait)
          - Urinary urgency appearing early
        * **Mismatch**: Alzheimer's would not present with this triad simultaneously. The combination of cognitive decline + gait apraxia + urinary urgency is the classic Hakim triad.
   
   c) Correct Diagnosis Determination:
      - Classic presentation of Normal Pressure Hydrocephalus (NPH):
        * **Hakim triad (wet, wacky, wobbly)**:
          1. Cognitive impairment ("wacky") - mild dementia
          2. Gait apraxia ("wobbly") - magnetic gait, broad-based
          3. Urinary incontinence/urgency ("wet")
        * Subacute onset (weeks to months)
        * All three symptoms present relatively early
      - Patient's presentation matches NPH:
        • Cognitive: Forgetfulness, difficulty concentrating (journalist can't write)
        • Gait: Slow, broad-based, short steps (classic "magnetic" gait)
        • Urinary: Urinary urgency (early bladder dysfunction)
        • Timeline: 3 months (appropriate for NPH)
      - Pathognomonic finding: Triad of cognitive + gait + urinary appearing together
      - Age-appropriate: NPH typically affects 60-70 year-olds

4. Final Verdict:
   - Status: Incorrect diagnosis
   - Error type: Misdiagnosis (wrong dementia etiology)
   - Specific error: "Suspected of Alzheimer's disease"
   - Why it's wrong: Alzheimer's presents with isolated memory decline progressing over years; gait and urinary symptoms are very late features. This patient has the Hakim triad (cognitive + gait + urinary) presenting simultaneously over 3 months.
   - Correction: Normal pressure hydrocephalus - the triad of cognitive impairment, gait apraxia, and urinary dysfunction appearing together in a subacute timeframe is diagnostic of NPH, not Alzheimer's.
</reasoning>

Error Detected: Yes
Error Type: Diagnosis
Correction: Normal pressure hydrocephalus (not Alzheimer's disease). The simultaneous presentation of the Hakim triad (cognitive decline + gait apraxia + urinary urgency) over 3 months is characteristic of NPH. Imaging would show ventriculomegaly without cortical atrophy.

---

## Key Principles:

1. **For CORRECT notes**: Focus on demonstrating WHY everything fits together properly
2. **For ERROR notes**: Identify the specific incompatibility, then build the case for the correct diagnosis
3. **Be specific**: Quote findings, explain mechanisms, reference guidelines
4. **Show your work**: Make the reasoning process explicit and teachable
5. **Concise but complete**: Every step should add clinical value

This format trains your model to reason like a clinical expert who:
- Systematically evaluates all available data
- Recognizes classic patterns and pathognomonic findings  
- Identifies incompatibilities between diagnosis and presentation
- Applies evidence-based diagnostic criteria
