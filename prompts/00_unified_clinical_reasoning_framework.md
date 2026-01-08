# Universal Clinical Reasoning Framework

This framework is used by both **Generator** (creating notes) and **Critic** (assessing notes) roles. The same 4-step structure applies to all clinical reasoning tasks.

## The 4-Step Framework

### Step 1: Clinical Picture Analysis
**Purpose**: Establish the clinical context

**Generator perspective**: "What are the key facts I need to address in this note?"
- Patient demographics and risk factors
- Chief complaint and presenting symptoms  
- Relevant medical history and comorbidities
- Exposures, occupations, or contextual factors

**Critic perspective**: "What are the key facts I need to verify?"
- Same elements, but looking for consistency and completeness

---

### Step 2: Key Clinical Assertions
**Purpose**: Identify the core medical claims

**Generator perspective**: "What claims am I making in this note?"
- Primary diagnosis or differential
- Identified causative organisms (if applicable)
- Treatment plan and interventions
- Expected outcomes or follow-up

**Critic perspective**: "What claims is this note making that I need to validate?"
- Extract and list all medical assertions
- Identify which assertions are most critical to verify

---

### Step 3: Systematic Verification
**Purpose**: Ensure clinical soundness through structured checks

**Both perspectives use the same 3 sub-checks**:

#### a) Clinical Coherence
- Do symptoms support the stated diagnosis?
- Are all presenting features explained?
- Is the timeline consistent?

#### b) Safety Compatibility  
- Patient contraindications: Any conditions that preclude this treatment?
- Drug-drug interactions: Polypharmacy concerns?
- Drug-disease interactions: Comorbidity conflicts?
- Dosing appropriateness: Age/weight/renal/hepatic adjustments needed?

#### c) Guideline Alignment
- Is this the standard-of-care approach?
- Are diagnostic steps appropriate?
- Is treatment choice evidence-based?
- Are doses and durations guideline-concordant?

**Generator uses this to BUILD a safe note**  
**Critic uses this to EVALUATE an existing note**

---

### Step 4: Final Verdict/Plan
**Purpose**: Synthesize reasoning into actionable conclusion

**Generator perspective**: "This note is sound because..."
- Summarize why the management is appropriate
- Confirm all safety checks passed
- State confidence in the plan

**Critic perspective**: "This note is [correct/incorrect] because..."
- State overall assessment
- If error: specify what's wrong and why
- If correct: validate the clinical reasoning

---

## Example: Same Framework, Different Roles

### Scenario: 58F with asthma presenting with new A-fib

#### Generator Mode (Creating CORRECT note):
