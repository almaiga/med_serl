#!/usr/bin/env python3
"""Generate a synthetic test dataset for the judge-logprob calibration pilot.

Emits a JSONL file matching the v6_run2 game log schema so it can be consumed
by test_judge_logprobs_pilot.py via --local-logs. 48 hand-crafted cases across
8 categories, balanced so we can measure both HIGH-confidence-when-right and
LOW-confidence-when-wrong for calibration analysis.

Categories:
  1a. clear_error         (8) — anaphylaxis, DKA, meningitis: obvious errors
  1b. clear_benign        (8) — brand/generic, abbrev, temporal-unit swaps
  2a. medium_error        (6) — requires clinical reasoning
  2b. medium_benign       (6) — jargon substitution, restructuring
  3a. subtle_error        (6) — descriptor mismatch, drug-class error
  3b. suspicious_benign   (4) — clean rewrite that could be over-flagged
  3c. injector_over_edit  (4) — real error introduced in "benign" mode
  4.  injector_failure    (6) — truncation + garbled clinical text

Usage:
    python3 scripts/self_play/generate_synthetic_test_dataset.py
    # writes data_processed/synthetic_test/game_synthetic_v1.jsonl
    # + data_processed/synthetic_test/game_synthetic_v1_manifest.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def make_case(
    note_id: str,
    mode: str,
    sentences: List[str],
    changed_sid: int,
    modified_sentence: str,
    failure_mode: str,
    expected_verdict: str,
    expected_sid,
    difficulty: str,
    rationale: str,
    error_type: str = "",
) -> Dict:
    """Build one game-log-schema-compatible test case."""
    correct_note = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    modified_list = list(sentences)
    modified_list[changed_sid - 1] = modified_sentence
    modified_note = "\n".join(f"{i+1}. {s}" for i, s in enumerate(modified_list))
    return {
        "phase": "game_complete",
        "note_id": note_id,
        "mode": mode,
        "error_type": error_type,
        "correct_sentences": correct_note,
        "modified_sentences": modified_note,
        "original_sentence": sentences[changed_sid - 1],
        "modified_sentence": modified_sentence,
        "changed_sid": changed_sid,
        # Test-specific metadata (not in the real game log schema):
        "failure_mode": failure_mode,
        "expected_verdict": expected_verdict,
        "expected_sid": expected_sid,
        "difficulty": difficulty,
        "rationale": rationale,
    }


# ══════════════════════════════════════════════════════════════════════════
# CATEGORY 1a — CLEAR REAL ERRORS (8 cases)
# Judge should CONFIDENTLY output the sid
# ══════════════════════════════════════════════════════════════════════════
CASES_1A = [
    make_case(
        note_id="synth-1a-01",
        mode="error_injection", error_type="pharmacotherapy",
        sentences=[
            "A 24-year-old woman is brought to the emergency department 5 minutes after eating peanuts at a restaurant.",
            "She developed hives, wheezing, and swelling of her lips.",
            "She has a history of asthma and known peanut allergy.",
            "On arrival, she is anxious and tachypneic.",
            "Blood pressure is 82/54 mm Hg, pulse 130/min, respirations 32/min, oxygen saturation 88% on room air.",
            "Diffuse wheezing is heard on chest auscultation and her tongue is edematous.",
            "IV access is obtained.",
            "Oxygen is applied via non-rebreather.",
            "The patient is placed in Trendelenburg position.",
            "IM epinephrine 0.3 mg is administered to the anterolateral thigh.",
        ],
        changed_sid=10,
        modified_sentence="IM diphenhydramine 50 mg is administered to the anterolateral thigh.",
        failure_mode="clear_error", expected_verdict="CHANGED", expected_sid=10,
        difficulty="easy",
        rationale="Anaphylaxis with hypotension requires IM epinephrine as first-line; diphenhydramine is only an adjunct.",
    ),
    make_case(
        note_id="synth-1a-02",
        mode="error_injection", error_type="pharmacotherapy",
        sentences=[
            "A 58-year-old man presents with 45 minutes of crushing substernal chest pain radiating to the left arm.",
            "He is diaphoretic and appears in distress.",
            "PMH: hypertension, hyperlipidemia, type 2 diabetes.",
            "Blood pressure 148/92, pulse 96, respirations 20.",
            "ECG shows 3-mm ST elevation in V2-V5 with reciprocal changes in inferior leads.",
            "Troponin is pending.",
            "Cardiac catheterization lab is activated.",
            "Aspirin 324 mg is given PO chewed.",
            "Heparin bolus is administered.",
            "The patient is transferred to the cath lab.",
        ],
        changed_sid=8,
        modified_sentence="Acetaminophen 500 mg is given PO.",
        failure_mode="clear_error", expected_verdict="CHANGED", expected_sid=8,
        difficulty="easy",
        rationale="STEMI mandates aspirin 162-325 mg chewed; acetaminophen has no antiplatelet activity.",
    ),
    make_case(
        note_id="synth-1a-03",
        mode="error_injection", error_type="management",
        sentences=[
            "A 19-year-old man with type 1 diabetes presents with 2 days of nausea, vomiting, and abdominal pain.",
            "He ran out of his insulin last week.",
            "Blood pressure 96/58, pulse 118, respirations 28 with a fruity odor to his breath.",
            "Fingerstick glucose is 512 mg/dL.",
            "Arterial pH 7.10, bicarbonate 8 mEq/L, anion gap 26.",
            "Serum potassium 5.4 mEq/L.",
            "IV normal saline bolus is started.",
            "IV insulin drip is initiated at 0.1 units/kg/hr.",
            "The patient is admitted to the ICU.",
        ],
        changed_sid=8,
        modified_sentence="Subcutaneous insulin sliding scale is ordered every 6 hours.",
        failure_mode="clear_error", expected_verdict="CHANGED", expected_sid=8,
        difficulty="easy",
        rationale="DKA requires continuous IV insulin infusion; SC sliding scale is inadequate and inappropriate for acute DKA.",
    ),
    make_case(
        note_id="synth-1a-04",
        mode="error_injection", error_type="pharmacotherapy",
        sentences=[
            "A 32-year-old woman presents with 12 hours of fever, headache, neck stiffness, and photophobia.",
            "Temperature 39.8°C, pulse 112, blood pressure 118/78.",
            "She is confused and unable to touch chin to chest without severe pain.",
            "Kernig and Brudzinski signs are positive.",
            "Blood cultures are drawn.",
            "Lumbar puncture shows cloudy fluid with 4200 WBCs (95% neutrophils), protein 220 mg/dL, glucose 20 mg/dL.",
            "IV ceftriaxone 2 g and IV vancomycin are started empirically.",
            "IV dexamethasone is given before antibiotics.",
            "The patient is admitted to the ICU.",
        ],
        changed_sid=7,
        modified_sentence="Oral azithromycin 500 mg once daily is started empirically.",
        failure_mode="clear_error", expected_verdict="CHANGED", expected_sid=7,
        difficulty="easy",
        rationale="Bacterial meningitis requires immediate IV ceftriaxone + vancomycin; oral azithromycin does not cross the BBB adequately.",
    ),
    make_case(
        note_id="synth-1a-05",
        mode="error_injection", error_type="pharmacotherapy",
        sentences=[
            "A 74-year-old woman with atrial fibrillation and CHA2DS2-VASc score of 5 presents for medication review.",
            "PMH: hypertension, type 2 diabetes, prior TIA, congestive heart failure.",
            "Blood pressure 138/82, pulse irregularly irregular at 88.",
            "Renal function is normal.",
            "She has no history of bleeding.",
            "Warfarin is initiated with a target INR of 2-3.",
        ],
        changed_sid=6,
        modified_sentence="Warfarin 100 mg daily is initiated with a target INR of 2-3.",
        failure_mode="clear_error", expected_verdict="CHANGED", expected_sid=6,
        difficulty="easy",
        rationale="Warfarin initiation is typically 2.5-5 mg daily; 100 mg is a fatal overdose.",
    ),
    make_case(
        note_id="synth-1a-06",
        mode="error_injection", error_type="management",
        sentences=[
            "A 52-year-old man presents with acute onset of severe pain, redness, and swelling of the right great toe.",
            "He was awakened from sleep by the pain 6 hours ago.",
            "PMH: hypertension, obesity.",
            "Temperature 37.8°C, blood pressure 142/88.",
            "The right first MTP joint is erythematous, warm, and exquisitely tender to touch.",
            "Serum uric acid is 9.2 mg/dL.",
            "Joint aspiration shows negatively birefringent needle-shaped crystals.",
            "Indomethacin 50 mg PO three times daily is prescribed.",
        ],
        changed_sid=8,
        modified_sentence="Allopurinol 100 mg daily is started for the acute attack.",
        failure_mode="clear_error", expected_verdict="CHANGED", expected_sid=8,
        difficulty="easy",
        rationale="Allopurinol should NOT be initiated during acute gout; it can prolong or worsen the attack. NSAIDs, colchicine, or steroids are first-line.",
    ),
    make_case(
        note_id="synth-1a-07",
        mode="error_injection", error_type="management",
        sentences=[
            "An 11-month-old boy is brought to the emergency department for 3 days of vomiting and watery diarrhea.",
            "He has had decreased urine output for the past 12 hours.",
            "Temperature 38.2°C, pulse 168, blood pressure 78/44, capillary refill 4 seconds.",
            "He is lethargic with sunken eyes and dry mucous membranes.",
            "Weight is 9 kg.",
            "IV access is obtained.",
            "Normal saline 20 mL/kg IV bolus is given.",
            "Repeat vitals are obtained after the bolus.",
        ],
        changed_sid=7,
        modified_sentence="D5W 20 mL/kg IV bolus is given.",
        failure_mode="clear_error", expected_verdict="CHANGED", expected_sid=7,
        difficulty="easy",
        rationale="Volume resuscitation in pediatric dehydration requires isotonic fluid (NS or LR); D5W is hypotonic once glucose is metabolized and can worsen hyponatremia.",
    ),
    make_case(
        note_id="synth-1a-08",
        mode="error_injection", error_type="diagnosis",
        sentences=[
            "A 42-year-old woman presents with 3 hours of sudden onset chest pain and shortness of breath.",
            "She returned from a 12-hour flight yesterday.",
            "PMH: oral contraceptive use, prior calf DVT 2 years ago.",
            "Blood pressure 108/68, pulse 118, respirations 26, oxygen saturation 89% on room air.",
            "Lungs are clear.",
            "CT pulmonary angiogram shows a large filling defect in the right main pulmonary artery.",
            "The patient is diagnosed with pulmonary embolism.",
        ],
        changed_sid=7,
        modified_sentence="The patient is diagnosed with acute pericarditis.",
        failure_mode="clear_error", expected_verdict="CHANGED", expected_sid=7,
        difficulty="easy",
        rationale="CTPA showing pulmonary artery filling defect is diagnostic of PE, not pericarditis (which would show diffuse ST elevation on ECG, no imaging finding described).",
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# CATEGORY 1b — CLEAR MEANING-PRESERVING (8 cases)
# Judge should CONFIDENTLY output CORRECT
# ══════════════════════════════════════════════════════════════════════════
CASES_1B = [
    make_case(
        note_id="synth-1b-01", mode="benign",
        sentences=[
            "A 62-year-old man presents for a routine follow-up.",
            "PMH: hypertension, hyperlipidemia.",
            "He takes lisinopril 20 mg daily and atorvastatin 40 mg daily.",
            "Blood pressure 132/78, pulse 72.",
            "Lipid panel shows LDL of 88 mg/dL.",
        ],
        changed_sid=3,
        modified_sentence="He takes lisinopril 20 mg daily and Lipitor 40 mg daily.",
        failure_mode="meaning_preserving_benign", expected_verdict="CORRECT",
        expected_sid=None, difficulty="easy",
        rationale="Lipitor is the brand name for atorvastatin; identical clinical meaning.",
    ),
    make_case(
        note_id="synth-1b-02", mode="benign",
        sentences=[
            "A 55-year-old woman with type 2 diabetes presents for follow-up.",
            "HbA1c is 7.4%.",
            "She is taking metformin 1000 mg twice daily.",
            "She reports occasional GI upset.",
            "Blood pressure 128/74.",
        ],
        changed_sid=3,
        modified_sentence="She is taking Glucophage 1000 mg twice daily.",
        failure_mode="meaning_preserving_benign", expected_verdict="CORRECT",
        expected_sid=None, difficulty="easy",
        rationale="Glucophage is the brand name for metformin; identical clinical meaning.",
    ),
    make_case(
        note_id="synth-1b-03", mode="benign",
        sentences=[
            "A 45-year-old woman presents for annual physical exam.",
            "PMH: T2DM, HTN, hyperlipidemia.",
            "Medications include metformin, lisinopril, and atorvastatin.",
            "Blood pressure 128/78, BMI 31.",
            "HbA1c 6.8%.",
        ],
        changed_sid=2,
        modified_sentence="PMH: type 2 diabetes mellitus, hypertension, hyperlipidemia.",
        failure_mode="meaning_preserving_benign", expected_verdict="CORRECT",
        expected_sid=None, difficulty="easy",
        rationale="Expansion of standard medical abbreviations; identical clinical meaning.",
    ),
    make_case(
        note_id="synth-1b-04", mode="benign",
        sentences=[
            "A 68-year-old man with a prior myocardial infarction 3 years ago presents for follow-up.",
            "He is asymptomatic and adherent to medications.",
            "Aspirin 81 mg, atorvastatin 40 mg, and metoprolol succinate 50 mg daily.",
            "Blood pressure 118/72, pulse 62.",
            "ECG shows old Q waves in the inferior leads, no acute changes.",
        ],
        changed_sid=1,
        modified_sentence="A 68-year-old man with a prior MI 3 years ago presents for follow-up.",
        failure_mode="meaning_preserving_benign", expected_verdict="CORRECT",
        expected_sid=None, difficulty="easy",
        rationale="Contraction of 'myocardial infarction' to 'MI' is standard medical shorthand; identical meaning.",
    ),
    make_case(
        note_id="synth-1b-05", mode="benign",
        sentences=[
            "A 34-year-old woman presents with fatigue for the past 7 days.",
            "She denies weight loss, fever, or night sweats.",
            "PMH: hypothyroidism on levothyroxine.",
            "TSH is 3.8 mIU/L.",
            "Complete blood count shows hemoglobin 10.8 g/dL.",
        ],
        changed_sid=1,
        modified_sentence="A 34-year-old woman presents with fatigue for the past 1 week.",
        failure_mode="meaning_preserving_benign", expected_verdict="CORRECT",
        expected_sid=None, difficulty="easy",
        rationale="'1 week' and '7 days' are exact temporal equivalents.",
    ),
    make_case(
        note_id="synth-1b-06", mode="benign",
        sentences=[
            "A 71-year-old man presents for follow-up 24 months after his aortic valve replacement.",
            "He denies chest pain, dyspnea, or syncope.",
            "Blood pressure 128/76, pulse 68.",
            "Cardiac exam: mechanical S2 click without murmurs.",
            "INR is 2.4.",
        ],
        changed_sid=1,
        modified_sentence="A 71-year-old man presents for follow-up 2 years after his aortic valve replacement.",
        failure_mode="meaning_preserving_benign", expected_verdict="CORRECT",
        expected_sid=None, difficulty="easy",
        rationale="'24 months' and '2 years' are exact temporal equivalents.",
    ),
    make_case(
        note_id="synth-1b-07", mode="benign",
        sentences=[
            "A 5-year-old girl is brought to the ED for pyrexia and rash.",
            "Symptoms began 3 days ago.",
            "Temperature 39.4°C.",
            "Physical exam shows a diffuse maculopapular rash.",
            "The mother reports she has not received the MMR vaccine.",
        ],
        changed_sid=1,
        modified_sentence="A 5-year-old girl is brought to the ED for fever and rash.",
        failure_mode="meaning_preserving_benign", expected_verdict="CORRECT",
        expected_sid=None, difficulty="easy",
        rationale="'Pyrexia' is a medical synonym for 'fever'; identical meaning.",
    ),
    make_case(
        note_id="synth-1b-08", mode="benign",
        sentences=[
            "A 29-year-old woman presents with nausea, vomiting, and abdominal pain for 6 hours.",
            "Pain is in the right lower quadrant, 8/10, sharp.",
            "Temperature 37.9°C, WBC 14,200.",
            "Abdominal exam reveals rebound tenderness at McBurney point.",
            "CT abdomen shows an inflamed appendix.",
        ],
        changed_sid=1,
        modified_sentence="A 29-year-old woman presents with vomiting, nausea, and abdominal pain for 6 hours.",
        failure_mode="meaning_preserving_benign", expected_verdict="CORRECT",
        expected_sid=None, difficulty="easy",
        rationale="Reordering symptoms in a list does not change clinical meaning.",
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# CATEGORY 2a — MEDIUM REAL ERRORS (6 cases)
# Judge SHOULD get it but with less certainty
# ══════════════════════════════════════════════════════════════════════════
CASES_2A = [
    make_case(
        note_id="synth-2a-01",
        mode="error_injection", error_type="pharmacotherapy",
        sentences=[
            "A 48-year-old man presents to establish care.",
            "PMH: hypertension, moderate persistent asthma requiring daily ICS/LABA.",
            "Blood pressure is 152/94 on two visits.",
            "He is not currently on antihypertensive medication.",
            "The physician initiates propranolol 40 mg twice daily for blood pressure control.",
        ],
        changed_sid=5,
        modified_sentence="The physician initiates propranolol 40 mg twice daily for blood pressure control.",
        failure_mode="medium_error", expected_verdict="CHANGED", expected_sid=5,
        difficulty="medium",
        rationale="Non-selective beta-blockers (propranolol) are contraindicated in asthma; cardioselective beta-blockers or other agents (ACEi, thiazide) should be used.",
    ),
    make_case(
        note_id="synth-2a-02",
        mode="error_injection", error_type="management",
        sentences=[
            "A 56-year-old woman presents with chest pain and shortness of breath.",
            "She has a documented severe iodinated contrast allergy with prior anaphylaxis.",
            "Blood pressure 138/84, pulse 92, oxygen saturation 94% on room air.",
            "ECG shows nonspecific ST-T changes.",
            "D-dimer is elevated.",
            "The physician orders a CT pulmonary angiogram with IV contrast to evaluate for PE.",
        ],
        changed_sid=6,
        modified_sentence="The physician orders a CT pulmonary angiogram with IV contrast to evaluate for PE.",
        failure_mode="medium_error", expected_verdict="CHANGED", expected_sid=6,
        difficulty="medium",
        rationale="Documented anaphylactic contrast allergy is a strong contraindication; V/Q scan is the preferred alternative for PE workup.",
    ),
    make_case(
        note_id="synth-2a-03",
        mode="error_injection", error_type="management",
        sentences=[
            "A 47-year-old woman presents with a new palpable breast mass in the upper outer quadrant.",
            "It is firm, non-tender, and fixed to underlying tissue, measuring 2 cm.",
            "There is no skin dimpling or nipple discharge.",
            "Family history: mother with breast cancer at age 52.",
            "A follow-up appointment is scheduled for 6 months from today.",
        ],
        changed_sid=5,
        modified_sentence="A follow-up appointment is scheduled for 6 months from today.",
        failure_mode="medium_error", expected_verdict="CHANGED", expected_sid=5,
        difficulty="medium",
        rationale="Palpable fixed breast mass with concerning family history requires urgent imaging (diagnostic mammogram + US) and biopsy, not 6-month observation.",
    ),
    make_case(
        note_id="synth-2a-04",
        mode="error_injection", error_type="management",
        sentences=[
            "A 68-year-old man with type 2 diabetes on metformin presents with sepsis from pyelonephritis.",
            "He is admitted for IV antibiotics.",
            "Serum creatinine on admission is 2.8 mg/dL (baseline 1.0).",
            "Blood pressure 92/58, pulse 108.",
            "IV fluids are started.",
            "His home metformin is continued during admission.",
        ],
        changed_sid=6,
        modified_sentence="His home metformin is continued during admission.",
        failure_mode="medium_error", expected_verdict="CHANGED", expected_sid=6,
        difficulty="medium",
        rationale="Metformin should be held during AKI (creatinine has tripled from baseline) due to risk of lactic acidosis.",
    ),
    make_case(
        note_id="synth-2a-05",
        mode="error_injection", error_type="pharmacotherapy",
        sentences=[
            "A 62-year-old man presents 2 days after a large anterior STEMI managed with PCI.",
            "LDL cholesterol on admission was 148 mg/dL.",
            "He is being discharged on aspirin, ticagrelor, metoprolol, and lisinopril.",
            "Blood pressure 118/72, pulse 64.",
            "The physician adds pravastatin 10 mg daily to his regimen.",
        ],
        changed_sid=5,
        modified_sentence="The physician adds pravastatin 10 mg daily to his regimen.",
        failure_mode="medium_error", expected_verdict="CHANGED", expected_sid=5,
        difficulty="medium",
        rationale="Post-MI patients require high-intensity statin therapy (atorvastatin 80 mg or rosuvastatin 20-40 mg); pravastatin 10 mg is inadequate for secondary prevention.",
    ),
    make_case(
        note_id="synth-2a-06",
        mode="error_injection", error_type="management",
        sentences=[
            "A 44-year-old woman on chronic prednisone 40 mg daily for 8 months for autoimmune hepatitis presents for follow-up.",
            "Her disease is now in remission.",
            "The physician instructs her to stop prednisone immediately today.",
            "Follow-up is scheduled in 4 weeks.",
        ],
        changed_sid=3,
        modified_sentence="The physician instructs her to stop prednisone immediately today.",
        failure_mode="medium_error", expected_verdict="CHANGED", expected_sid=3,
        difficulty="medium",
        rationale="After 8 months of high-dose steroids, abrupt cessation risks adrenal crisis; a slow taper over weeks to months is required.",
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# CATEGORY 2b — MEDIUM MEANING-PRESERVING (6 cases)
# Judge SHOULD say CORRECT but with less certainty
# ══════════════════════════════════════════════════════════════════════════
CASES_2B = [
    make_case(
        note_id="synth-2b-01", mode="benign",
        sentences=[
            "A 60-year-old man with a long history of high blood pressure presents for follow-up.",
            "Current medications: amlodipine 10 mg daily, HCTZ 25 mg daily.",
            "Blood pressure today is 138/82.",
            "He denies chest pain or shortness of breath.",
            "Basic metabolic panel is unremarkable.",
        ],
        changed_sid=1,
        modified_sentence="A 60-year-old man with a long history of hypertension presents for follow-up.",
        failure_mode="meaning_preserving_medium", expected_verdict="CORRECT",
        expected_sid=None, difficulty="medium",
        rationale="'High blood pressure' and 'hypertension' are semantically identical.",
    ),
    make_case(
        note_id="synth-2b-02", mode="benign",
        sentences=[
            "A 33-year-old woman presents for evaluation of intermittent palpitations.",
            "She reports the episodes last 5-10 minutes and self-resolve.",
            "ECG in the office shows normal sinus rhythm.",
            "A 24-hour Holter monitor was ordered by the physician.",
            "Follow-up is scheduled in 2 weeks.",
        ],
        changed_sid=4,
        modified_sentence="The physician ordered a 24-hour Holter monitor.",
        failure_mode="meaning_preserving_medium", expected_verdict="CORRECT",
        expected_sid=None, difficulty="medium",
        rationale="Passive-to-active voice conversion with identical clinical content.",
    ),
    make_case(
        note_id="synth-2b-03", mode="benign",
        sentences=[
            "A 51-year-old man is admitted for community-acquired pneumonia.",
            "His initial workup includes CBC, BMP, blood cultures, and a chest X-ray.",
            "Vitals: BP 118/74, HR 96, RR 22, temp 38.7°C.",
            "Pulse oximetry shows 92% on room air.",
            "IV ceftriaxone and azithromycin are started.",
        ],
        changed_sid=2,
        modified_sentence="His initial workup includes complete blood count, basic metabolic panel, blood cultures, and a chest radiograph.",
        failure_mode="meaning_preserving_medium", expected_verdict="CORRECT",
        expected_sid=None, difficulty="medium",
        rationale="Abbreviation expansion; 'chest X-ray' and 'chest radiograph' are synonymous.",
    ),
    make_case(
        note_id="synth-2b-04", mode="benign",
        sentences=[
            "A 27-year-old woman presents with 2 days of dysuria and urinary frequency.",
            "She denies fever, flank pain, or hematuria.",
            "Urinalysis shows leukocyte esterase positive, nitrites positive, WBCs 25-50/hpf.",
            "She is prescribed nitrofurantoin 100 mg PO BID for 5 days.",
        ],
        changed_sid=4,
        modified_sentence="She is prescribed nitrofurantoin 100 mg twice daily by mouth for 5 days.",
        failure_mode="meaning_preserving_medium", expected_verdict="CORRECT",
        expected_sid=None, difficulty="medium",
        rationale="Abbreviations 'PO' and 'BID' expanded to 'by mouth' and 'twice daily'; identical meaning.",
    ),
    make_case(
        note_id="synth-2b-05", mode="benign",
        sentences=[
            "A 40-year-old man presents with sudden onset of unilateral throbbing headache, photophobia, and nausea.",
            "Similar episodes have occurred 4-5 times per month for the past 6 months.",
            "Neurological exam is normal.",
            "The patient is diagnosed with migraine without aura.",
            "Sumatriptan 50 mg PO is prescribed for acute attacks.",
        ],
        changed_sid=1,
        modified_sentence="A 40-year-old man presents with sudden onset of one-sided pulsating headache, light sensitivity, and nausea.",
        failure_mode="meaning_preserving_medium", expected_verdict="CORRECT",
        expected_sid=None, difficulty="medium",
        rationale="Descriptor synonyms: unilateral=one-sided, throbbing=pulsating, photophobia=light sensitivity; identical meaning.",
    ),
    make_case(
        note_id="synth-2b-06", mode="benign",
        sentences=[
            "A 39-year-old woman presents with fatigue, weight gain, cold intolerance, and constipation for 4 months.",
            "PMH: uneventful.",
            "TSH is 12.4 mIU/L, free T4 is 0.6 ng/dL.",
            "She is diagnosed with primary hypothyroidism.",
            "Levothyroxine 75 mcg daily is started.",
        ],
        changed_sid=1,
        modified_sentence="A 39-year-old woman presents with 4 months of fatigue, weight gain, cold intolerance, and constipation.",
        failure_mode="meaning_preserving_medium", expected_verdict="CORRECT",
        expected_sid=None, difficulty="medium",
        rationale="Sentence restructure moves duration to the front; symptom list and clinical meaning unchanged.",
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# CATEGORY 3a — SUBTLE REAL ERRORS (6 cases)
# Adversarial: judge may miss these (CRITICAL CALIBRATION PROBE)
# ══════════════════════════════════════════════════════════════════════════
CASES_3A = [
    make_case(
        note_id="synth-3a-01",
        mode="error_injection", error_type="diagnosis",
        sentences=[
            "A 34-year-old woman presents with a rash that appeared on her left forearm after a scratch three days ago.",
            "She reports a similar lesion on her elbows three years ago that resolved with topical treatment.",
            "Her older sister has a history of skin problems.",
            "On physical exam you find a circular, scaly, pearl-colored rash superimposed over an abrasion on her left forearm.",
            "Nail exam shows pitting.",
            "The patient is diagnosed with psoriasis.",
        ],
        changed_sid=4,
        modified_sentence="On physical exam you find a circular, scaly, pearl-colored rash superimposed over an abrasion on her left forearm.",
        failure_mode="real_judge_failure", expected_verdict="CHANGED", expected_sid=4,
        difficulty="hard",
        rationale="Psoriasis has silvery-white scales, not pearl-colored. 'Pearl-colored' describes basal cell carcinoma; descriptor is inconsistent with the stated diagnosis. Requires careful clinical reasoning.",
    ),
    make_case(
        note_id="synth-3a-02",
        mode="error_injection", error_type="pharmacotherapy",
        sentences=[
            "A 44-year-old woman presents with 6 months of episodic headaches, palpitations, and diaphoresis.",
            "Between episodes she is asymptomatic.",
            "During an episode her blood pressure was recorded at 220/130.",
            "24-hour urine metanephrines are 4x elevated.",
            "Abdominal MRI shows a 4-cm right adrenal mass.",
            "She is being prepared for surgical resection.",
            "Propranolol 40 mg twice daily is started for blood pressure control prior to surgery.",
        ],
        changed_sid=7,
        modified_sentence="Propranolol 40 mg twice daily is started for blood pressure control prior to surgery.",
        failure_mode="real_judge_failure", expected_verdict="CHANGED", expected_sid=7,
        difficulty="hard",
        rationale="Pheochromocytoma pre-op requires alpha-blockade (phenoxybenzamine) BEFORE beta-blockade; unopposed beta-blockade with high catecholamines causes hypertensive crisis. Nuanced but well-established.",
    ),
    make_case(
        note_id="synth-3a-03",
        mode="error_injection", error_type="management",
        sentences=[
            "A 62-year-old man with CKD presents with weakness and lethargy.",
            "PMH: hypertension, T2DM, CKD stage 4.",
            "Blood pressure 148/88, pulse 74.",
            "Serum potassium is 5.8 mEq/L, bicarbonate 18 mEq/L.",
            "ECG shows peaked T waves in V2-V4.",
            "Calcium gluconate is administered.",
            "Potassium 20 mEq PO is added to his home regimen.",
        ],
        changed_sid=7,
        modified_sentence="Potassium 20 mEq PO is added to his home regimen.",
        failure_mode="real_judge_failure", expected_verdict="CHANGED", expected_sid=7,
        difficulty="hard",
        rationale="Patient is HYPERkalemic (K+ 5.8) with ECG changes; adding potassium supplementation would worsen the emergency. Requires recognizing the number relative to normal.",
    ),
    make_case(
        note_id="synth-3a-04",
        mode="error_injection", error_type="pharmacotherapy",
        sentences=[
            "A 24-year-old man presents with 8 hours of fever, headache, and stiff neck.",
            "Temperature 39.6°C, pulse 118, blood pressure 108/62.",
            "He is somnolent but arousable.",
            "Kernig sign is positive.",
            "Lumbar puncture shows 3800 WBCs (91% neutrophils), protein 240 mg/dL, glucose 22 mg/dL.",
            "Gram stain shows Gram-positive diplococci.",
            "IV ceftriaxone 2 g and IV vancomycin are started.",
        ],
        changed_sid=7,
        modified_sentence="IV ceftriaxone 2 g and IV vancomycin are started.",
        failure_mode="real_judge_failure", expected_verdict="CHANGED", expected_sid=7,
        difficulty="hard",
        rationale="Bacterial meningitis in an adult also requires adjunctive dexamethasone (given before or with the first antibiotic dose); omitting it is a well-documented management error. Subtle — the antibiotics are correct, the OMISSION is the error.",
    ),
    make_case(
        note_id="synth-3a-05",
        mode="error_injection", error_type="diagnosis",
        sentences=[
            "A 22-year-old college student is brought to the ED by his roommate for confusion.",
            "PMH: type 1 diabetes on insulin pump.",
            "Fingerstick glucose in the field was 38 mg/dL.",
            "IV dextrose was administered en route.",
            "On arrival his mental status has improved, glucose is now 128 mg/dL.",
            "Kussmaul breathing is noted, thought to be secondary to the hypoglycemic episode.",
            "He is admitted for observation.",
        ],
        changed_sid=6,
        modified_sentence="Kussmaul breathing is noted, thought to be secondary to the hypoglycemic episode.",
        failure_mode="real_judge_failure", expected_verdict="CHANGED", expected_sid=6,
        difficulty="hard",
        rationale="Kussmaul breathing is a compensatory response to metabolic acidosis (classically DKA), NOT hypoglycemia. Attribution is clinically wrong and requires recognizing the pattern-diagnosis mismatch.",
    ),
    make_case(
        note_id="synth-3a-06",
        mode="error_injection", error_type="diagnosis",
        sentences=[
            "A 68-year-old right-handed man presents with sudden onset of left-sided face droop, right arm weakness, and expressive aphasia 90 minutes ago.",
            "Blood pressure 168/92, pulse 88, glucose 118.",
            "NIH stroke scale is 12.",
            "Non-contrast CT shows no hemorrhage.",
            "CT angiography confirms a left middle cerebral artery M1 occlusion.",
            "The patient is diagnosed with a right hemispheric ischemic stroke.",
            "Thrombolysis with alteplase is initiated.",
        ],
        changed_sid=6,
        modified_sentence="The patient is diagnosed with a right hemispheric ischemic stroke.",
        failure_mode="real_judge_failure", expected_verdict="CHANGED", expected_sid=6,
        difficulty="hard",
        rationale="Aphasia + right-arm weakness in a right-handed patient localizes to the LEFT hemisphere (dominant hemisphere for language). CTA confirmed left MCA occlusion. Diagnosis 'right hemispheric' is wrong laterality — subtle because it contradicts the imaging in the same note.",
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# CATEGORY 3b — MEANING-PRESERVING THAT COULD BE OVER-FLAGGED (4 cases)
# Judge might incorrectly say CHANGED (CRITICAL CALIBRATION PROBE)
# ══════════════════════════════════════════════════════════════════════════
CASES_3B = [
    make_case(
        note_id="synth-3b-01", mode="benign",
        sentences=[
            "A 33-year-old woman presents with 3 weeks of right jaw pain worse in the morning.",
            "Her husband has noticed her grinding her teeth at night.",
            "She denies jaw locking or clicking prior to symptom onset.",
            "PMH: depression on fluoxetine.",
            "On exam, mild tenderness at the right angle of the mandible; jaw opening produces a slight click.",
            "The remainder of the exam is unremarkable.",
            "Nighttime bite guard is recommended.",
        ],
        changed_sid=7,
        modified_sentence="Nighttime bite guard is recommended.",
        failure_mode="judge_over_flag_candidate", expected_verdict="CORRECT",
        expected_sid=None, difficulty="hard",
        rationale="TMJ with bruxism is correctly managed with nighttime occlusal splint (bite guard). Recommendation is clinically appropriate; judge may over-flag due to specificity of recommendation.",
    ),
    make_case(
        note_id="synth-3b-02", mode="benign",
        sentences=[
            "A 5-mm nodule is noted on chest CT performed for another indication.",
            "The patient has never smoked.",
            "Family history is negative for lung cancer.",
            "Watchful waiting with repeat CT in 12 months is recommended per Fleischner guidelines.",
        ],
        changed_sid=4,
        modified_sentence="Conservative management with repeat CT in 12 months is recommended per Fleischner guidelines.",
        failure_mode="judge_over_flag_candidate", expected_verdict="CORRECT",
        expected_sid=None, difficulty="hard",
        rationale="'Watchful waiting' and 'conservative management' are semantically equivalent in this context; both describe active surveillance without intervention.",
    ),
    make_case(
        note_id="synth-3b-03", mode="benign",
        sentences=[
            "A 45-year-old man presents to the ED with fever and productive cough for 4 days.",
            "Temperature is 100.4°F, blood pressure 128/78, pulse 96.",
            "Respiratory exam reveals crackles at the right base.",
            "Chest X-ray shows a right lower lobe infiltrate.",
        ],
        changed_sid=2,
        modified_sentence="Temperature is 38.0°C, blood pressure 128/78, pulse 96.",
        failure_mode="judge_over_flag_candidate", expected_verdict="CORRECT",
        expected_sid=None, difficulty="hard",
        rationale="100.4°F and 38.0°C are the exact same temperature (converted precisely); unit change with identical clinical meaning.",
    ),
    make_case(
        note_id="synth-3b-04", mode="benign",
        sentences=[
            "A 55-year-old man presents with 30 minutes of substernal chest pressure radiating to his left arm.",
            "PMH: hypertension, hyperlipidemia, smoking.",
            "Blood pressure 148/88, pulse 92.",
            "An EKG is obtained and shows 2-mm ST depressions in V4-V6.",
            "Troponin is ordered.",
        ],
        changed_sid=4,
        modified_sentence="An ECG is obtained and shows 2-mm ST depressions in V4-V6.",
        failure_mode="judge_over_flag_candidate", expected_verdict="CORRECT",
        expected_sid=None, difficulty="medium",
        rationale="'EKG' and 'ECG' are interchangeable abbreviations for electrocardiogram; identical meaning.",
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# CATEGORY 3c — INJECTOR OVER-EDIT (real error introduced in benign mode) (4)
# ══════════════════════════════════════════════════════════════════════════
CASES_3C = [
    make_case(
        note_id="synth-3c-01", mode="benign",
        sentences=[
            "A 36-year-old woman presents for a routine gynecological examination.",
            "She feels well.",
            "Menses occur with normal flow at regular 28-day intervals.",
            "Her last menstrual period was 20 days ago.",
            "She is sexually active with one partner and uses condoms.",
            "She takes a daily multivitamin.",
            "Physical examination and pelvic exam are unremarkable.",
        ],
        changed_sid=6,
        modified_sentence="She takes daily oral tinctures of local anesthetics, ephedra, ginkgo biloba, and bacopa infructus.",
        failure_mode="injector_over_edit", expected_verdict="CHANGED", expected_sid=6,
        difficulty="medium",
        rationale="Oral tinctures of local anesthetics are dangerous (systemic toxicity); ephedra was banned by the FDA in 2004; 'bacopa infructus' is an incorrect binomial. Introduces multiple clinical inaccuracies in what should be a benign rewrite.",
    ),
    make_case(
        note_id="synth-3c-02", mode="benign",
        sentences=[
            "A 22-year-old man presents for a preventive care visit.",
            "He is a college student, plays intramural soccer, and reports feeling well.",
            "PMH: unremarkable.",
            "No known drug allergies.",
            "Vital signs are within normal limits.",
            "Immunizations are up to date.",
        ],
        changed_sid=4,
        modified_sentence="Allergies: penicillin (anaphylaxis) and sulfa drugs (severe rash).",
        failure_mode="injector_over_edit", expected_verdict="CHANGED", expected_sid=4,
        difficulty="medium",
        rationale="Original clearly states 'No known drug allergies'; adding two severe allergies fabricates clinical history that would affect future prescribing decisions.",
    ),
    make_case(
        note_id="synth-3c-03", mode="benign",
        sentences=[
            "A 41-year-old woman presents for annual physical.",
            "She feels well and denies specific complaints.",
            "PMH: hypothyroidism on levothyroxine.",
            "Blood pressure 118/74, pulse 68, temperature 36.8°C.",
            "Weight is stable, BMI 24.",
            "Physical exam is unremarkable.",
        ],
        changed_sid=4,
        modified_sentence="Blood pressure 198/124, pulse 68, temperature 36.8°C.",
        failure_mode="injector_over_edit", expected_verdict="CHANGED", expected_sid=4,
        difficulty="medium",
        rationale="Original BP 118/74 (normal) changed to 198/124 (hypertensive urgency/emergency range); clinically meaningful change not consistent with 'benign rewrite'.",
    ),
    make_case(
        note_id="synth-3c-04", mode="benign",
        sentences=[
            "A 58-year-old man with a history of atrial fibrillation on warfarin presents for INR check.",
            "He denies bleeding or falls.",
            "Blood pressure 128/78, pulse 76, irregularly irregular.",
            "INR is 2.4.",
            "Continue current warfarin dose.",
            "Recheck in 4 weeks.",
        ],
        changed_sid=1,
        modified_sentence="A 58-year-old man with a history of atrial fibrillation on warfarin and aspirin 325 mg daily presents for INR check.",
        failure_mode="injector_over_edit", expected_verdict="CHANGED", expected_sid=1,
        difficulty="medium",
        rationale="Adding aspirin 325 mg to warfarin without a clear indication introduces significant bleeding risk; changes the medication regimen materially, not just a rewording.",
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# CATEGORY 4 — INJECTOR-SIDE FAILURES (truncation + garbled) (6 cases)
# ══════════════════════════════════════════════════════════════════════════
CASES_4 = [
    make_case(
        note_id="synth-4-01", mode="benign",
        sentences=[
            "A 14-year-old boy is brought to the physician for evaluation of back pain over the past 6 months.",
            "The pain is worse with exercise and reclining.",
            "He plays on the high school swim team and lifts weights regularly.",
            "No trauma or joint problems reported.",
            "His father has a disc herniation.",
            "On exam, palpation of the lumbosacral spinous processes reveals a step-off deformity.",
            "Waddling gait is noted.",
            "The patient is diagnosed with spondylolisthesis.",
        ],
        changed_sid=8,
        modified_sentence="The patient is diagnosed with sl.",
        failure_mode="injector_truncation", expected_verdict="CHANGED", expected_sid=8,
        difficulty="easy",
        rationale="Truncation mid-word ('sl.' instead of 'spondylolisthesis'). Judge should easily flag as text-changed.",
    ),
    make_case(
        note_id="synth-4-02", mode="benign",
        sentences=[
            "A 51-year-old woman with type 2 diabetes presents for follow-up.",
            "HbA1c is 8.4%.",
            "She has been non-adherent to her current oral regimen.",
            "The physician started her on metformin extended-release 1000 mg twice daily.",
            "Follow-up in 3 months with repeat HbA1c.",
        ],
        changed_sid=4,
        modified_sentence="The physician started her on met",
        failure_mode="injector_truncation", expected_verdict="CHANGED", expected_sid=4,
        difficulty="easy",
        rationale="Sentence truncated mid-word after 'met' (metformin cut off); grammatically incomplete.",
    ),
    make_case(
        note_id="synth-4-03", mode="benign",
        sentences=[
            "A 68-year-old man presents with worsening dyspnea over 2 weeks.",
            "PMH: HFrEF with LVEF 30%, on GDMT.",
            "JVP is elevated at 12 cm.",
            "Bibasilar crackles on lung auscultation.",
            "The physician increased his furosemide dose and scheduled follow-up in one week for reassessment of volume status.",
        ],
        changed_sid=5,
        modified_sentence="The physician increased his furosemide dose and scheduled",
        failure_mode="injector_truncation", expected_verdict="CHANGED", expected_sid=5,
        difficulty="easy",
        rationale="Sentence truncated after 'scheduled', missing key management detail; incomplete text.",
    ),
    make_case(
        note_id="synth-4-04", mode="benign",
        sentences=[
            "<NAME/> is a <AGE/> yo man with recurrent SCC of the left oral cavity currently on radiation therapy.",
            "He is admitted with left facial cellulitis and started on IV Abx.",
            "Prealbumin is 10.2 on <DATE/> and 9.4 on <DATE/>; BMI 21.",
            "The nutrition team recommends placement of a Dobhoff feeding tube with optimization and G-tube placement in 2-3 weeks.",
            "The patient is also being managed for protein-calorie malnutrition.",
        ],
        changed_sid=4,
        modified_sentence="The nutrition team recommends placement of a Kao feeding tube with optimization and G-tube placement in 2-3 weeks.",
        failure_mode="injector_garbled", expected_verdict="CHANGED", expected_sid=4,
        difficulty="easy",
        rationale="'Kao feeding tube' is not a recognized medical device; likely a garbled hallucination replacing 'Dobhoff'.",
    ),
    make_case(
        note_id="synth-4-05", mode="benign",
        sentences=[
            "This patient was admitted with a concomitant diagnosis of hyponatremia.",
            "Ms. <NAME/> presented with intractable nausea and vomiting of unclear etiology.",
            "Her Na was 134 and management has included 3 L of NS with monitoring of her BMP.",
        ],
        changed_sid=2,
        modified_sentence="Ms. <NAME/> presented with intractable nausea and vomiting of s/p etiology.",
        failure_mode="injector_garbled", expected_verdict="CHANGED", expected_sid=2,
        difficulty="easy",
        rationale="'s/p etiology' is grammatically nonsensical ('s/p' means 'status post', doesn't modify 'etiology'). Broken medical text.",
    ),
    make_case(
        note_id="synth-4-06", mode="benign",
        sentences=[
            "A 60-year-old woman with COPD presents for follow-up.",
            "She reports increased dyspnea on exertion over the past month.",
            "PMH: 40 pack-year smoking history, quit 5 years ago.",
            "She is currently on tiotropium and albuterol as needed.",
            "Pulmonary function tests show FEV1/FVC of 0.62 with FEV1 55% predicted.",
            "The physician adds a long-acting beta-agonist to her regimen and orders pulmonary rehabilitation.",
        ],
        changed_sid=6,
        modified_sentence="The physician the patient acute add long-acting rehabilitation orders beta-agonist to regimen her.",
        failure_mode="injector_garbled", expected_verdict="CHANGED", expected_sid=6,
        difficulty="easy",
        rationale="Word-order scrambled into ungrammatical sequence; content is unparseable as coherent medical documentation.",
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════
ALL_CASES = (
    CASES_1A + CASES_1B + CASES_2A + CASES_2B
    + CASES_3A + CASES_3B + CASES_3C + CASES_4
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="data_processed/synthetic_test/game_synthetic_v1.jsonl",
    )
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    with open(out, "w") as f:
        for case in ALL_CASES:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # Write manifest
    from collections import Counter
    counts_by_mode = Counter(c["mode"] for c in ALL_CASES)
    counts_by_failure = Counter(c["failure_mode"] for c in ALL_CASES)
    counts_by_difficulty = Counter(c["difficulty"] for c in ALL_CASES)
    counts_by_expected = Counter(c["expected_verdict"] for c in ALL_CASES)

    manifest = {
        "total_cases": len(ALL_CASES),
        "by_mode": dict(counts_by_mode),
        "by_failure_mode": dict(counts_by_failure),
        "by_difficulty": dict(counts_by_difficulty),
        "by_expected_verdict": dict(counts_by_expected),
    }
    manifest_path = out.with_name(out.stem + "_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(ALL_CASES)} cases to: {out}")
    print(f"Manifest: {manifest_path}")
    print()
    print(f"By mode:            {dict(counts_by_mode)}")
    print(f"By expected verdict: {dict(counts_by_expected)}")
    print(f"By difficulty:      {dict(counts_by_difficulty)}")
    print()
    print("By failure mode:")
    for k in sorted(counts_by_failure):
        print(f"  {k:<32s} {counts_by_failure[k]}")


if __name__ == "__main__":
    main()
