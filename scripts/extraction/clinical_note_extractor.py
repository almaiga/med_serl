import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Load prompt template
def load_prompt_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Load medical notes from JSONL
def load_medical_notes(jsonl_path, num_examples=None):
    notes = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if num_examples is not None and i >= num_examples:
                break
            notes.append(json.loads(line))
    return notes

# Initialize model
def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

# Extract using the model
def extract_with_model(model, tokenizer, system_prompt, user_template, note_text, max_new_tokens=512):
    user_input = user_template.format(note_text=note_text)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Save extraction results
def save_extraction(output_path, note_obj, extraction, error_occurred=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "note_id": note_obj.get("note_id"),
        "error_type": note_obj.get("error_type"),
        "corrected_sentence": note_obj.get("corrected_sentence"),
        "extraction": extraction,
        "extraction_error": error_occurred,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, 'a') as f:
        f.write(json.dumps(result) + '\n')

# Main execution
if __name__ == "__main__":
    config_path = Path("configs/prompts/clinical_note_extractor.json")
    data_path = Path("data_processed/medec_paired/train_val_split/sft_train.jsonl")
    output_path = Path("data_processed/parsed_medical_note/extractions.jsonl")
    
    # Load configuration and data
    print("Loading configuration and data...")
    config = load_prompt_config(config_path)
    notes = load_medical_notes(data_path, num_examples=None)
    
    print("Loading Llama 3.1-8B-Instruct...")
    model, tokenizer = load_model()
    
    system_prompt = config.get("system_prompt", "")
    user_template = config.get("user_template", "")
    
    # Process each note with progress bar
    for note_obj in tqdm(notes, desc="Processing notes", unit="note"):
        note_id = note_obj.get('note_id')
        corrected_note = note_obj.get("correct_note", "")
        error_type = note_obj.get("error_type", "")
        corrected_sentence = note_obj.get("corrected_sentence", "")
        
        tqdm.write(f"\nProcessing: {note_id} (Error Type: {error_type})")
        
        # Extract using the model
        try:
            extraction = extract_with_model(model, tokenizer, system_prompt, user_template, corrected_note)
            save_extraction(output_path, note_obj, extraction, error_occurred=False)
            tqdm.write(f"✓ Successfully extracted for {note_id}")
        except Exception as e:
            tqdm.write(f"✗ Error during extraction for {note_id}: {str(e)}")
            save_extraction(output_path, note_obj, str(e), error_occurred=True)
    
    print(f"\n✓ All extractions saved to: {output_path}")