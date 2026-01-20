import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Load prompt template
def load_prompt_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Load medical notes from JSONL
def load_medical_notes(jsonl_path, num_examples=3):
    notes = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
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

# Main execution
if __name__ == "__main__":
    config_path = Path("configs/prompts/clinical_note_extractor.json")
    data_path = Path("data_processed/medec_paired/train_val_split/sft_train.jsonl")
    
    # Load configuration and data
    print("Loading configuration and data...")
    config = load_prompt_config(config_path)
    notes = load_medical_notes(data_path, num_examples=3)
    
    print("Loading Llama 3.1-8B-Instruct...")
    model, tokenizer = load_model()
    
    system_prompt = config.get("system_prompt", "")
    user_template = config.get("user_template", "")
    
    # Process each note
    for i, note_obj in enumerate(notes):
        print(f"\n{'='*80}")
        print(f"Example {i+1} - Note ID: {note_obj.get('note_id')}")
        print(f"{'='*80}")
        
        corrected_note = note_obj.get("correct_note", "")
        error_type = note_obj.get("error_type", "")
        corrected_sentence = note_obj.get("corrected_sentence", "")
        
        print(f"Error Type: {error_type}")
        print(f"\nGround Truth Corrected Sentence:\n{corrected_sentence}")
        
        # Extract using the model
        print("\nExtracting with model...")
        extraction = extract_with_model(model, tokenizer, system_prompt, user_template, corrected_note)
        
        print(f"\nModel Extraction:\n{extraction}")
        print(f"\n{'='*80}\n")