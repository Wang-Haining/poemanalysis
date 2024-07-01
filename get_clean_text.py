import json
import transformers
import torch
from transformers import BitsAndBytesConfig

# Load the model and tokenizer
model_id = "gradientai/Llama-3-8B-Instruct-262k"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    quantization_config=nf4_config
)

# define the system prompt for the model
system_prompt = "You are an AI that extracts poems and their interpretations from text. Your responses should be in JSON format with keys 'poem' and 'interpretation'."

def apply_instruction(text, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=4*1024,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.01
    )
    result = outputs[0]["generated_text"][len(prompt):]
    return result

def process_texts_with_instruction(data, system_prompt):
    processed_data = []
    for entry in data:
        text = entry['llm_friendly_text']
        extracted_data = apply_instruction(text, system_prompt)
        try:
            extracted_json = json.loads(extracted_data)
            processed_data.append({
                'title': entry['title'],
                'author': entry['author'],
                'url': entry['url'],
                'extracted_poem': extracted_json.get('poem', ''),
                'extracted_interpretation': extracted_json.get('interpretation', '')
            })
        except json.JSONDecodeError:
            print(f"Failed to decode JSON for entry: {entry['title']}")
            processed_data.append({
                'title': entry['title'],
                'author': entry['author'],
                'url': entry['url'],
                'extracted_poem': '',
                'extracted_interpretation': ''
            })
    return processed_data

def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

# Paths to the files
llm_friendly_text_file_path = 'processed_poem_data.jsonl'
output_file_path = 'final_extracted_poems.jsonl'

# Load LLM-friendly text data
poem_data = []
with open(llm_friendly_text_file_path, 'r') as file:
    for line in file:
        poem_data.append(json.loads(line))

# Process and save the data
processed_poem_data = process_texts_with_instruction(poem_data, system_prompt)
save_to_jsonl(processed_poem_data, output_file_path)

print(f"Data processed and saved to {output_file_path}")
