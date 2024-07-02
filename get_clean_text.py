import json
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

# load the model and tokenizer
model_id = "gradientai/Llama-3-8B-Instruct-262k"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=nf4_config
)

# define the system prompt for the model
system_prompt = '''You are an assistant designed to extract poems and their interpretations from text. Your responses should be formatted in JSON with the following structure:
{
  "Poem": "The extracted poem text",
  "Interpretation": {
    "Summary": "The summary section",
    "Structure and Form": "Explanation related to structure and form",
    "Literary Devices": "Explanation related to literary devices",
    ...
  }
}'''


def apply_instruction(text, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=8192,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.01,
    )
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    return result


def process_texts_with_instruction(data, processed_urls, system_prompt):
    processed_data = []
    for entry in data:
        if entry['url'] not in processed_urls:
            text = entry['markdown']
            extracted_data = apply_instruction(text, system_prompt)
            processed_data.append({'url': entry['url'], 'clean_text': extracted_data})
    return processed_data


def save_to_jsonl(data, file_path):
    with open(file_path, 'a') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')


def load_processed_urls(file_path):
    processed_urls = set()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                processed_urls.add(entry['url'])
    except FileNotFoundError:
        pass
    return processed_urls

# paths to the files
markdown_file_path = 'poemAnalysis_success.jsonl'
output_file_path = 'poemAnalysis_corpus.jsonl'

# load markdown files
poem_data = []
with open(markdown_file_path, 'r') as file:
    for line in file:
        poem_data.append(json.loads(line))

# load already processed URLs
processed_urls = load_processed_urls(output_file_path)

# process new data and save the results
processed_poem_data = process_texts_with_instruction(poem_data[:10], processed_urls, system_prompt)
save_to_jsonl(processed_poem_data, output_file_path)

print(f"Data processed and saved to {output_file_path}")
