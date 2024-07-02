import json
import transformers
import torch
from transformers import BitsAndBytesConfig

# load the model and tokenizer
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
        max_new_tokens=8*1024,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.01
    )
    result = outputs[0]["generated_text"][len(prompt):]
    return result


def process_texts_with_instruction(data, processed_urls, system_prompt):
    processed_data = []
    for entry in data:
        if entry['url'] not in processed_urls:
            text = entry['markdown']
            extracted_data = apply_instruction(text, system_prompt)
            try:
                extracted_json = json.loads(extracted_data)
                processed_data.append({'url': entry['url'], 'data': extracted_json})
            except json.JSONDecodeError:
                print(f"Failed to decode JSON for entry: {entry['url']}")
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

# load LLM-friendly text data
poem_data = []
with open(markdown_file_path, 'r') as file:
    for line in file:
        poem_data.append(json.loads(line))

# load already processed URLs
processed_urls = load_processed_urls(output_file_path)

# process new data and save the results
processed_poem_data = process_texts_with_instruction(poem_data, processed_urls, system_prompt)
save_to_jsonl(processed_poem_data, output_file_path)

print(f"Data processed and saved to {output_file_path}")
