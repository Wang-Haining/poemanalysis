import json

def load_from_jsonl(file_path):
    # load data from a JSONL file
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_to_jsonl(file_path, data):
    # save data to a JSONL file
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def deduplicate_entries(entries):
    # deduplicate list of entries based on their URLs
    unique_entries = {}
    for entry in entries:
        unique_entries[entry['url']] = entry
    return list(unique_entries.values())

if __name__ == '__main__':
    success_file = 'poemAnalysis_success.jsonl'

    # load existing successes
    try:
        processed_poem_data = load_from_jsonl(success_file)
    except FileNotFoundError:
        processed_poem_data = []

    # deduplicate entries
    deduplicated_data = deduplicate_entries(processed_poem_data)

    # save deduplicated data back to the file
    save_to_jsonl(success_file, deduplicated_data)

    # report the number of unique entries
    print(f"Number of unique entries after deduplication: {len(deduplicated_data)}")
