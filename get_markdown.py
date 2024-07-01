import json
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent


def get_markdown(url, headers):
    # fetch markdown content from the given URL
    jina_reader_url = f'https://r.jina.ai/{url}'
    response = requests.get(jina_reader_url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        return None


def process_entry(entry, headers):
    # process a single entry to fetch its markdown content
    url = entry['url']
    markdown = get_markdown(url, headers)
    if markdown:
        return {
            'title': entry['title'],
            'author': entry['author'],
            'url': url,
            'markdown': markdown
        }
    else:
        return None


def save_to_jsonl(file_path, data):
    # save data to a JSONL file
    with open(file_path, 'a', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def load_from_jsonl(file_path):
    # load data from a JSONL file
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def process_poem_data(poem_data, success_file, max_retries=20, max_workers=4):
    # process poem data with multiple retries and save successes and failures
    ua = UserAgent()
    retries = 0
    failures = poem_data

    while failures and retries < max_retries:
        print(f"Retry attempt {retries + 1}")
        successes = []
        current_failures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {
                executor.submit(process_entry, entry, {'User-Agent': ua.random}): entry
                for entry in failures
            }
            for future in tqdm(as_completed(future_to_entry),
                               total=len(future_to_entry)):
                result = future.result()
                if result:
                    successes.append(result)
                else:
                    current_failures.append(future_to_entry[future])

        save_to_jsonl(success_file, successes)

        failures = current_failures
        retries += 1
        time.sleep(.5)  # to avoid hitting the server too frequently

    if failures:
        print(f"Failed to process {len(failures)} entries after {max_retries} retries.")

    return failures


if __name__ == '__main__':
    input_file = 'poemAnalysis_4718.json'
    success_file = 'poemAnalysis_success.jsonl'

    # load existing successes
    try:
        processed_poem_data = load_from_jsonl(success_file)
    except FileNotFoundError:
        processed_poem_data = []

    # load the initial data
    poem_data = json.load(open(input_file, 'r'))
    all_urls = {entry['url'] for entry in poem_data}
    processed_urls = {entry['url'] for entry in processed_poem_data}
    remaining_entries = [entry for entry in poem_data if
                         entry['url'] not in processed_urls]

    # process the remaining entries
    if remaining_entries:
        failed_urls = process_poem_data(remaining_entries, success_file)
        print(f"Number of URLs not successfully scraped: {len(failed_urls)}")

    # report the number of URLs not successfully scraped
    total_unsuccessful = len(all_urls - processed_urls)
    print(f"Total number of URLs not successfully scraped: {total_unsuccessful}")
