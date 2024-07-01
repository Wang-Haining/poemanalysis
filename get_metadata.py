import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import json

base_url = 'https://poemanalysis.com/poem-explorer/'
output_file = 'poem_urls.txt'
ua = UserAgent()


def get_poem_data(page_num):
    url = f'{base_url}?_paged={page_num}'
    headers = {'User-Agent': ua.random}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    poems = []
    items = soup.select('.fwpl-result')

    for item in items:
        title_elem = item.select_one('.facet-post-title a')
        author_elem = item.select_one('.facet-poet .fwpl-term')

        if title_elem and author_elem:
            title = title_elem.text.strip()
            href = title_elem['href']
            author = author_elem.text.strip()
            poems.append({
                'title': title,
                'author': author,
                'url': href
            })

    return poems


def crawl_poem_data(total_pages):
    all_poem_data = []
    for page_num in range(1, total_pages + 1):
        poem_data = get_poem_data(page_num)
        all_poem_data.extend(poem_data)
        print(f'Crawled page {page_num} of {total_pages}')
        time.sleep(1)  # add delay to prevent being blocked by the server
    return all_poem_data


def save_data_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':

    # as of Jun 25, 2024
    total_pages = 315
    crawled_data = crawl_poem_data(total_pages)
    output_file = f'poemAnalysis_{len(crawled_data)}.json'
    save_data_to_json(crawled_data, output_file)
