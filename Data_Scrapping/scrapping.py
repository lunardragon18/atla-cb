import requests
from bs4 import BeautifulSoup

import os

class Scrapper:
    def __init__(self,data_path):
        self.data_path = data_path
        self.links = self.get_character_links()


    def get_character_links(self):
        url = "https://avatar.fandom.com/wiki/Category:Characters"
        base = "https://avatar.fandom.com"

        links = []

        while url:
            responses = requests.get(url)
            soup = BeautifulSoup(responses.text, 'html.parser')

            for li in soup.select('.category-page__members a.category-page__member-link'):
                href = li.get('href')
                name = li.get('title')
                link = base + href
                if 'Category' in name or 'Netflix' in name or 'games' in name:
                    continue

                links.append({'name': name, 'url': link})

            next_page = soup.select(".category-page__pagination a.category-page__pagination-next")

            if next_page:
                url = next_page[0].get('href')

            else:
                url = None

        return links

    def scrape_full_avatar_page(self,url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ChatbotScraper/1.0)'
        }

        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.content, "html.parser")
        content_div = soup.find('div', class_='mw-parser-output')
        content = content_div.find_all('p')
        text = ""
        for tag in content:
            txt = tag.get_text(separator=' ', strip=True)
            if txt:
                text += txt + "\n"
        return text.strip()

    def make_data(self):
        os.makedirs(self.data_path, exist_ok=True)

        for link in self.links:
            name = link["name"].replace("/", "-")
            filepath = os.path.join(self.data_path, name + ".txt")
            if (os.path.exists(filepath)):
                continue

            text = self.scrape_full_avatar_page(link["url"])

            with open(filepath, "w",encoding="utf-8") as f:
                f.write(text)

