import requests
import os
from dotenv import load_dotenv
import wikipediaapi

load_dotenv()
TRIPADVISOR_API_KEY = os.environ["TRIPADVISOR_API_KEY"]


def get_data_from_tripadvisor(city: str, country: str):
    url = f"https://api.content.tripadvisor.com/api/v1/location/search?key={TRIPADVISOR_API_KEY}&searchQuery={city}%2C%20{country}&category=attractions&language=en"

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)

    return response.json().get("data")


def get_data_from_wikipedia(city: str):
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="MyProjectName (merlin@example.com)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    wiki_page = wiki_wiki.page(city)

    return wiki_page
    # return wiki_page.text


def get_data():
    pass
