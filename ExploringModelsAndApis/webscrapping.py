import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI
from dotenv import load_dotenv
import os


# OLLAMA - using local models with the OpenAI API spec
OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')
OPEN_MODEL = "llama3.2"

# OPEN AI - using closed source frontier models with the OpenAI API
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
# OPENAI_BASE_URL = "https://api.openai.com/v1" 
openai = OpenAI()
# openai = OpenAI(base_url=OPENAI_BASE_URL, api_key=api_key)
CLOSED_MODEL = "gpt-5-nano"



class Website:
    """
    A utility class to represent a Website that we have scraped
    """
    url: str
    title: str
    text: str

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irr in soup.body(["script", "style", "img", "input"]):
            irr.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

# ed = Website("https://edwarddonner.com")
# print(ed.title)
# print(ed.text)

system_prompt = "You are a snarky assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown. Wait and think if you can summarize more better, using lesser tokens."

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "The contents of this website is as follows; \
        please provide a short summary of this website in markdown. \
        If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize(url):
    website = Website(url)
    messages = messages_for(website)
    # response = openai.chat.completions.create(model=CLOSED_MODEL, messages=messages)
    response = ollama.chat.completions.create(model=OPEN_MODEL, messages=messages)
    # display(Markdown(response.choices[0].message.content))
    print(response.choices[0].message.content)

summarize("https://naines.github.io")

