import os
import json
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from scraper import fetch_website_links, fetch_website_contents


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
CLOSED_MODEL = "gpt-4.1-nano"



# 1. fetch summary, fetch links
# 2. llm call to have nice summary
# 3. llm call to have relevant links only

################################   PROMPTS   ##################################################################
link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""
##########################################################################################################

def get_links_user_prompt(url):
    user_prompt = f"""
    Here is the list of links on the website {url} -
    Please decide which of these are relevant web links for a brochure about the company, 
    respond with the full https URL in JSON format.
    Do not include Terms of Service, Privacy, email links.

    Links (some might be relative links):

    """
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt

# user prompt:
# print(get_links_user_prompt("https://edwarddonner.com"))

def select_relevant_links(url):
    print(f"Selecting relevant links for {url} by calling {CLOSED_MODEL}")
    response = openai.chat.completions.create(
        model=CLOSED_MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    links = json.loads(result) #convert to dictonary
    print(f"Found {len(links['links'])} relevant links")
    return links

# print(select_relevant_links("https://edwarddonner.com"))


def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url) #llm call to select relevant links

    # GET CONTENT HEADER = link['type']
    # GET content body for each link['url']
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links['links']:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result


# print(fetch_page_and_all_relevant_links("https://edwarddonner.com"))

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt

def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
    )
    result = response.choices[0].message.content
    print(result)

create_brochure("Edward Donner", "https://edwarddonner.com")
# create_brochure("HuggingFace", "https://huggingface.co")