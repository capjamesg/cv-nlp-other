import os

import requests
from autodistill_gpt_text import GPTClassifier
from autodistill_setfit import SetFitTrainer
from bs4 import BeautifulSoup

from autodistill.text_classification import TextClassificationOntology

classifier = GPTClassifier(
    ontology=TextClassificationOntology(
        {
            "computer vision": "computer vision",
            "natural language processing": "natural language processing",
        }
    ),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

classifier.label("records.jsonl")

target_model = SetFitTrainer()

# train a model
target_model.train("./records.jsonl", output_dir="output", epochs=3)

# # load the trained model
trained_target_model = SetFitTrainer("output")

# trained_target_model.model.push_to_hub("output")

URL = "https://news.ycombinator.com/newest?p=4"

# get all titleline and links nested within
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

titlelines = soup.find_all("span", class_="titleline")

titles = [title.text for title in titlelines]
links = [title.find("a")["href"] for title in titlelines]

titles_and_links = list(zip(titles, links))

titles_and_links.append(
    ["NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections", ""]
)

for title, link in titles_and_links:
    pred = trained_target_model.predict(title)

    print(pred, "\t", title)
