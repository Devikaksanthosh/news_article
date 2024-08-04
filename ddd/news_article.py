import requests
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import spacy

# Fetching the News Article
def fetch_news_article(api_key):
    url = f'https://newsapi.org/v2/everything?q=tesla&from=2024-07-04&sortBy=publishedAt&apiKey=6ea3dcbc9a3d402d8b0ddb5f14808ef1'
    response = requests.get(url)
    articles = response.json().get('articles')
    
    if articles:
        return articles[0].get('content')
    return None

# Replace with your actual API key
api_key = '36d71b4b4eb946f398cba86a3b2648d5'  # Replace with your actual API key
article = fetch_news_article(api_key)
if not article:
    print("No articles found.")
    exit()

print("Article:", article)

# Ensure all necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

def nltk_ner(text):
    words = word_tokenize(text)
    tags = pos_tag(words)
    tree = ne_chunk(tags, binary=False)
    
    entities = []
    for subtree in tree:
        if isinstance(subtree, Tree):
            entity_name = " ".join([word for word, tag in subtree.leaves()])
            entity_type = subtree.label()
            entities.append((entity_name, entity_type))
    
    return entities

nltk_entities = nltk_ner(article)
print("NLTK Entities:", nltk_entities)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def spacy_ner(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

spacy_entities = spacy_ner(article)
print("SpaCy Entities:", spacy_entities)

# Compare Results
def compare_entities(nltk_entities, spacy_entities):
    nltk_set = set(nltk_entities)
    spacy_set = set(spacy_entities)
    
    common = nltk_set.intersection(spacy_set)
    only_nltk = nltk_set.difference(spacy_set)
    only_spacy = spacy_set.difference(nltk_set)
    
    return common, only_nltk, only_spacy

common, only_nltk, only_spacy = compare_entities(nltk_entities, spacy_entities)

print("Common Entities:", common)
print("Only in NLTK:", only_nltk)
print("Only in SpaCy:", only_spacy)