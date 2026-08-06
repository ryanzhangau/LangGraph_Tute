import bs4
import requests
from langchain_core.documents import Document


def native_load_webpage(url: str) -> Document:
    res = requests.get(url)
    res.raise_for_status()
    soup = bs4.BeautifulSoup(res.text, "html.parser")

    # Isolate relevant body text elements
    text_elements = soup.find_all(["article", "p", "h1", "h2", "h3"])
    clean_text = "\n\n".join([el.get_text() for el in text_elements])

    return Document(page_content=clean_text, metadata={"source": url})