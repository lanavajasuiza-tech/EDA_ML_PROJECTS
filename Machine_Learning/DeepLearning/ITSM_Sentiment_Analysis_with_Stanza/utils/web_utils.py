from bs4 import BeautifulSoup
import urllib.request

def fetch_text_from_url(url):
    """
    Extrae texto desde una URL.
    """
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(strip=True)
