import logging
from typing import List
from bs4 import BeautifulSoup
import requests

from arklex.utils.loader import Loader, CrawledURLObject, Document

logger = logging.getLogger(__name__)

class DebateLoader(Loader):
    """A loader specifically designed for debate content."""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.kialo.com/topics"
    
    def crawl_urls(self, url_objects: List[CrawledURLObject]) -> List[CrawledURLObject]:
        """Override crawl_urls to handle debate-specific content."""
        logger.info(f"Start crawling {len(url_objects)} debate URLs")
        docs = []
        
        for url_obj in url_objects:
            try:
                logger.info(f"Loading debate URL: {url_obj.url}")
                response = requests.get(url_obj.url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract debate-specific content
                text_list = []
                
                # Get the main debate topic
                topic_element = soup.find('h1', class_='topic-title')
                if topic_element:
                    text_list.append(f"Topic: {topic_element.text.strip()}")
                
                # Get arguments
                arguments = soup.find_all('div', class_='argument')
                for arg in arguments:
                    # Get argument type (pro/con)
                    arg_type = arg.get('data-type', 'unknown')
                    # Get argument content
                    content = arg.find('div', class_='content')
                    if content:
                        text_list.append(f"{arg_type.upper()}: {content.text.strip()}")
                
                # Get supporting evidence if available
                evidence = soup.find_all('div', class_='evidence')
                for ev in evidence:
                    text_list.append(f"Evidence: {ev.text.strip()}")
                
                text_output = "\n".join(text_list)
                
                # Get title
                title = url_obj.url
                title_elem = soup.find('title')
                if title_elem:
                    title = title_elem.text.strip()
                
                docs.append(
                    CrawledURLObject(
                        id=url_obj.id,
                        url=url_obj.url,
                        content=text_output,
                        metadata={
                            "title": title,
                            "source": url_obj.url,
                            "type": "debate"
                        },
                    )
                )
                
            except Exception as err:
                logger.error(f"Error crawling debate URL {url_obj.url}: {err}")
                docs.append(
                    CrawledURLObject(
                        id=url_obj.id,
                        url=url_obj.url,
                        content=None,
                        metadata={
                            "title": url_obj.url,
                            "source": url_obj.url,
                            "type": "debate"
                        },
                        is_error=True,
                        error_message=str(err),
                    )
                )
        
        return docs
    
    def get_all_urls(self, base_url: str, max_num: int) -> List[str]:
        """Override get_all_urls to handle debate-specific URL structure."""
        logger.info(f"Getting debate URLs from: {base_url}, maximum number: {max_num}")
        urls_visited = []
        base_url = base_url.split("#")[0].rstrip("/")
        urls_to_visit = [base_url]
        
        while urls_to_visit:
            if len(urls_visited) >= max_num:
                break
            current_url = urls_to_visit.pop(0)
            if current_url not in urls_visited:
                urls_visited.append(current_url)
                new_urls = self.get_outsource_urls(current_url, base_url)
                # Filter for debate-specific URLs
                new_urls = [url for url in new_urls if '/topics/' in url]
                urls_to_visit.extend(new_urls)
                urls_to_visit = list(set(urls_to_visit))
        
        logger.info(f"Debate URLs visited: {urls_visited}")
        return sorted(urls_visited[:max_num])
    
    def _check_url(self, full_url: str, base_url: str) -> bool:
        """Override _check_url to handle debate-specific URL validation."""
        # Skip non-debate URLs and file types
        kw_list = ['.pdf', '.jpg', '.png', '.docx', '.xlsx', '.pptx', '.zip', ".jpeg"]
        if (full_url.startswith(base_url) and 
            full_url and 
            not any(kw in full_url for kw in kw_list) and 
            full_url != base_url and
            '/topics/' in full_url):
            return True
        return False 