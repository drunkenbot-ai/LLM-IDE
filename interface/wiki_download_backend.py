from __future__ import annotations
import re
import time
from typing import Dict, List, Optional
import requests


class WikipediaDownloaderBackend:
    """Backend class for downloading Wikipedia pages"""

    def __init__(self):
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.min_request_interval = 2.0
        self.last_request_time = 0
        self.is_running = False

    def _rate_limit(self):
        """Rate limiting for Wikipedia API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        self._rate_limit()
        print(params)
        try:
            response = self.session.get(
                self.api_url,
                params=params,
                headers={'User-Agent': 'DrunkenBot-Wikipedia-GUI/1.0'}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}

    def search_pages(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for Wikipedia pages"""
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': limit
        }

        data = self._make_request(params)
        if 'error' in data:
            return []

        results = data.get('query', {}).get('search', [])
        pages = []
        for result in results:
            pages.append({
                'title': result['title'],
                'pageid': result['pageid'],
                'snippet': result.get('snippet', ''),
                'size': result.get('size', 0),
                'wordcount': result.get('wordcount', 0)
            })
        return pages

    def get_page_content(self, title: str) -> Optional[Dict]:
        """Get full page content"""
        params = {
            'action': 'parse',
            'page': title,
            'format': 'json',
            'prop': 'text|revid|categories|links',
            'formatversion': 2
        }

        data = self._make_request(params)
        if 'error' in data:
            return None

        parse_data = data.get('parse', {})
        if not parse_data:
            return None

        html_content = parse_data.get('text', '')
        plain_text = self._clean_html(html_content)

        return {
            'title': title,
            'text': plain_text,
            'revid': parse_data.get('revid', 0),
            'categories': parse_data.get('categories', []),
            'timestamp': datetime.utcnow().isoformat()
        }

    def _clean_html(self, html_content: str) -> str:
        """Extract plain text from HTML"""
        import html
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = re.sub(r'\[\d+\]', '', text)
        return text

    def sanitize_filename(self, title: str) -> str:
        """Create safe filename"""
        safe = re.sub(r'[<>:"/\\|?*]', '_', title)
        if len(safe) > 200:
            safe = safe[:200]
        return safe


# ============================================================================
# Worker Thread for Downloading
# ============================================================================

