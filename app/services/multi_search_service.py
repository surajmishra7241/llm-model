import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
from urllib.parse import quote_plus
from app.config import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title: str
    content: str
    url: str
    source: str
    relevance_score: float
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None

class MultiSearchService:
    def __init__(self):
        self.session_timeout = aiohttp.ClientTimeout(total=30)
        self.enabled_sources = getattr(settings, 'SEARCH_SOURCES', ['duckduckgo', 'reddit'])
        
    async def search_all_sources(
        self,
        query: str,
        sources: List[str] = None,
        max_results_per_source: int = 5
    ) -> Dict[str, List[SearchResult]]:
        """Search across multiple sources simultaneously"""
        
        if sources is None:
            sources = self.enabled_sources
            
        # Filter to available sources only
        available_sources = ['duckduckgo', 'reddit', 'hackernews', 'wikipedia']
        sources = [s for s in sources if s in available_sources]
        
        if not sources:
            logger.warning("No valid search sources provided")
            return {}
        
        tasks = []
        
        # Create search tasks for each source
        for source in sources:
            if source == 'duckduckgo':
                tasks.append(self._search_duckduckgo(query, max_results_per_source))
            elif source == 'reddit':
                tasks.append(self._search_reddit(query, max_results_per_source))
            elif source == 'hackernews':
                tasks.append(self._search_hackernews(query, max_results_per_source))
            elif source == 'wikipedia':
                tasks.append(self._search_wikipedia(query, max_results_per_source))
        
        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and organize results
        combined_results = {}
        for i, result in enumerate(results):
            source_name = sources[i] if i < len(sources) else f"source_{i}"
            if isinstance(result, Exception):
                logger.error(f"Search failed for {source_name}: {str(result)}")
                combined_results[source_name] = []
            else:
                combined_results[source_name] = result
                
        return combined_results
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo Instant Answer API"""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                url = "https://api.duckduckgo.com/"
                params = {
                    'q': query,
                    'format': 'json',
                    'no_redirect': '1',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_duckduckgo_results(data, max_results)
                    else:
                        logger.error(f"DuckDuckGo search failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            return []
    
    async def _search_reddit(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Reddit using their JSON API"""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                headers = {'User-Agent': 'AI-Agent-Search/1.0'}
                url = f"https://www.reddit.com/search.json"
                params = {
                    'q': query,
                    'sort': 'relevance',
                    'limit': min(max_results, 25),
                    't': 'all'
                }
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_reddit_results(data, max_results)
                    else:
                        logger.error(f"Reddit search failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Reddit search error: {str(e)}")
            return []
    
    async def _search_hackernews(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Hacker News using Algolia API"""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                url = "https://hn.algolia.com/api/v1/search"
                params = {
                    'query': query,
                    'tags': 'story',
                    'hitsPerPage': min(max_results, 50)
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_hackernews_results(data, max_results)
                    else:
                        logger.error(f"Hacker News search failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Hacker News search error: {str(e)}")
            return []
    
    async def _search_wikipedia(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Wikipedia using their API"""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
                encoded_query = quote_plus(query)
                
                async with session.get(f"{url}{encoded_query}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_wikipedia_results(data, query)
                    elif response.status == 404:
                        # Try search API if direct lookup fails
                        return await self._search_wikipedia_opensearch(session, query, max_results)
                    else:
                        logger.error(f"Wikipedia search failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Wikipedia search error: {str(e)}")
            return []
    
    async def _search_wikipedia_opensearch(self, session: aiohttp.ClientSession, query: str, max_results: int) -> List[SearchResult]:
        """Fallback Wikipedia search using OpenSearch API"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': min(max_results, 10),
                'namespace': 0,
                'format': 'json'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    if len(data) >= 4:
                        titles, descriptions, urls = data[1], data[2], data[3]
                        for i, (title, desc, url) in enumerate(zip(titles, descriptions, urls)):
                            if i >= max_results:
                                break
                            results.append(SearchResult(
                                title=title,
                                content=desc or f"Wikipedia article about {title}",
                                url=url,
                                source='wikipedia',
                                relevance_score=0.7 - (i * 0.1),
                                timestamp=datetime.now(),
                                metadata={'type': 'wikipedia_article'}
                            ))
                    return results
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Wikipedia OpenSearch error: {str(e)}")
            return []
    
    def _parse_duckduckgo_results(self, data: dict, max_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo API response"""
        results = []
        
        # Parse instant answer
        if data.get('Answer'):
            results.append(SearchResult(
                title=data.get('Heading', 'DuckDuckGo Answer'),
                content=data['Answer'],
                url=data.get('AnswerURL', ''),
                source='duckduckgo',
                relevance_score=0.9,
                timestamp=datetime.now(),
                metadata={'type': 'instant_answer'}
            ))
        
        # Parse abstract
        if data.get('Abstract'):
            results.append(SearchResult(
                title=data.get('Heading', 'Abstract'),
                content=data['Abstract'],
                url=data.get('AbstractURL', ''),
                source='duckduckgo',
                relevance_score=0.8,
                timestamp=datetime.now(),
                metadata={'type': 'abstract'}
            ))
        
        # Parse related topics
        for topic in data.get('RelatedTopics', [])[:max_results]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append(SearchResult(
                    title=topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                    content=topic['Text'],
                    url=topic.get('FirstURL', ''),
                    source='duckduckgo',
                    relevance_score=0.6,
                    timestamp=datetime.now(),
                    metadata={'type': 'related_topic'}
                ))
        
        return results[:max_results]
    
    def _parse_reddit_results(self, data: dict, max_results: int) -> List[SearchResult]:
        """Parse Reddit API response"""
        results = []
        
        if 'data' in data and 'children' in data['data']:
            for i, post in enumerate(data['data']['children'][:max_results]):
                if i >= max_results:
                    break
                    
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                content = post_data.get('selftext', '') or post_data.get('url', '')
                
                results.append(SearchResult(
                    title=title,
                    content=content[:500] + '...' if len(content) > 500 else content,
                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                    source='reddit',
                    relevance_score=0.7 - (i * 0.05),
                    timestamp=datetime.fromtimestamp(post_data.get('created_utc', 0)) if post_data.get('created_utc') else datetime.now(),
                    metadata={
                        'subreddit': post_data.get('subreddit', ''),
                        'score': post_data.get('score', 0),
                        'num_comments': post_data.get('num_comments', 0)
                    }
                ))
        
        return results
    
    def _parse_hackernews_results(self, data: dict, max_results: int) -> List[SearchResult]:
        """Parse Hacker News API response"""
        results = []
        
        for i, hit in enumerate(data.get('hits', [])[:max_results]):
            if i >= max_results:
                break
                
            title = hit.get('title', '')
            url = hit.get('url', f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}")
            
            results.append(SearchResult(
                title=title,
                content=hit.get('story_text', '') or f"Hacker News discussion: {title}",
                url=url,
                source='hackernews',
                relevance_score=0.75 - (i * 0.05),
                timestamp=datetime.fromisoformat(hit.get('created_at', '').replace('Z', '+00:00')) if hit.get('created_at') else datetime.now(),
                metadata={
                    'points': hit.get('points', 0),
                    'num_comments': hit.get('num_comments', 0),
                    'author': hit.get('author', '')
                }
            ))
        
        return results
    
    def _parse_wikipedia_results(self, data: dict, query: str) -> List[SearchResult]:
        """Parse Wikipedia API response"""
        if data.get('type') == 'standard':
            return [SearchResult(
                title=data.get('title', ''),
                content=data.get('extract', ''),
                url=data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                source='wikipedia',
                relevance_score=0.8,
                timestamp=datetime.now(),
                metadata={
                    'type': 'wikipedia_summary',
                    'thumbnail': data.get('thumbnail', {}).get('source') if data.get('thumbnail') else None
                }
            )]
        return []

class SearchResultAggregator:
    def __init__(self):
        self.source_weights = {
            'duckduckgo': 0.9,
            'wikipedia': 0.8,
            'reddit': 0.7,
            'hackernews': 0.75
        }
    
    def aggregate_results(
        self,
        search_results: Dict[str, List[SearchResult]],
        max_final_results: int = 10
    ) -> List[SearchResult]:
        """Combine and rank results from multiple sources"""
        
        all_results = []
        
        # Flatten results and apply source weights
        for source, results in search_results.items():
            weight = self.source_weights.get(source, 0.5)
            for result in results:
                result.relevance_score *= weight
                all_results.append(result)
        
        # Remove duplicates based on URL similarity
        unique_results = self._deduplicate_results(all_results)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return unique_results[:max_final_results]
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL and content similarity"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            # Normalize URL for comparison
            normalized_url = self._normalize_url(result.url)
            
            if normalized_url and normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)
            elif not normalized_url:
                # If no URL, check content similarity (basic)
                content_hash = hashlib.md5(result.content[:100].encode()).hexdigest()
                if content_hash not in seen_urls:
                    seen_urls.add(content_hash)
                    unique_results.append(result)
        
        return unique_results
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication"""
        if not url:
            return ""
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            return f"{parsed.netloc}{parsed.path}".lower()
        except:
            return url.lower()
