# 标准库导入
import json
import time
import sys
import re
import os
import random
from datetime import datetime
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

def extract_publish_time(container, url):
    if container:
        time_selectors = [
            '.time', '.date', '.publish-time', '.article-time', '.news-time',
            'time', '.timestamp', '.pubtime', '.publish-date', '.news-date'
        ]

        for selector in time_selectors:
            time_tag = container.select_one(selector)
            if time_tag:
                time_text = time_tag.get_text(strip=True)
                if time_text:
                    return time_text

        text = container.get_text()

        date_patterns = [
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?(\s\d{1,2}:\d{1,2}(:\d{1,2})?)?',
            r'\d{1,2}[-/月]\d{1,2}[日]?(\s\d{1,2}:\d{1,2}(:\d{1,2})?)?',
            r'\d{1,2}:\d{1,2}(:\d{1,2})?'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

    if url:
        url_date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
        match = re.search(url_date_pattern, url)
        if match:
            return match.group(0)

    return ''

def get_news_content(url, title, max_retries=2):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    }

    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin1']

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '').lower()
            if 'charset=' in content_type:
                detected_encoding = content_type.split('charset=')[-1].strip()
                if detected_encoding:
                    encodings.insert(0, detected_encoding)

            html_text = None
            for encoding in encodings:
                try:
                    html_text = response.content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if html_text is None:
                html_text = response.text

            soup = BeautifulSoup(html_text, 'html.parser')

            content_selectors = [
                'article', '.article-content', '.news-content', '.content',
                '.article-body', '.news-text', '.article-text', '.news-detail',
                '.article', '#article', '.main-content', '.main-article'
            ]

            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    paragraphs = content_element.find_all('p')
                    if paragraphs:
                        valid_paragraphs = []
                        for p in paragraphs:
                            text = p.get_text(strip=True)
                            if text and len(text) > 15:
                                valid_paragraphs.append(text)
                            if len(valid_paragraphs) >= 3:
                                break

                        if valid_paragraphs:
                            content = ' '.join(valid_paragraphs)
                            if not is_gibberish(content):
                                return content[:500]

            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                content = meta_desc.get('content')
                if not is_gibberish(content):
                    return content[:500]

            body_text = soup.body.get_text(strip=True) if soup.body else ""
            if body_text and len(body_text) > 50 and not is_gibberish(body_text):
                sentences = body_text.split('。')
                content = '。'.join(sentences[:3]) + '。' if sentences else body_text[:500]
                return content[:500]

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)

    return f"{title}。这是一条关于豆粕期货市场的重要新闻。"

# 检查文本是否为乱码
def is_gibberish(text):
    if not text:
        return True

    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / len(text) > 0.5:
        return True

    if text.count('') > 5 or text.count('?') > len(text) * 0.2:
        return True

    if any('\u4e00' <= c <= '\u9fff' for c in text):
        if text.count('。') == 0 and text.count('，') == 0 and text.count('、') == 0:
            return True

    return False

def get_news_info(url, max_retries=3, retry_delay=2):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            if response.encoding == 'ISO-8859-1':
                response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            raw_news_list = []
            news_items = soup.select('a.news-title-font_1xS-F')

            if news_items:
                for item in news_items:
                    link = item.get('href')
                    title = item.get_text(strip=True)
                    if link:
                        publish_time = extract_publish_time(item.parent, link)

                        if not publish_time:
                            publish_time = datetime.now().strftime('%Y-%m-%d')

                        raw_news = {
                            'url': link,
                            'title': title if title else '无标题',
                            'publish_time': publish_time
                        }
                        raw_news_list.append(raw_news)
            else:
                news_containers = soup.select('.news-item, .article-item, .news-box, .news-list li, .article-list li')

                if news_containers:
                    for container in news_containers:
                        link_tag = container.find('a')
                        if link_tag:
                            href = link_tag.get('href')
                            if href and not href.startswith('javascript:') and not href.startswith('#'):
                                full_url = urljoin(url, href)
                                title = link_tag.get_text(strip=True)
                                if not title:
                                    title_tag = container.find(['h1', 'h2', 'h3', 'h4', 'h5', '.title', '.news-title'])
                                    if title_tag:
                                        title = title_tag.get_text(strip=True)

                                publish_time = extract_publish_time(container, full_url)

                                if not publish_time:
                                    publish_time = datetime.now().strftime('%Y-%m-%d')

                                raw_news = {
                                    'url': full_url,
                                    'title': title if title else '无标题',
                                    'publish_time': publish_time
                                }
                                raw_news_list.append(raw_news)

            if raw_news_list:
                return raw_news_list

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return []

def process_news_content(news_list, max_content_retries=2):
    processed_news = []
    
    for news in news_list:
        try:
            content = get_news_content(news['url'], news['title'], max_content_retries)
            if content:
                news['content'] = content
                news['id'] = len(processed_news) + 1
                news['date'] = news.get('publish_time', datetime.now().strftime('%Y-%m-%d'))
                news['source'] = '自动采集'
                news['category'] = 'market'
                news['is_featured'] = False
                news['views'] = random.randint(50, 500)
                
                processed_news.append(news)
        except Exception as e:
            continue
            
    return processed_news

def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    news_sources = [
        "https://futures.financialchina.shop/soybean-meal/",
        "https://finance.sina.com.cn/futures/"
    ]
    
    all_news = []
    
    for source in news_sources:
        news_list = get_news_info(source)
        if news_list:
            all_news.extend(news_list)
    
    if all_news:
        processed_news = process_news_content(all_news)
        
        if processed_news:
            all_news = sorted(processed_news, key=lambda x: x.get('publish_time', ''), reverse=True)
            save_to_file(all_news, 'news_info.json')
            print(f"采集完成，共获取 {len(all_news)} 条新闻")
        else:
            print("未能成功获取有效内容")
    else:
        print("未能从新闻源获取数据")

if __name__ == "__main__":
    main()