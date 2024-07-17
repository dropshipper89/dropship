import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.cluster import KMeans # type: ignore
import numpy as np
import time

def collect_data(url, class_title, class_content):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        posts = soup.find_all('div', class_=class_title)
        data = []
        for post in posts:
            title = post.find('h2', class_=class_content['title']).text.strip()
            content = post.find('p', class_=class_content['content']).text.strip()
            data.append({'title': title, 'content': content})
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while collecting data: {e}")
        return pd.DataFrame()

def collect_twitter_data():
    url = "https://x.com/explore/tabs/trending/"
    class_title = 'post'
    class_content = {'title': 'post-title', 'content': 'post-content'}
    return collect_data(url, class_title, class_content)

def collect_amazon_data():
    url = "https://www.amazon.com/s?k=trending+products&dc&ds=v1%3AJWF6tfr9p3fL%2B65uqgd0f3E8eTVd1Shd2D1RWUTouCc&crid=3FOLCJ93ZIIOS&qid=1720618391&sprefix=trending+products%2Ctodays-deals%2C187&ref=sr_ex_n_0"
    class_title = 's-result-item'
    class_content = {'title': 'a-size-mini a-spacing-none a-color-base s-line-clamp-2', 'content': 'a-price-whole'}
    return collect_data(url, class_title, class_content)

def collect_temu_data():
    url = "https://www.temu.com/ul/kuiper/un2.html?_p_rfs=1&subj=un-search-web&_p_jump_id=960&_x_vst_scene=adg&search_key=trendiga%20produkter&_x_ads_channel=bing&_x_ads_sub_channel=search&_x_ads_account=176324494&_x_ads_set=521047070&_x_ads_id=1312819265035152&_x_ads_creative_id=82051431980311&_x_ns_source=o&_x_ns_msclkid=07fb508f896919d697693618b27dfdc2&_x_ns_match_type=p&_x_ns_bid_match_type=bb&_x_ns_query=trendiga%20produkter%20att%20s%C3%A4lja&_x_ns_keyword=trendiga%20produkter&_x_ns_device=c&_x_ns_targetid=kwd-82052299847253%3Aloc-174&_x_ns_extensionid=&msclkid=07fb508f896919d697693618b27dfdc2&utm_source=bing&utm_medium=cpc&utm_campaign=0617_SEARCH_2108872802681652122&utm_term=trendiga%20produkter&utm_content=0617_SEARCH_2769150087229934389&adg_ctx=f-3dac0533"
    class_title = 's-result-item'
    class_content = {'title': 'a-size-mini a-spacing-none a-color-base s-line-clamp-2', 'content': 'a-price-whole'}
    return collect_data(url, class_title, class_content)


# Data preprocessing function
def preprocess_data(df):
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(df['text'])
    return tfidf_vectors

# Trend identification function
def identify_trends(tfidf_vectors):
    kmeans = KMeans(n_clusters=5)
    labels = kmeans.fit_predict(tfidf_vectors)
    return labels

# Niche and product identification function
def identify_niches_and_products(labels):
    niche_counts = np.zeros(5)
    product_counts = np.zeros(5)
    for label in labels:
        niche_counts[label] += 1
        product_counts[label] += 1
    return niche_counts, product_counts

# Agent update function
def update_agent():
    df_twitter = collect_twitter_data()
    df_amazon = collect_amazon_data()
    df_temu = collect_temu_data()