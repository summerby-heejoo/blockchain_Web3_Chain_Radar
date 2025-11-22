import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

# KeyBERT (키워드 추출) – 설치 안 돼 있으면 자동으로 fallback 되도록 처리
try:
    from keybert import KeyBERT
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
except Exception:
    kw_model = None



# ===============================================
# Streamlit Basic Setup
# ===============================================
st.set_page_config(
    page_title="Web3 Chain Radar",
    page_icon="🏂",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
}
[data-testid="stMetric"] {
    background-color: #2b2b2b;
    padding: 14px;
    border-radius: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===============================================
# 숫자형 Metric (가격/변동률)
# ===============================================

def colored_metric(label, price, change):
    color = "green" if change >= 0 else "red"
    arrow = "▲" if change >= 0 else "▼"

    st.markdown(
        f"""
        <div style='background-color:#2b2b2b;
                    padding:12px;
                    border-radius:8px;
                    text-align:center;
                    color:white;'>
            <div style='font-size:18px;'>{label}</div>
            <div style='font-size:22px; font-weight:bold;'>${price:,}</div>
            <div style='font-size:18px; color:{color};'>
                {arrow} {abs(change)}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================================
# 상태용 Metric (높음/중간/낮음 + 확장/축소)
# green / yellow / red 자동 색상 적용
# ===============================================
def colored_status(label, value):
    # 상태에 따른 색상 지정
    if value in ["높음", "확장 국면"]:
        color = "limegreen"
    elif value in ["중간"]:
        color = "gold"
    elif value in ["낮음", "축소 국면"]:
        color = "red"
    else:
        color = "white"

    st.markdown(
        f"""
        <div style='background-color:#2b2b2b;
                    padding:12px;
                    border-radius:8px;
                    text-align:center;
                    color:white;'>
            <div style='font-size:18px;'>{label}</div>
            <div style='font-size:22px; font-weight:bold; color:{color};'>
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================================
# Global Market Summary 디자인
# ===============================================
def custom_metric(label, value, change=None):
    """
    - value: 숫자/문자 그대로 표시
    - change: +% or -%
    색상은 변동률 기준으로 자동 결정
    """

    if change is None:
        color = "white"
        arrow = ""
        change_text = ""
    else:
        if change >= 0:
            color = "limegreen"
            arrow = "▲"
        else:
            color = "red"
            arrow = "▼"

        change_text = f"<div style='font-size:18px; color:{color};'>{arrow} {abs(change):.2f}%</div>"

    st.markdown(
        f"""
        <div style='background-color:#2b2b2b;
                    padding:12px;
                    border-radius:8px;
                    text-align:center;
                    color:white;'>
            <div style='font-size:18px;'>{label}</div>
            <div style='font-size:22px; font-weight:bold;'>{value}</div>
            {change_text}
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================================
# fear&greed 디자인
# ===============================================
def fear_greed_card(score, diff):
    # 색상 규칙
    if score >= 70:
        color = "limegreen"
    elif score >= 40:
        color = "gold"
    else:
        color = "red"

    arrow = "▲" if diff >= 0 else "▼"

    st.markdown(
        f"""
        <div style='background-color:#2b2b2b;
                    padding:12px;
                    border-radius:8px;
                    text-align:center;
                    color:white;'>
            <div style='font-size:18px;'>Fear & Greed Index</div>
            <div style='font-size:22px; font-weight:bold; color:{color};'>{score}</div>
            <div style='font-size:18px; color:{color};'>
                {arrow} {abs(diff)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ===============================================
# 한국어 폰트
# ===============================================

def generate_wordcloud(text):
    # 한글 폰트 경로 (너가 업로드한 NanumGothic.ttf)
    font_path = "fonts/NanumGothic.ttf"
    if not os.path.exists(font_path):
        font_path = None  # 폰트 없으면 fallback

    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        font_path=font_path,
        colormap="cool"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig



# ===============================================
# Fear & Greed Proxy API (안정적, 차단 없음)
# ===============================================
@st.cache_data(ttl=3600)
def load_fear_greed_api():
    url = "https://api.alternative.me/fng/?limit=2&format=json"  # 최근 2일 데이터 불러오기

    try:
        r = requests.get(url, timeout=5)
        data = r.json()

        today = data["data"][0]          # 오늘 데이터
        yesterday = data["data"][1]      # 전일 데이터

        now_score = int(today["value"])
        prev_score = int(yesterday["value"])
        diff = now_score - prev_score
        rating = today["value_classification"]

        # timestamp → 날짜 변환
        ts = int(today["timestamp"])
        today_date = datetime.fromtimestamp(ts)

        # 히스토리용 데이터프레임 생성 (2일 이상 확장하려면 limit=30으로 바꾸면 됨)
        hist = pd.DataFrame([
            {
                "date": datetime.fromtimestamp(int(item["timestamp"])),
                "score": int(item["value"])
            }
            for item in data["data"]
        ])

        return {
            "score": now_score,
            "rating": rating,
            "diff": diff,
            "hist": hist.sort_values("date")
        }

    except Exception as e:
        st.error(f"Fear & Greed Proxy API 오류: {e}")
        return {
            "score": 50,
            "rating": "Neutral",
            "diff": 0,
            "hist": pd.DataFrame({
                "date": pd.date_range(end=pd.Timestamp.today(), periods=30),
                "score": np.random.randint(40, 60, 30),
            })
        }


# ===============================================
# BTC Active Addresses (실시간 데이터)
# - 무료 API: Blockchain.com Charts
# - 의미: 최근 30일 동안 실제 사용된 BTC 주소 수
# - 용도: 네트워크 활성도 / 시장 강도 판단
# ===============================================
@st.cache_data(ttl=300)  # 5분 캐시
def load_btc_active_addresses():
    """
    Blockchain.com Charts API를 이용하여
    최근 30일 동안의 Bitcoin 활성 주소(active addresses) 데이터를 불러온다.

    반환되는 데이터:
        - date: 날짜(datetime)
        - active_addresses: 활성 주소 수(int)
    """
    url = "https://api.blockchain.info/charts/n-unique-addresses?timespan=30days&format=json"

    try:
        # API 호출
        r = requests.get(url, timeout=5)
        js = r.json()

        # 데이터프레임 변환
        df = pd.DataFrame(js["values"])
        df["date"] = df["x"].apply(lambda t: datetime.fromtimestamp(t))
        df = df.rename(columns={"y": "active_addresses"})

        return df[["date", "active_addresses"]]

    except Exception as e:
        # 오류 시 더미 데이터 반환 (서비스 지속성 확보)
        st.error(f"BTC Active Addresses API 오류 발생: {e}")
        return pd.DataFrame({
            "date": pd.date_range(end=pd.Timestamp.today(), periods=30),
            "active_addresses": np.random.randint(700000, 900000, 30)
        })


# ===============================================
# CoinGecko 실시간 가격 API
# ===============================================
@st.cache_data(ttl=60)
def load_prices_multi(coin_list):
    """
    coin_list 형식:
    [
        {"id": "bitcoin", "symbol": "BTC"},
        {"id": "ethereum", "symbol": "ETH"},
        {"id": "solana", "symbol": "SOL"},
    ]
    """

    ids = ",".join([c["id"] for c in coin_list])

    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={ids}&vs_currencies=usd&include_24hr_change=true"
    )

    r = requests.get(url, timeout=5)
    data = r.json()

    output = {}
    for c in coin_list:
        cid = c["id"]
        symbol = c["symbol"]

        if cid in data:
            output[symbol] = {
                "price": data[cid]["usd"],
                "change": round(data[cid]["usd_24h_change"], 2)
            }

    return output

# ===============================================
# 글로벌 마켓 요약 (CoinGecko Global API)
# ===============================================
@st.cache_data(ttl=300)
def load_global_market():
    url = "https://api.coingecko.com/api/v3/global"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()["data"]

        return {
            "total_mcap": data["total_market_cap"].get("usd", 0),
            "mcap_change_24h": data.get("market_cap_change_percentage_24h_usd", 0),
            "btc_dominance": data["market_cap_percentage"].get("btc", 0),
            "active_coins": data.get("active_cryptocurrencies", 0)
        }
    except Exception as e:
        st.error(f"Global Market API 오류: {e}")
        return {
            "total_mcap": 0,
            "mcap_change_24h": 0,
            "btc_dominance": 0,
            "active_coins": 0
        }

# ===============================================
# Web3 섹터 시총 데이터 (실시간: CoinGecko Categories API)
# ===============================================
# ===============================================
# Web3 섹터 시총 데이터 (실시간: CoinGecko Categories API)
#  - 원시 카테고리 → 핵심 6개 섹터로 분류
# ===============================================
def _classify_core_sector(name: str) -> str:
    n = name.lower()

    if "ai" in n or "artificial intelligence" in n:
        return "AI"
    if "layer 2" in n or "layer-2" in n or "l2" in n or "rollup" in n:
        return "Layer2"
    if "defi" in n or "dex" in n or "yield" in n or "lending" in n or "amm" in n:
        return "DeFi"
    if "nft" in n or "collectible" in n:
        return "NFT"
    if "gaming" in n or "gamefi" in n or "metaverse" in n:
        return "Gaming"
    if "real world" in n or "rwa" in n or "tokenized" in n:
        return "RWA"
    return "Infra/기타"


@st.cache_data(ttl=300)
def load_sectors_realtime():
    url = "https://api.coingecko.com/api/v3/coins/categories"

    try:
        r = requests.get(url, timeout=5)
        data = r.json()

        sectors = []
        for d in data:
            name = d.get("name", "Unknown")
            mc = d.get("market_cap", 0)
            mc_chg = d.get("market_cap_change_24h", 0)
            category_id = d.get("id", "")

            sectors.append({
                "category_id": category_id,
                "sector": name,
                "market_cap": mc,
                "market_cap_change_24h": mc_chg,
                "core_sector": _classify_core_sector(name)
            })

        df = pd.DataFrame(sectors)
        return df

    except Exception as e:
        st.error(f"Sectors API 오류: {e}")
        return pd.DataFrame(columns=["category_id", "sector", "market_cap", "market_cap_change_24h", "core_sector"])



# ===============================================
# 섹터별 Top 상승/하락 프로젝트
# category_id 기준으로 조회
# ===============================================
@st.cache_data(ttl=300)
def load_sector_top_movers(category, top=10):
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        f"?vs_currency=usd&category={category}&order=market_cap_desc"
        "&price_change_percentage=24h&per_page=100&page=1"
    )

    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        df = pd.DataFrame(data)

        df = df[["name", "symbol", "current_price", "price_change_percentage_24h"]]

        top_gainers = df.sort_values("price_change_percentage_24h", ascending=False).head(top)
        top_losers = df.sort_values("price_change_percentage_24h").head(top)

        return top_gainers, top_losers

    except:
        return pd.DataFrame(), pd.DataFrame()


# ===============================================
# NEWS FETCH — (1) Google News (KR, crypto) + (2) Cointelegraph RSS
# ===============================================
# ===============================================
# NEWS FETCH — CryptoPanic + Cointelegraph + 한국어 뉴스 혼합
#  - 글로벌: CryptoPanic, Cointelegraph
#  - 한국어: Google News(암호화폐/블록체인 검색)
# ===============================================
@st.cache_data(ttl=1800)
def load_news_all():

    news_items = []

    # -------- 1) CryptoPanic API (글로벌, 영어) --------
    try:
        res = requests.get("https://cryptopanic.com/api/v1/posts/?auth_token=&public=true", timeout=5)
        js = res.json()
        for item in js.get("results", []):
            news_items.append({
                "title": item["title"],
                "source": item["source"]["title"],
                "summary_raw": item.get("description", item["title"]),
                "lang": "en"
            })
    except:
        pass

    # -------- 2) Cointelegraph RSS (글로벌, 영어) --------
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss")
        for entry in feed.entries[:10]:
            news_items.append({
                "title": entry.title,
                "source": "Cointelegraph",
                "summary_raw": BeautifulSoup(entry.summary, "html.parser").text,
                "lang": "en"
            })
    except:
        pass

    # -------- 3) Google News RSS (한국어, '암호화폐 OR 비트코인 OR 블록체인') --------
    # -------- 한국어 Google News (본문 포함) --------
    kr_feed_url = (
        "https://news.google.com/rss/search?"
        "q=암호화폐+OR+비트코인+OR+블록체인&hl=ko&gl=KR&ceid=KR:ko"
    )
    feed_kr = feedparser.parse(kr_feed_url)

    for entry in feed_kr.entries[:40]:
        url = entry.link.replace("./articles/", "https://news.google.com/articles/")
        body = extract_article_body(url)

        news_items.append({
            "title": entry.title,
            "source": "Google News KR",
            "summary_raw": body if len(body) > 100 else entry.title,  # 본문 우선
            "url": url,
            "lang": "ko"
        })

    # -------- 4) (옵션) 코인데스크 한국어 HTML 스크래핑 — 구조 바뀌면 깨질 수 있음 --------
    try:
        r = requests.get("https://www.coindesk.com/ko", timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        # 메인 기사 카드 기준으로 제목 일부 긁기 (필요시 직접 class 수정하면 됨)
        for h in soup.find_all("h3")[:15]:
            title = h.get_text(strip=True)
            if not title:
                continue
            news_items.append({
                "title": title,
                "source": "코인데스크 코리아(스크랩)",
                "summary_raw": title,
                "lang": "ko"
            })
    except:
        pass

    df = pd.DataFrame(news_items)
    if df.empty:
        return pd.DataFrame(columns=["title", "source", "summary_raw", "lang"])
    return df


# ===============================================
# 요약 함수 (KR/EN 모두 사용 가능 – 핵심 문장 2~3개 추출)
# ===============================================

def summarize(text, max_sentences=3, max_chars=400):
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ").strip()
    if len(text) == 0:
        return ""

    # 너무 짧으면 그대로
    if len(text) <= 120:
        return text

    # 문장 단위로 잘라보기 (영어/한국어 혼합 고려)
    # . ? ! 기준 split
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    # 문장이 너무 적으면 앞부분만 자른다
    if len(sentences) <= 1:
        return (text[:max_chars] + "...") if len(text) > max_chars else text

    # 길이 기준 상위 문장들 뽑아서 요약 (간단한 heuristic)
    ranked = sorted(
        sentences,
        key=lambda s: len(s),
        reverse=True
    )

    picked = ranked[:max_sentences]
    summary = " ".join(picked)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "..."
    return summary

# ===============================================
# 뉴스 본문 넘기기 (문장 단위 그래프 랭킹)
# ===============================================

def extract_article_body(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")

        # 뉴스 사이트 공통 패턴
        selectors = [
            "article", 
            ".article-body",
            ".article-content",
            ".content",
            "#article",
            ".post-content"
        ]

        for sel in selectors:
            body = soup.select_one(sel)
            if body:
                text = body.get_text(" ", strip=True)
                if len(text) > 150:  # 본문 최소 길이
                    return text

        # fallback: 문서 전체
        return " ".join([p.get_text(strip=True) for p in soup.find_all("p")])[:2000]

    except:
        return ""


# ===============================================
# 한국어 TextRank 요약 (문장 단위 그래프 랭킹)
# ===============================================
def textrank_summarize(text, max_sent=3):
    text = text.replace("\n", " ").strip()
    if len(text) < 40:   # 너무 짧으면 그냥 반환
        return text

    # 1) 문장 단위로 분리 (한국어라 대충 . ? ! 와 '다.' 기준)
    #    완벽하진 않지만 실무용으론 충분
    import re
    # 먼저 마침표 기준으로 자르고, 너무 짧은 조각은 버림
    raw_sents = re.split(r'(?<=[\.!?])\s+', text)
    sents = [s.strip() for s in raw_sents if len(s.strip()) > 10]

    if len(sents) <= max_sent:
        return " ".join(sents)

    # 2) TF-IDF로 문장 벡터화
    vectorizer = TfidfVectorizer(stop_words="english")  # 한/영 혼용이라 그냥 english stopword만
    X = vectorizer.fit_transform(sents)

    # 3) 문장 간 코사인 유사도 → 그래프 (TextRank 기본 구조)
    sim_matrix = cosine_similarity(X, X)

    # 4) TextRank 반복 (PageRank 유사)
    n = sim_matrix.shape[0]
    scores = np.ones(n) / n
    d = 0.85  # damping factor

    for _ in range(20):
        scores = (1 - d) / n + d * sim_matrix.dot(scores) / (sim_matrix.sum(axis=1) + 1e-8)

    # 5) 상위 점수 문장 max_sent개 선택 (원래 순서 유지)
    ranked_idx = np.argsort(scores)[::-1][:max_sent]
    ranked_idx = sorted(ranked_idx)  # 원래 등장 순서

    selected = [sents[i] for i in ranked_idx]
    summary = " ".join(selected)

    return summary


# ===============================================
# KeyBERT 기반 키워드 추출 (fallback 포함)
# ===============================================
def extract_keywords(text, top_k=5):
    text = text.replace("\n", " ").strip()
    if len(text) < 20:
        return []

    # 1) KeyBERT가 사용 가능하면 그걸로
    if kw_model is not None:
        try:
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_k
            )
            return [k[0] for k in keywords]
        except Exception:
            pass

    # 2) 실패하면 TF-IDF 기반 간이 키워드
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words="english"
    )
    X = vectorizer.fit_transform([text])
    scores = X.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    idx = np.argsort(scores)[::-1][:top_k]
    return [terms[i] for i in idx]



# ===============================================
# Topic Clustering (뉴스 토픽 클러스터링)
# ===============================================
def topic_clustering(df, n_clusters=5):
    if df.empty:
        df["topic"] = []
        return df

    texts = df["summary_raw"].fillna("").tolist()
    if len(texts) < 3:
        df["topic"] = 0
        return df

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(texts)

    # 데이터 수보다 클러스터 수가 많지 않도록 조정
    k = min(n_clusters, max(1, len(df) // 3))
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(X)

    df["topic"] = labels
    return df


# ===============================================
# Global Market Summary (시총 / 도미넌스 / 거래량)
# CoinGecko Free API
# ===============================================
@st.cache_data(ttl=300)
def load_global_market():
    url = "https://api.coingecko.com/api/v3/global"

    try:
        r = requests.get(url, timeout=5)
        data = r.json()["data"]

        return {
            "market_cap": data["total_market_cap"]["usd"],
            "volume_24h": data["total_volume"]["usd"],
            "btc_dominance": data["market_cap_percentage"]["btc"],
            "eth_dominance": data["market_cap_percentage"]["eth"],
            "market_cap_change_24h": data.get("market_cap_change_percentage_24h_usd", 0)
        }

    except Exception as e:
        st.error(f"Global Market API 오류: {e}")
        return {
            "market_cap": 0,
            "volume_24h": 0,
            "btc_dominance": 0,
            "eth_dominance": 0,
            "market_cap_change_24h": 0
        }


# ===============================================
# Navigation
# ===============================================
page = st.sidebar.radio(
    "Navigation",
    ["📌 Home", "📰 News", "🧩 Sectors"]
)


# ===============================================
# PAGE 1 — HOME
# ===============================================
if page == "📌 Home":

    st.title("📊 Web3 Chain Radar Dashboard")

    # 데이터 불러오기
    fg = load_fear_greed_api()
    global_mkt = load_global_market()

    # 실시간 가격
    coin_list = [
        {"id": "bitcoin", "symbol": "BTC"},
        {"id": "ethereum", "symbol": "ETH"},
        {"id": "solana", "symbol": "SOL"},
    ]

    prices = load_prices_multi(coin_list)

    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        fear_greed_card(fg["score"], fg["diff"])

    with col2:
        colored_metric("BTC", prices["BTC"]["price"], prices["BTC"]["change"])

    with col3:
        colored_metric("ETH", prices["ETH"]["price"], prices["ETH"]["change"])

    with col4:
        colored_metric("SOL", prices["SOL"]["price"], prices["SOL"]["change"])

# ===============================================
# Global Market Summary (실시간)
# ===============================================
    
    st.subheader("🌍 Global Market Summary")

    gm = load_global_market()

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        custom_metric(
            "전체 암호화폐 시총 (USD)",
            f"{gm['market_cap']:,.0f}",
            gm["market_cap_change_24h"]
        )

    with g2:
        custom_metric(
            "24h 거래량 (USD)",
            f"{gm['volume_24h']:,.0f}"
        )

    with g3:
        colored_status("BTC Dominance", f"{gm['btc_dominance']:.2f}%")

    with g4:
        colored_status("ETH Dominance", f"{gm['eth_dominance']:.2f}%")


    # ======== 3 COLUMN LAYOUT ========
    left, center, right = st.columns([2, 5, 2])

    # LEFT ---------------------------
    with left:
        st.subheader("📌 시장 요약 메모")
        st.write(
            "- Fear & Greed 지수: 시장 심리\n"
            "- BTC/ETH/SOL: 단기 가격 모니터링\n"
            "- 글로벌 시총 / Dominance: 자금 흐름 체크\n"
        )   

    # CENTER ---------------------------
    with center:
        st.subheader("📈 BTC Active Addresses (30일 실데이터)")
        btc_active = load_btc_active_addresses()
        st.plotly_chart(
            px.line(btc_active, x="date", y="active_addresses", height=300),
            use_container_width=True
        )
      

    # RIGHT ---------------------------
    with right:
        st.subheader("📉 리스크 분석")

        # Fear & Greed 기반 시장 리스크 등급
        score = fg["score"]
        risk = "높음" if score > 70 else "중간" if score > 40 else "낮음"

        colored_status("시장 리스크", risk)


        # BTC 추세 판단
        trend = "확장 국면" if prices["BTC"]["change"] > 0 else "축소 국면"
        colored_status("BTC 추세", trend)




# ===============================================
# PAGE 2 — NEWS (감성 제거 + 10개씩 페이지)
# ===============================================
elif page == "📰 News":

    st.title("📰 Web3 뉴스 분석 (글로벌 + 한국어)")

    df = load_news_all()

    if df.empty:
        st.warning("불러온 뉴스가 없습니다. 잠시 후 다시 시도해 주세요.")
    else:
        # TextRank 요약 + KeyBERT 키워드 생성
        df["summary"] = df["summary_raw"].apply(lambda x: textrank_summarize(x, max_sent=3))
        df["keywords"] = df["summary_raw"].apply(lambda x: extract_keywords(x, top_k=5))

        # 언어 필터
        st.subheader("🧩 필터")
        lang_opt = st.selectbox("언어", ["전체", "한국어만", "영어만"])

        df_page_base = df.copy()

        if lang_opt == "한국어만":
            df_page_base = df_page_base[df_page_base["lang"] == "ko"]
        elif lang_opt == "영어만":
            df_page_base = df_page_base[df_page_base["lang"] == "en"]

        # -------- Pagination (10개씩 출력) --------
        page_size = 10
        total_pages = (len(df_page_base) - 1) // page_size + 1

        current_page = st.number_input(
            "페이지 선택 (10개씩 표시)",
            min_value=1,
            max_value=total_pages,
            step=1
        )

        start = (current_page - 1) * page_size
        end = start + page_size

        df_page = df_page_base.iloc[start:end]   # ← 여기! df_view → df_page

        # 테이블 요약
        st.subheader("📄 뉴스 리스트")
        st.dataframe(
            df_page[["title", "source", "lang", "keywords"]],
            height=300
        )

        # 뉴스 카드 상세
        st.subheader("📰 뉴스 상세 카드")

        for _, row in df_page.iterrows():        # ← 여기! df_view → df_page
            st.markdown(f"### {row['title']}")
            st.markdown(f"**Source:** {row['source']} · **언어:** {row['lang']}")
            st.markdown(f"**키워드:** {row['keywords']}")
            st.write(row["summary"])
            st.divider()

        # WordCloud (요약 기반)
        st.subheader("☁️ 요약 기반 WordCloud")
        text_wc = " ".join(df_page["summary"].tolist())

        fig_wc = generate_wordcloud(text_wc)
        st.pyplot(fig_wc)



# ===============================================
# PAGE 3 — SECTORS
# ===============================================
elif page == "🧩 Sectors":

    st.title("🧩 Web3 섹터 분석 — 핵심 6개 그룹")

    sectors_rt = load_sectors_realtime()
    if sectors_rt.empty:
        st.warning("섹터 데이터를 불러오지 못했습니다.")
    else:
        # 핵심 섹터별 시총/변화율 집계
        core_summary = (
            sectors_rt
            .groupby("core_sector")
            .agg(
                total_mcap=("market_cap", "sum"),
                avg_mcap_chg=("market_cap_change_24h", "mean")
            )
            .reset_index()
        )

        # Infra/기타는 맨 아래로 보내기
        core_summary["sort_key"] = core_summary["core_sector"].apply(
            lambda x: 1 if x == "Infra/기타" else 0
        )
        core_summary = core_summary.sort_values(["sort_key", "core_sector"]).drop(columns=["sort_key"])

        st.subheader("📊 핵심 섹터별 시총 & 24h 변화율")
        st.dataframe(core_summary, height=300)

        # 변화율 바 차트
        st.subheader("📈 섹터별 24h 시총 변화율 (평균)")
        fig_bar = px.bar(
            core_summary,
            x="core_sector",
            y="avg_mcap_chg",
            labels={"core_sector": "섹터", "avg_mcap_chg": "24h 변화율(%)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📈 섹터별 Top Movers (코인 단위)")

        core_choices = core_summary["core_sector"].tolist()
        chosen_core = st.selectbox("분석할 섹터 선택", core_choices)

        # 선택된 core 섹터에 속한 원시 카테고리들
        subset_cats = sectors_rt[sectors_rt["core_sector"] == chosen_core]

        all_gainers = []
        all_losers = []

        for _, row in subset_cats.iterrows():
            cat_id = row["category_id"]
            top_g, top_l = load_sector_top_movers(cat_id, top=5)
            if not top_g.empty:
                top_g["category"] = row["sector"]
                all_gainers.append(top_g)
            if not top_l.empty:
                top_l["category"] = row["sector"]
                all_losers.append(top_l)

        if all_gainers:
            df_g = pd.concat(all_gainers, ignore_index=True)
            df_g = df_g.sort_values("price_change_percentage_24h", ascending=False).head(10)
            st.markdown("🔼 **상승 Top 10 코인**")
            st.dataframe(df_g, height=300)
        else:
            st.info("상승 코인 데이터를 가져오지 못했습니다.")

        if all_losers:
            df_l = pd.concat(all_losers, ignore_index=True)
            df_l = df_l.sort_values("price_change_percentage_24h", ascending=True).head(10)
            st.markdown("🔽 **하락 Top 10 코인**")
            st.dataframe(df_l, height=300)
        else:
            st.info("하락 코인 데이터를 가져오지 못했습니다.")
