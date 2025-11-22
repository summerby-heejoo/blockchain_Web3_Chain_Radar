
import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="Web3 Chain Radar", layout="wide")

# ---------- Helper functions ----------
def fetch_coingecko_market_chart(coin_id='ethereum', days=30):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency':'usd','days': days}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code == 200:
        data = r.json()
        # price: [ [ts, price], ... ]
        df_price = pd.DataFrame(data['prices'], columns=['ts','price'])
        df_price['date'] = pd.to_datetime(df_price['ts'], unit='ms')
        # market_caps
        df_mcap = pd.DataFrame(data['market_caps'], columns=['ts','market_cap'])
        df_mcap['date'] = pd.to_datetime(df_mcap['ts'], unit='ms')
        # volumes
        df_vol = pd.DataFrame(data['total_volumes'], columns=['ts','volume'])
        df_vol['date'] = pd.to_datetime(df_vol['ts'], unit='ms')
        # merge by nearest timestamp (simpler: group by date)
        df = pd.DataFrame({'date': df_price['date'].dt.floor('D').unique()})
        df = df.merge(df_price.groupby(df_price['date'].dt.floor('D'))['price'].mean().rename('price'), left_on='date', right_index=True)
        df = df.merge(df_mcap.groupby(df_mcap['date'].dt.floor('D'))['market_cap'].mean().rename('market_cap'), left_on='date', right_index=True)
        df = df.merge(df_vol.groupby(df_vol['date'].dt.floor('D'))['volume'].mean().rename('volume'), left_on='date', right_index=True)
        return df.sort_values('date')
    else:
        return None

def safe_fetch_coingecko(coin_id, days=30):
    try:
        df = fetch_coingecko_market_chart(coin_id, days)
        if df is None or df.empty:
            raise Exception("No data")
        return df
    except Exception as e:
        # fallback: generate synthetic series
        today = pd.to_datetime('today').normalize()
        dates = [today - pd.Timedelta(days=i) for i in range(days-1, -1, -1)]
        df = pd.DataFrame({'date': dates})
        base = 100 + abs(hash(coin_id))%100
        df['price'] = [base*(1 + 0.01*((i%7)-3)) for i in range(len(dates))]
        df['market_cap'] = df['price'] * 1e7
        df['volume'] = 1e6 + (pd.Series(range(len(dates))) * 1000)
        return df

# Optional: placeholder for on-chain metrics (requires API keys)
def fetch_eth_onchain_summary(api_key=None, days=30):
    # placeholder: user can add Etherscan integration here.
    # For now, return synthetic daily tx count and avg gas (fallback)
    today = pd.to_datetime('today').normalize()
    dates = [today - pd.Timedelta(days=i) for i in range(days-1, -1, -1)]
    df = pd.DataFrame({'date': dates})
    df['tx_count'] = [2000000 + (i*1000)%200000 for i in range(len(dates))]
    df['avg_gas'] = [20 + (i%5) for i in range(len(dates))]
    return df

def fetch_news_keywords(query='Web3', limit=10):
    # lightweight "news" using CoinGecko / trending or just simulated short list
    # To avoid external API keys, we use CoinGecko's status_updates or trending if available.
    try:
        r = requests.get('https://api.coingecko.com/api/v3/search/trending', timeout=10)
        if r.status_code == 200:
            items = r.json().get('coins', [])
            lines = [f"{c['item']['name']} ({c['item']['symbol']})" for c in items][:limit]
            return lines
    except:
        pass
    # fallback sample
    return ["NFT marketplace launches", "Layer2 adoption rising", "Stablecoin regulatory news", "BTC ETF flows"]

# ---------- UI ----------
st.title("Web3 Chain Radar — 간단 포트폴리오/모니터링 대시보드")
st.write("간단한 Web3 트렌드 리서치 + 온체인·시세 지표 비교 (데모).")

# Sidebar: chain selection & window
chains = {
    'Ethereum':'ethereum',
    'Solana':'solana',
    'BNB Chain':'binancecoin',
    'Polygon':'polygon',
}
selected = st.sidebar.multiselect("비교할 체인 (최대 3개)", options=list(chains.keys()), default=['Ethereum','Solana'])
days = st.sidebar.slider("기간(일)", 7, 90, 30)

# Top: Today’s highlights (news)
st.subheader("오늘의 Web3 트렌드 요약 (간단)")
news = fetch_news_keywords()
cols = st.columns(4)
for i, n in enumerate(news[:4]):
    cols[i].metric(label=f"이슈 {i+1}", value=n)

# Middle: price/time-series comparison
st.subheader("가격·거래량·시가총액 비교")
if not selected:
    st.info("사이드바에서 비교할 체인을 선택하세요.")
else:
    # fetch data
    df_map = {}
    for ch in selected:
        coin_id = chains.get(ch)
        df_map[ch] = safe_fetch_coingecko(coin_id, days)

    # price chart
    price_df = pd.DataFrame()
    for name, df in df_map.items():
        tmp = df[['date','price']].copy()
        tmp = tmp.rename(columns={'price':name})
        if price_df.empty:
            price_df = tmp
        else:
            price_df = price_df.merge(tmp, on='date', how='outer')
    fig_price = px.line(price_df, x='date', y=[c for c in price_df.columns if c!='date'], title='Price (USD)')
    st.plotly_chart(fig_price, use_container_width=True)

    # marketcap table top 5 latest
    latest = []
    for name, df in df_map.items():
        r = df.sort_values('date').iloc[-1]
        latest.append({'chain':name, 'price':r['price'], 'market_cap':r['market_cap'], 'volume':r['volume']})
    st.write(pd.DataFrame(latest).sort_values('market_cap', ascending=False).reset_index(drop=True))

# Bottom: On-chain example metrics (synthetic or API)
st.subheader("예시: 온체인 요약 (샘플)")
eth_chain = fetch_eth_onchain_summary(days=days)
fig = px.line(eth_chain, x='date', y='tx_count', title='Daily Tx Count (sample)')
st.plotly_chart(fig, use_container_width=True)
st.write("참고: Etherscan / Solscan API 키를 추가하면 실제 온체인 지표로 대체 가능합니다.")

# Insights
st.subheader("자동 도출 인사이트 (샘플)")
insights = []
# Simple heuristic insights from price data
for name, df in df_map.items():
    first = df.iloc[0]['price']
    last = df.iloc[-1]['price']
    change = (last-first)/first*100
    insights.append(f"{name}: 최근 {days}일간 가격 변동 {change:.2f}%")
for it in insights:
    st.write("- " + it)

st.markdown("---")
st.caption("Made with ❤️ — Web3 Chain Radar. Source: CoinGecko (price & market data).")
