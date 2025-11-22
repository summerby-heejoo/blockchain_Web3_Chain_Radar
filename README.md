# 🏂 Web3 Chain Radar  
### Real-Time Crypto Market Intelligence Dashboard  
#### Streamlit 기반 실시간 Web3 시장 분석 시스템

---

## 📌 프로젝트 소개

**Web3 Chain Radar**는  
Web3/블록체인 산업 리서처·데이터 애널리스트 직무(JD)에 최적화된  
**실시간 암호화폐·온체인·섹터·뉴스 분석 대시보드**입니다.

모든 기능은 실제 API 기반이며, 한국어 뉴스 요약과 키워드·클러스터링까지 제공하여  
**“실무 리서치 보고서를 자동 생성해주는 수준의 분석 도구”**를 목표로 설계되었습니다.

---

## 🚀 핵심 기능 요약

### 🧭 1) 실시간 시장 지표(Home)
- Fear & Greed Index (Alternative.me API)
- BTC / ETH / SOL 실시간 가격 (CoinGecko API)
- 가격 등락 카드(초록/빨강)
- 시장 리스크 등급(높음/중간/낮음)
- BTC 추세 판단(확장국면 / 축소국면)
- Global Market Summary  
  - 전체 시총  
  - 24h 시총 변화율  
  - BTC Dominance  
  - ETH Dominance  
- BTC Active Addresses (Blockchain.com API)  
  → 온체인 활성도 기반 시장 체력 판단

---

### 📰 2) Web3 뉴스 분석 (글로벌 + 한국어)
#### **한국어까지 완전 지원하는 실전 뉴스 분석 엔진**
- CryptoPanic (글로벌)
- Cointelegraph RSS (글로벌)
- Google News (한국어 ‘암호화폐·블록체인’ 키워드)
- 코인데스크 코리아(HTML 스크랩 기반 일부)
- 뉴스 본문 자동 크롤링(추출→정제)

#### 분석 기능
- 한국어/영어 TextRank Summarization (2~3문장 자동 요약)
- KeyBERT 기반 핵심 키워드 추출
- 뉴스 10개 단위 페이지네이션
- 뉴스 언어 필터(전체/한국어만/영어만)
- WordCloud (NanumGothic 폰트 적용 — 깨짐 해결)
- Topic Clustering  
  (TF-IDF + KMeans 기반 토픽 자동 분류)

---

### 🧩 3) Web3 섹터 분석 (코어 6개)
150개 이상의 CoinGecko Categories 데이터를  
아래 **핵심 6개 섹터로 자동 그룹화**:

| Core Sector | 포함 범위 |
|-------------|-----------|
| AI | AI, 머신러닝, 자동화 |
| Layer2 | L2, Rollup, ZK, Optimistic |
| DeFi | DEX, Lending, Yield, AMM |
| NFT | NFT, Collectibles |
| Gaming | GameFi, Metaverse |
| RWA | Real World Asset, Tokenized Asset |
| Infra/기타 | 나머지 모든 범주 |

#### 제공 기능
- 핵심 섹터 그룹별 시총 집계
- 24h 섹터 성과 (평균 변화율)
- 섹터별 Top Gainers / Losers (코인 단위, 실시간)
- CoinGecko categories 기반 실시간 분석

---

## 🛠 기술 스택

### Front-End / Dashboard
- Streamlit
- Plotly Express
- Matplotlib (WordCloud 렌더링)
- Custom HTML/CSS

### Data / NLP / ML
- TextRank Summarization (KR/EN)
- KeyBERT keyword extraction
- TF-IDF + KMeans Topic Clustering
- 한글 WordCloud (NanumGothic.ttf)

### API Layer
| API | 사용 목적 |
|-----|-----------|
| Alternative.me | Fear & Greed Index |
| CoinGecko | 가격, 섹터 시총, 시장 지표 |
| Blockchain.com | BTC Active Addresses |
| CryptoPanic | 글로벌 암호화폐 뉴스 |
| Cointelegraph | RSS 뉴스 |
| Google News | 한국어 뉴스 자동 크롤링 |
| BeautifulSoup | 본문 추출 및 정제 |

---


[def]: image.png
[def]: image-1.png
[def]: image-2.png
[def]: image-3.png
[def]: image-4.png