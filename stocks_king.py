# datetime을 통해 현재 시각을 조회하여 최신 시점으로부터 5개년 실적 분석
# 입력창에 입력이 쉽도록 단순 '______' 을 placeholder로 개선.
# pip install fuzzywuzzy python-levenshtein 필요! 

import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from fuzzywuzzy import process, fuzz
from datetime import date, timedelta

# 페이지 설정
st.set_page_config(page_title="주식왕 스토킹", layout="wide")

# 💡 FIX: 파일 형식 변경에 더 유연하게 대응하도록 수정한 티커 다운로드 함수
@st.cache_data
def get_nasdaq_nyse_tickers():
    """NASDAQ과 NYSE의 공식 티커 목록을 다운로드하여 리스트로 반환합니다."""
    nasdaq_url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
    other_url = "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"
    try:
        # '|'로 구분된 CSV 파일을 읽어옵니다.
        nasdaq_df = pd.read_csv(nasdaq_url, sep='|', skipfooter=1, engine='python')
        other_df = pd.read_csv(other_url, sep='|', skipfooter=1, engine='python')

        # NYSE 티커만 필터링합니다 ('Exchange' 컬럼 값이 'N'인 경우).
        nyse_df = other_df[other_df['Exchange'] == 'N']
        
        # 💡 FIX: 열 이름을 하드코딩하는 대신, 첫 번째 열을 티커 심볼로 동적으로 선택합니다.
        nasdaq_tickers = nasdaq_df.iloc[:, 0]
        nyse_tickers = nyse_df.iloc[:, 0]
        
        # 두 리스트를 합치고, 중복 및 NaN 값을 제거한 후 최종 리스트를 반환합니다.
        all_tickers = pd.concat([nasdaq_tickers, nyse_tickers]).dropna().unique().tolist()
        return all_tickers
    except Exception as e:
        st.error(f"티커 목록을 다운로드하는 데 실패했습니다: {e}")
        return []

def handle_suggestion_click(suggestion):
    st.session_state.ticker_input = suggestion
    st.session_state.submit_run = True

def go_home():
    st.session_state.ticker_input = ""
    st.session_state.submit_run = False

# --- 기본 표시 화면 함수 ---
def display_default_view():
    st.subheader("주식왕 스토킹")
    st.markdown("---")
    index_tickers = {"Dow Jones": "^DJI", "S&P 500": "^GSPC", "NASDAQ": "^IXIC"}
    
    try:
        index_data = yf.download(list(index_tickers.values()), period="3d", progress=False)['Close']
        cols = st.columns(3)
        for i, (name, ticker) in enumerate(index_tickers.items()):
            if ticker in index_data.columns and len(index_data[ticker].dropna()) >= 2:
                current_price, prev_price = index_data[ticker].iloc[-1], index_data[ticker].iloc[-2]
                delta, delta_pct = current_price - prev_price, (current_price - prev_price) / prev_price * 100
                cols[i].metric(label=name, value=f"{current_price:,.2f}", delta=f"{delta:,.2f} ({delta_pct:.2f}%)")
    except Exception:
        st.warning("주요 지수 데이터를 가져오는 데 실패했습니다.")

    st.markdown("---")
    st.info("💡 아래는 실시간 급등주가 아닌, 시장을 대표하는 주요 종목들의 당일 시세입니다. (티커 클릭 시 상세 조회)")

    nasdaq_movers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA']
    nyse_movers = ['JPM', 'JNJ', 'V', 'WMT', 'BRK-B', 'UNH', 'XOM']
    
    col1, col2 = st.columns(2)
    try:
        with col1:
            st.markdown("#### NASDAQ Key Movers")
            nasdaq_data = yf.download(nasdaq_movers, period="2d", progress=False)
            if not nasdaq_data.empty:
                nasdaq_perf = ((nasdaq_data['Close'].iloc[-1] - nasdaq_data['Open'].iloc[-1]) / nasdaq_data['Open'].iloc[-1]) * 100
                nasdaq_perf_df = nasdaq_perf.reset_index(); nasdaq_perf_df.columns = ['Ticker', 'Change (%)']
                
                # 💡 FIX: 데이터프레임 대신 버튼과 텍스트로 표를 동적으로 생성
                header_cols = st.columns([2, 1]); header_cols[0].markdown("**Ticker**"); header_cols[1].markdown("**Change (%)**")
                for _, row in nasdaq_perf_df.iterrows():
                    ticker, change = row['Ticker'], row['Change (%)']
                    row_cols = st.columns([2, 1])
                    row_cols[0].button(ticker, key=f"btn_{ticker}", on_click=handle_suggestion_click, args=[ticker], use_container_width=True)
                    color = "red" if change < 0 else "green"
                    row_cols[1].markdown(f"<p style='color: {color}; text-align: right;'>{change:.2f}%</p>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### NYSE Key Movers")
            nyse_data = yf.download(nyse_movers, period="2d", progress=False)
            if not nyse_data.empty:
                nyse_perf = ((nyse_data['Close'].iloc[-1] - nyse_data['Open'].iloc[-1]) / nyse_data['Open'].iloc[-1]) * 100
                nyse_perf_df = nyse_perf.reset_index(); nyse_perf_df.columns = ['Ticker', 'Change (%)']
                
                # 💡 FIX: NYSE 목록도 동일하게 버튼으로 생성
                header_cols = st.columns([2, 1]); header_cols[0].markdown("**Ticker**"); header_cols[1].markdown("**Change (%)**")
                for _, row in nyse_perf_df.iterrows():
                    ticker, change = row['Ticker'], row['Change (%)']
                    row_cols = st.columns([2, 1])
                    row_cols[0].button(ticker, key=f"btn_{ticker}", on_click=handle_suggestion_click, args=[ticker], use_container_width=True)
                    color = "red" if change < 0 else "green"
                    row_cols[1].markdown(f"<p style='color: {color}; text-align: right;'>{change:.2f}%</p>", unsafe_allow_html=True)
    except Exception:
        st.warning("주요 종목 데이터를 가져오는 데 실패했습니다.")

# --- 사이드바 ---
with st.sidebar:
    st.title("Financial Analysis")
    if 'ticker_input' not in st.session_state:
        st.session_state.ticker_input = ''
    if 'submit_run' not in st.session_state:
        st.session_state.submit_run = False

    ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", key='ticker_input', placeholder="ex: AAPL")
    period = st.selectbox("Enter a time frame", ("1D", "5D", "1M", "6M", "YTD", "1Y", "5Y"), index=2)
    
    submit = st.button("Submit", key='submit_button')
    if submit:
        st.session_state.submit_run = True

# --- 유틸리티 함수 ---
def format_value(value):
    if value is None or not isinstance(value, (int, float)): return "N/A"
    suffixes, suffix_index = ["", "K", "M", "B", "T"], 0
    if value == 0: return "$0.0"
    while abs(value) >= 1000 and suffix_index < len(suffixes) - 1:
        value /= 1000
        suffix_index += 1
    return f"${value:.1f}{suffixes[suffix_index]}"

def safe_format(value, fmt="{:.2f}", fallback="N/A"):
    try: return fmt.format(value) if value is not None else fallback
    except (ValueError, TypeError): return fallback

def get_next_trading_day(df, date):
    after = df[df.index > date]
    return after.index[0] if not after.empty else None

def get_same_or_next_trading_day(df, date):
    if date in df.index: return date
    return get_next_trading_day(df, date)

# 티커 검색시 원래 티커가 존재하면 -> submit 로직으로 이동/ 존재하지 않을 경우 티커 추천
if st.session_state.submit_run and ticker.strip():
        try:
            with st.spinner('Fetching data...'):
                stock = yf.Ticker(ticker.upper())
                info = stock.info
                if not info or info.get('regularMarketPrice') is None:
                    st.error(f"Could not find data for ticker '{ticker.upper()}'.")
                    all_tickers = get_nasdaq_nyse_tickers()
                    if all_tickers:
                        input_len = len(ticker)
                        candiates = [t for t in all_tickers if abs(len(t)-input_len)<=1]
                        if candiates:
                            suggestion, score = process.extractOne(ticker.upper(),candiates)
                            if score > 75:
                                st.warning(f"Did you mean **{suggestion}**?")
                                st.button(f"Search for {suggestion}", on_click=handle_suggestion_click,args=[suggestion])
                else:

                    col_header, col_button = st.columns([4,1])
                    with col_header:
                       st.subheader(f"{ticker.upper()} - {info.get('longName', 'N/A')}")

                    with col_button:
                       st.button("🏠 Home", on_click=go_home, use_container_width=True)
                    period_map = {
                        "1D": ("1d", "5m"), "5D": ("5d", "30m"), "1M": ("1mo", "1d"),
                        "6M": ("6mo", "1d"), "YTD": ("ytd", "1d"), "1Y": ("1y", "1d"),
                        "5Y": ("5y", "1wk"),
                    }
                    selected_period, interval = period_map.get(period, ("1mo", "1d"))
                    history = stock.history(period=selected_period, interval=interval)
                    st.line_chart(pd.DataFrame(history["Close"]))

                    col1, col2, col3 = st.columns(3)
                    
                    stock_info_data = [
                        ("Country", info.get('country', 'N/A')), ("Sector", info.get('sector', 'N/A')),
                        ("Industry", info.get('industry', 'N/A')), ("Market Cap", format_value(info.get('marketCap'))),
                        ("Enterprise Value", format_value(info.get('enterpriseValue'))),
                        ("Employees", f"{info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else "N/A")
                    ]
                    df_stock_info = pd.DataFrame(stock_info_data, columns=["Stock Info", "Value"])
                    col1.dataframe(df_stock_info, width=400, hide_index=True)

                    price_info_data = [
                        ("Current Price", safe_format(info.get('currentPrice'), fmt="${:,.2f}")),
                        ("Previous Close", safe_format(info.get('previousClose'), fmt="${:,.2f}")),
                        ("Day High", safe_format(info.get('dayHigh'), fmt="${:,.2f}")),
                        ("Day Low", safe_format(info.get('dayLow'), fmt="${:,.2f}")),
                        ("52 Week High", safe_format(info.get('fiftyTwoWeekHigh'), fmt="${:,.2f}")),
                        ("52 Week Low", safe_format(info.get('fiftyTwoWeekLow'), fmt="${:,.2f}"))
                    ]
                    df_price_info = pd.DataFrame(price_info_data, columns=["Price Info", "Value"])
                    col2.dataframe(df_price_info, width=400, hide_index=True)

                    dividend_yield = info.get('dividendYield')
                    dividend_yield_formatted = safe_format(dividend_yield * 100, fmt="{:.2f}%") if dividend_yield else 'N/A'
                    biz_metrics_data = [
                        ("EPS (FWD)", safe_format(info.get('forwardEps'))),
                        ("P/E (FWD)", safe_format(info.get('forwardPE'))), ("PEG Ratio", safe_format(info.get('pegRatio'))),
                        ("Div Rate (FWD)", safe_format(info.get('dividendRate'), fmt="${:.2f}")),
                        ("Div Yield (FWD)", dividend_yield_formatted),
                        ("Recommendation", info.get('recommendationKey', 'N/A').capitalize())
                    ]
                    df_biz_metrics = pd.DataFrame(biz_metrics_data, columns=["Business Metrics", "Value"])
                    col3.dataframe(df_biz_metrics, width=400, hide_index=True)

                    st.header("Earnings Analysis")
                    try:
                        from datetime import date, timedelta
                        earnings = stock.get_earnings_dates(limit=24)
                        results = []
                        if earnings is None or earnings.empty:
                            st.info("No earnings data available for this stock.")
                        else:
                            today = date.today()
                            past_earnings = earnings[earnings.index.date < today]
                            if past_earnings.empty:
                                st.info("No past earnings announcements found to analyze.")
                            else:
                                latest_earnings_date = past_earnings.index.max()
                                start_date = latest_earnings_date.date() - timedelta(days=365 * 5)
                                history_long = stock.history(start=start_date, end=today)
                                earnings_to_analyze = past_earnings[past_earnings.index.date >= start_date]
                                for idx, row in earnings_to_analyze.iterrows():
                                    earnings_date = pd.to_datetime(idx).date()
                                    raw_time = row.get("Time", "")
                                    time_of_day = raw_time.lower() if isinstance(raw_time, str) else "pm"
                                    trading_day, prev_day = None, None
                                    try:
                                        if time_of_day == "am":
                                            trading_day = get_same_or_next_trading_day(history_long, idx)
                                            if trading_day:
                                                prev_days = history_long.index[history_long.index < trading_day]
                                                if len(prev_days) > 0: prev_day = prev_days[-1]
                                        else:
                                            trading_day = get_next_trading_day(history_long, idx)
                                            if trading_day:
                                                prev_days = history_long.index[history_long.index < trading_day]
                                                if len(prev_days) > 0: prev_day = prev_days[-1]
                                        if trading_day and prev_day:
                                            prev_close = history_long.loc[prev_day, "Close"]
                                            next_close = history_long.loc[trading_day, "Close"]
                                            if prev_close and next_close and prev_close != 0:
                                                pct_change = ((next_close - prev_close) / prev_close) * 100
                                                results.append({"Earnings Date": earnings_date, "Price Date": trading_day.date(), "Close % Change": f"{pct_change:.2f}%"})
                                            else:
                                                results.append({"Earnings Date": earnings_date, "Price Date": trading_day.date(), "Close % Change": "N/A"})
                                        else:
                                            results.append({"Earnings Date": earnings_date, "Price Date": "N/A", "Close % Change": "N/A"})
                                    except IndexError:
                                        results.append({"Earnings Date": earnings_date, "Price Date": "N/A", "Close % Change": "N/A"})
                                if results:
                                    df_earnings = pd.DataFrame(results)
                                    earn_col1, earn_col2 = st.columns([1, 2])
                                    with earn_col1:
                                        def format_pct_change_display(x):
                                            if pd.isnull(x) or str(x).strip().upper() in ("N/A", "", "NONE", "NAN", "____"): return "N/A"
                                            try: return f"{float(str(x).replace('%', '')):.2f}%"
                                            except (ValueError, TypeError): return "N/A"
                                        df_display = df_earnings.copy()
                                        df_display["Close % Change"] = df_display["Close % Change"].apply(format_pct_change_display)
                                        st.dataframe(df_display, width=400, height=450, hide_index=True)
                                    with earn_col2:
                                        def safe_str_to_float(x):
                                            try: return float(str(x).replace("%", "")) if str(x).strip().upper() not in ("N/A", "", "NONE", "NAN", "____") else None
                                            except (ValueError, TypeError): return None
                                        chart_data = df_earnings.copy()
                                        chart_data["Close % Change Numeric"] = chart_data["Close % Change"].apply(safe_str_to_float)
                                        chart_data.dropna(subset=["Close % Change Numeric"], inplace=True)
                                        if not chart_data.empty:
                                            chart_data["Earnings Date"] = chart_data["Earnings Date"].astype(str)
                                            chart = alt.Chart(chart_data).mark_bar().encode(x=alt.X('Earnings Date:O', title='Earnings Date', sort=None), y=alt.Y('Close % Change Numeric:Q', title='Close % Change (%)'), color=alt.condition(alt.datum['Close % Change Numeric'] > 0, alt.value('green'), alt.value('red')), tooltip=['Earnings Date', 'Price Date', alt.Tooltip('Close % Change Numeric', format='.2f')]).properties(title='Stock Price Change After Earnings')
                                            st.altair_chart(chart, use_container_width=True)
                                        else:
                                            st.info("Could not find valid data to create an earnings chart.")
                    except Exception as e:
                        st.warning(f"Could not fetch or process earnings data: {e}")
        
        except Exception as e:
            st.error(f"An unexpected error occurred : {e}")       
        finally:
            st.session_state.submit_run = False
else:
    if not ticker.strip() and st.session_state.submit_run:
        st.session_state.submit_run = False
    display_default_view()    