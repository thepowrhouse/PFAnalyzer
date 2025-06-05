import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.title("Portfolio Technical Analysis (NSE/BSE Supported)")

# Make table and app full width
st.markdown(
    """
    <style>
        .main .block-container, .block-container {
            padding-top: 3rem;
            padding-bottom: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
            max-width: 80vw !important;
            width: 80vw !important;
        }
        .stDataFrameContainer, .stDataFrame, .stTable {
            width: 80vw !important;
            min-width: 80vw !important;
            max-width: 80vw !important;
        }
        section.main > div { 
            max-width: 80vw !important;
        }
        .css-1v0mbdj, .css-1wrcr25 {
            width: 80vw !important;
            min-width: 80vw !important;
            max-width: 80vw !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

if uploaded_file is not None:
    df_portfolio = pd.read_csv(uploaded_file)
    if 'Ticker' not in df_portfolio.columns or 'Exchange' not in df_portfolio.columns:
        st.error("CSV must have 'Ticker' and 'Exchange' columns (Exchange should be 'NSE' or 'BSE').")
    else:
        results = []
        for _, row in df_portfolio.iterrows():
            ticker = str(row['Ticker']).strip()
            exchange = str(row['Exchange']).strip().upper()
            if exchange == 'NSE':
                yf_ticker = f"{ticker}.NS"
            elif exchange == 'BSE':
                yf_ticker = f"{ticker}.BO"
            else:
                results.append({
                    'Ticker': ticker,
                    'Name': 'Error',
                    'RSI': 'Error',
                    'MACD': 'Error',
                    'MACD Signal': 'Error',
                    'OBV Momentum': 'Error',
                    'MA50': 'Error',
                    'MA200': 'Error',
                    'Current Price': 'Error',
                    'Support (20d)': 'Error',
                    'Resistance (20d)': 'Error',
                    'Next Resistance (1y)': 'Error',
                    'PE Ratio': 'Error',
                    'Recommendation': f"Unknown exchange: {exchange}"
                })
                continue

            try:
                # Download 1 year for indicators, 5 years for next resistance
                data = yf.download(yf_ticker, period='1y')
                data_5y = yf.download(yf_ticker, period='5y')

                if data.empty or data_5y.empty:
                    results.append({
                        'Ticker': ticker,
                        'Name': 'No Data',
                        'RSI': 'No Data',
                        'MACD': 'No Data',
                        'MACD Signal': 'No Data',
                        'OBV Momentum': 'No Data',
                        'MA50': 'No Data',
                        'MA200': 'No Data',
                        'Current Price': 'No Data',
                        'Support (20d)': 'No Data',
                        'Resistance (20d)': 'No Data',
                        'Next Resistance (5y)': 'No Data',
                        'PE Ratio': 'No Data',
                        'Recommendation': 'No Data'
                    })
                    continue

                close = data['Close'].squeeze()
                volume = data['Volume'].squeeze()
                close_5y = data_5y['Close'].squeeze()

                # Indicator calculations
                rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
                macd_obj = ta.trend.MACD(close)
                macd = macd_obj.macd().iloc[-1]
                macd_signal = macd_obj.macd_signal().iloc[-1]
                obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
                obv_momentum = obv.iloc[-1] - obv.iloc[-6] if len(obv) > 5 else float('nan')
                ma50 = close.rolling(window=50).mean().iloc[-1]
                ma200 = close.rolling(window=200).mean().iloc[-1]
                last_close = close.iloc[-1]

                # Get full name and PE ratio
                info = yf.Ticker(yf_ticker).info
                if exchange.upper() == "BSE":
                    name = info.get('longName') or info.get('shortName')
                else:
                    name = info.get("longName", "")
                pe_ratio = info.get('trailingPE', float('nan'))

                # Support/Resistance logic (20d)
                last_20 = close[-20:] if len(close) >= 20 else close
                support = last_20.min() if len(last_20) > 0 else float('nan')
                resistance = last_20.max() if len(last_20) > 0 else float('nan')

                # Next resistance above current resistance (from 5y data)
                next_resist_candidates = [x for x in set(close_5y) if x > resistance]
                next_resistance = min(next_resist_candidates) if next_resist_candidates else float('nan')

                # Recommendation logic using a scoring system
                buy_score = 0
                sell_score = 0

                if rsi < 30:
                    buy_score += 1
                elif rsi > 70:
                    sell_score += 1

                if macd > macd_signal:
                    buy_score += 1
                elif macd < macd_signal:
                    sell_score += 1

                if pd.notna(ma50):
                    if last_close > ma50:
                        buy_score += 1
                    elif last_close < ma50:
                        sell_score += 1

                if pd.notna(obv_momentum):
                    if obv_momentum > 0:
                        buy_score += 1
                    elif obv_momentum < 0:
                        sell_score += 1

                if buy_score >= 3:
                    rec = 'BUY'
                elif sell_score >= 3:
                    rec = 'SELL'
                else:
                    rec = 'HOLD'

                results.append({
                    'Ticker': ticker,
                    'Name': name,
                    'RSI': round(rsi, 2),
                    'MACD': round(macd, 2),
                    'MACD Signal': round(macd_signal, 2),
                    'OBV Momentum': round(obv_momentum, 2) if pd.notna(obv_momentum) else float('nan'),
                    'MA50': round(ma50, 2) if pd.notna(ma50) else float('nan'),
                    'MA200': round(ma200, 2) if pd.notna(ma200) else float('nan'),
                    'Current Price': round(last_close, 2),
                    'Support (20d)': round(support, 2) if pd.notna(support) else float('nan'),
                    'Resistance (20d)': round(resistance, 2) if pd.notna(resistance) else float('nan'),
                    'Next Resistance (5y)': round(next_resistance, 2) if pd.notna(next_resistance) else float('nan'),
                    'PE Ratio': round(pe_ratio, 2) if pd.notna(pe_ratio) else float('nan'),
                    'Recommendation': rec
                })
            except Exception as e:
                results.append({
                    'Ticker': ticker,
                    'Name': 'Error',
                    'RSI': 'Error',
                    'MACD': 'Error',
                    'MACD Signal': 'Error',
                    'OBV Momentum': 'Error',
                    'MA50': 'Error',
                    'MA200': 'Error',
                    'Current Price': 'Error',
                    'Support (20d)': 'Error',
                    'Resistance (20d)': 'Error',
                    'Next Resistance (1y)': 'Error',
                    'PE Ratio': 'Error',
                    'Recommendation': f'Error: {e}'
                })

    # ...after the loop...
    st.subheader("Analysis Results")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.markdown("""
    ### Rationale for Buy/Sell/Hold Recommendations
    - **BUY**: The stock is considered a buy if at least 3 of the following are positive: RSI < 30 (oversold), MACD > MACD Signal (bullish momentum), Current Price > MA50 (uptrend), OBV Momentum > 0 (increasing volume on up days).
    - **SELL**: The stock is considered a sell if at least 3 of the following are negative: RSI > 70 (overbought), MACD < MACD Signal (bearish momentum), Current Price < MA50 (downtrend), OBV Momentum < 0 (increasing volume on down days).
    - **HOLD**: If neither buy nor sell conditions are strongly met, the recommendation is to hold, indicating a neutral or uncertain technical outlook.
    
    **Note:** These recommendations are based on technical indicators and do not account for fundamental analysis or external market factors. Please use your own judgment and consult a financial advisor before making investment decisions.
    """)

else:
    st.info("Please upload a CSV file with 'Ticker' and 'Exchange' columns (Exchange should be 'NSE' or 'BSE').")