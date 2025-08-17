import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np


def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")


def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)) )
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    return result.iloc[-1] if return_last_only else result.dropna()
    

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:
            return float(spline(dte))

    return term_spline


def get_current_price(ticker):
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]


def compute_recommendation(ticker):
    ticker = ticker.strip().upper()
    if not ticker:
        return "No stock symbol provided."
    
    try:
        stock = yf.Ticker(ticker)
        if len(stock.options) == 0:
            return f"Error: No options found for stock symbol '{ticker}'."
    except Exception:
        return f"Error: Could not retrieve data for '{ticker}'."
    
    try:
        exp_dates = filter_dates(stock.options)
    except:
        return "Error: Not enough option data."
    
    options_chains = {exp: stock.option_chain(exp) for exp in exp_dates}
    
    try:
        underlying_price = get_current_price(stock)
        if underlying_price is None:
            return "Error: Unable to retrieve underlying stock price."
    except Exception:
        return "Error: Unable to retrieve underlying stock price."
    
    atm_iv = {}
    straddle = None 
    i = 0
    for exp_date, chain in options_chains.items():
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            continue

        call_idx = (calls['strike'] - underlying_price).abs().idxmin()
        put_idx  = (puts['strike'] - underlying_price).abs().idxmin()

        call_iv = calls.loc[call_idx, 'impliedVolatility']
        put_iv  = puts.loc[put_idx, 'impliedVolatility']
        atm_iv[exp_date] = (call_iv + put_iv) / 2.0

        if i == 0:
            call_mid = (calls.loc[call_idx, 'bid'] + calls.loc[call_idx, 'ask']) / 2.0
            put_mid  = (puts.loc[put_idx, 'bid'] + puts.loc[put_idx, 'ask']) / 2.0
            straddle = call_mid + put_mid
        i += 1
    
    if not atm_iv:
        return "Error: Could not determine ATM IV for any expiration dates."
    
    today = datetime.today().date()
    dtes = []
    ivs = []
    for exp_date, iv in atm_iv.items():
        exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
        dtes.append((exp_date_obj - today).days)
        ivs.append(iv)
    
    term_spline = build_term_structure(dtes, ivs)
    ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])

    price_history = stock.history(period='3mo')
    iv30_rv30 = term_spline(30) / yang_zhang(price_history)
    avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]

    expected_move = f"{round(straddle / underlying_price * 100, 2)}%" if straddle else None

    return {
        'avg_volume': avg_volume >= 1_500_000,
        'iv30_rv30': iv30_rv30 >= 1.25,
        'ts_slope_0_45': ts_slope_0_45 <= -0.00406,
        'expected_move': expected_move
    }


# ---------- STREAMLIT APP ----------
st.title("📊 Earnings Play Checker")

stock = st.text_input("Enter Stock Symbol:", "")

if st.button("Submit"):
    with st.spinner("Fetching data..."):
        result = compute_recommendation(stock)

    if isinstance(result, str):
        st.error(result)
    else:
        avg_volume_bool = result['avg_volume']
        iv30_rv30_bool = result['iv30_rv30']
        ts_slope_bool = result['ts_slope_0_45']
        expected_move = result['expected_move']

        if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
            st.success("✅ Recommended")
        elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
            st.warning("⚠️ Consider")
        else:
            st.error("🚫 Avoid")

        st.write(f"**avg_volume:** {'PASS' if avg_volume_bool else 'FAIL'}")
        st.write(f"**iv30_rv30:** {'PASS' if iv30_rv30_bool else 'FAIL'}")
        st.write(f"**ts_slope_0_45:** {'PASS' if ts_slope_bool else 'FAIL'}")
        st.info(f"Expected Move: {expected_move}")
