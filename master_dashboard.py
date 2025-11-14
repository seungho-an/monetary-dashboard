import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import datetime
from fredapi import Fred  # Use FRED API
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings
import time
import os # <-- GitHub Actions에서 API 키를 가져오기 위해 import

# --- 0. Suppress Warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# --- 1. STATUS INDICATOR DEFINITIONS ---
# Define the colors for the status dots
STATUS_COLORS = {
    "red": "#FF5733",      # Very High Risk
    "yellow": "#FFC300",   # High Risk / Warning
    "grey": "#A0A0A0",     # Neutral
    "lightgreen": "#A1E8A1", # Good / Slight Caution
    "blue": "#5DADE2"      # Very Good / Stable
}

def get_status(value, thresholds):
    """Assigns a color and text based on a value and defined thresholds."""
    if value is None or pd.isna(value):
        return STATUS_COLORS['grey'], "Data N/A"
    if 'red_high' in thresholds and value >= thresholds['red_high']:
        return STATUS_COLORS['red'], "Very High Risk"
    if 'yellow_high' in thresholds and value >= thresholds['yellow_high']:
        return STATUS_COLORS['yellow'], "High Risk / Warning"
    if 'red_low' in thresholds and value <= thresholds['red_low']:
        return STATUS_COLORS['red'], "Very High Risk (Low)"
    if 'yellow_low' in thresholds and value <= thresholds['yellow_low']:
        return STATUS_COLORS['yellow'], "High Risk / Warning (Low)"
    if 'grey_low' in thresholds and value <= thresholds['grey_low']:
        return STATUS_COLORS['grey'], "Neutral"
    if 'blue_low' in thresholds and value <= thresholds['blue_low']:
        return STATUS_COLORS['blue'], "Very Good / Stable"
    if 'lightgreen_high' in thresholds and value <= thresholds['lightgreen_high']:
        return STATUS_COLORS['lightgreen'], "Good / Stable"
    
    # Default (assumes low is good if not specified)
    return STATUS_COLORS['blue'], "Very Good / Stable"

# --- 2. GLOBAL SETTINGS ---
print("--- Running Master Dashboard Script ---")
# *** MODIFICATION: Get API key from GitHub Secrets ***
fred_api_key = os.environ.get('FRED_API_KEY')
if not fred_api_key:
    print("CRITICAL ERROR: FRED_API_KEY environment variable not set.")
    # 로컬 테스트용으로 하드코딩된 키를 fallback으로 사용할 수 있습니다.
    # fred_api_key = '607b00d349d2ffaf1789f1e4846419d2' 
    # 하지만 GitHub에 올릴 때는 위 os.environ.get('FRED_API_KEY')만 사용해야 합니다.
    print("Falling back to hardcoded key for local test... DO NOT COMMIT THIS.")
    fred_api_key = '607b00d349d2ffaf1789f1e4846419d2' # 로컬 테스트용
    # exit() # GitHub Actions에서는 이 부분이 활성화되어야 합니다.


start_date_long = datetime.datetime(2000, 1, 1) # For long-term charts
start_date_short = datetime.datetime(2018, 1, 1) # For liquidity charts
end_date = datetime.datetime.now()
rolling_window_weeks_list = [13, 26, 52] 

# --- 3. UNIFIED DATA FETCH ---
print("Starting unified data fetch...")
fred = Fred(api_key=fred_api_key)

# Consolidate all tickers
fred_daily_tickers = [
    'BAA10Y', 'T10YIE', 'EFFR', 'IORB', 'IOER', 'SOFR', 
    'RRPONTSYAWARD', 'DPCREDIT', 'RRPONTSYD', 'BAMLH0A0HYM2', 
    'T10Y2Y', 'DTWEXBGS', 'NFCI'
]
fred_weekly_tickers = ['WRESBAL', 'TLAACBW027SBOG', 'IC4WSA']
fred_monthly_tickers = [
    'HOUST', 'NEWORDER', 'RRSFS'
]
# fred_quarterly_tickers = [] # CPATA was removed

yahoo_tickers = [
    '^MOVE', '^GSPC', '^IXIC', 'BTC-USD', '^DJI', '^RUT', '^VIX'
]

try:
    # --- 3a. FRED Daily Data ---
    print("Fetching FRED daily data...")
    all_daily_data = {}
    for ticker in fred_daily_tickers:
        all_daily_data[ticker] = fred.get_series(ticker, observation_start=start_date_long, observation_end=end_date)
    daily_data_raw = pd.DataFrame(all_daily_data)
    daily_data_raw = daily_data_raw.ffill() # First ffill for processing

    # --- 3b. FRED Weekly Data ---
    print("Fetching FRED weekly data...")
    all_weekly_data = {}
    for ticker in fred_weekly_tickers:
        all_weekly_data[ticker] = fred.get_series(ticker, observation_start=start_date_long, observation_end=end_date)
    weekly_data_raw = pd.DataFrame(all_weekly_data)
    weekly_data_raw = weekly_data_raw.ffill()

    # --- 3c. FRED Monthly Data ---
    print("Fetching FRED monthly data...")
    all_monthly_data = {}
    for ticker in fred_monthly_tickers:
        print(f"  Fetching monthly: {ticker}...") 
        all_monthly_data[ticker] = fred.get_series(ticker, observation_start=start_date_long, observation_end=end_date)
    monthly_data_raw = pd.DataFrame(all_monthly_data)
    monthly_data_raw = monthly_data_raw.ffill()

    # --- 3d. FRED Quarterly Data ---
    # (Removed)

    # --- 3e. Yahoo Finance Data ---
    print("Fetching Yahoo Finance data...")
    # *** MODIFICATION: Set auto_adjust=True explicitly to suppress warning ***
    asset_data_raw = yf.download(yahoo_tickers, start=start_date_long, end=end_date, auto_adjust=True)
    
    asset_prices_full = pd.DataFrame()
    
    # yfinance auto_adjust=True이면 'Close'만 반환하고, MultiIndex가 아닐 수 있습니다.
    # VIX는 'Close'를 사용해야 합니다.
    if isinstance(asset_data_raw.columns, pd.MultiIndex):
        # VIX를 위해 'Close' 열을 별도로 가져옵니다.
        vix_data = yf.download('^VIX', start=start_date_long, end=end_date, auto_adjust=False)['Close']
        asset_prices_full = asset_data_raw['Close'] # auto_adjust=True는 'Close'에 이미 수정된 값을 줍니다.
        asset_prices_full['^VIX'] = vix_data
    else:
        # 단일 티커 또는 auto_adjust=True가 기본값인 경우
        asset_prices_full = asset_data_raw
        if '^VIX' not in asset_data_raw.columns: # VIX가 'Close'가 아닐 경우 대비
             vix_data = yf.download('^VIX', start=start_date_long, end=end_date, auto_adjust=False)['Close']
             asset_prices_full['^VIX'] = vix_data

    # 'Close' 열에서 필요한 티커만 선택
    asset_prices_full = asset_prices_full[[col for col in yahoo_tickers if col in asset_prices_full.columns]]
    asset_prices_full = asset_prices_full.ffill()
    
    print("All data fetched successfully.")

except Exception as e:
    print(f"CRITICAL ERROR during data fetch: {e}")
    print("Please check API key, internet connection, and library installations.")
    exit()

# --- 4. UNIFIED DATA PROCESSING ---
print("Processing all dashboard data...")

# --- 4a. Market Risk Data (Dashboard 1) ---
df_market_risk = pd.concat([daily_data_raw[['BAA10Y', 'T10YIE']], asset_prices_full[['^MOVE', '^VIX']]], axis=1)
df_market_risk.rename(columns={'^MOVE': 'MOVE', '^VIX': 'VIX'}, inplace=True)
df_market_risk = df_market_risk.ffill().dropna(how='all')
df_market_risk['VIX_MOVE_SPREAD'] = df_market_risk['VIX'] - df_market_risk['MOVE']

# --- 4b. Liquidity Data (Dashboard 2) ---
daily_data_liq = daily_data_raw.copy()
weekly_data_liq = weekly_data_raw.copy() 

daily_data_liq['POLICY_RATE'] = daily_data_liq['IORB'].fillna(daily_data_liq['IOER'])
daily_data_liq['EFFR_Spread'] = (daily_data_liq['EFFR'] - daily_data_liq['POLICY_RATE']) * 100
daily_data_liq['SOFR_Spread'] = (daily_data_liq['SOFR'] - daily_data_liq['POLICY_RATE']) * 100
daily_data_liq['RRP_Spread'] = (daily_data_liq['RRPONTSYAWARD'] - daily_data_liq['POLICY_RATE']) * 100
daily_data_liq['DW_Spread'] = (daily_data_liq['DPCREDIT'] - daily_data_liq['POLICY_RATE']) * 100
daily_data_liq['RRPONTSYD_B'] = daily_data_liq['RRPONTSYD'] / 1000
weekly_data_liq['RESERVE_RATIO_PCT'] = (weekly_data_liq['WRESBAL'] / weekly_data_liq['TLAACBW027SBOG']) * 100

weekly_spread = daily_data_liq['EFFR_Spread'].resample('W-WED').mean()
weekly_ratio = weekly_data_liq['RESERVE_RATIO_PCT'].resample('W-WED').mean()
final_data_var = pd.concat([weekly_ratio, weekly_spread], axis=1)
final_data_var.columns = ['RESERVE_RATIO_PCT', 'SPREAD_BPS']
var_data_diff = final_data_var.dropna().diff().dropna()
y_ols = var_data_diff['SPREAD_BPS']
X_ols = sm.add_constant(var_data_diff['RESERVE_RATIO_PCT'])

elasticity_data = {}
print("  Running RollingOLS for 13, 26, 52-week elasticities...")
for window in rolling_window_weeks_list:
    rol_model = RollingOLS(y_ols, X_ols, window=window, min_nobs=window)
    rolling_results = rol_model.fit()
    elasticity_series = rolling_results.params['RESERVE_RATIO_PCT'].dropna()
    elasticity_data[f'{window}-Wk Elasticity'] = elasticity_series
elasticity_df = pd.DataFrame(elasticity_data) 

asset_norm = pd.DataFrame(index=asset_prices_full.index)
for col in ['^GSPC', '^IXIC', 'BTC-USD', '^DJI', '^RUT']:
    if col in asset_prices_full.columns:
        first_valid_price = asset_prices_full[col].dropna().iloc[0]
        if pd.notna(first_valid_price) and first_valid_price > 0:
            asset_norm[col] = (asset_prices_full[col] / first_valid_price) * 100
        else:
            asset_norm[col] = pd.NA
asset_norm = asset_norm.fillna(100)

if '^GSPC' in asset_prices_full.columns:
    sp500_prices = asset_prices_full['^GSPC'].dropna()
    rolling_high = sp500_prices.rolling(window=252, min_periods=1).max()
    drawdown = (sp500_prices - rolling_high) / rolling_high
    in_correction = (drawdown <= -0.10) & (drawdown > -0.20)
    in_decline = (drawdown <= -0.20)
    dynamic_shading_available = True
else:
    dynamic_shading_available = False

# --- 4c. Global Risk Data (Dashboard 3) ---
df_global_risk = daily_data_raw[['T10Y2Y', 'DTWEXBGS', 'NFCI']].copy()
df_global_risk = df_global_risk.ffill().dropna(how='all')

# --- 4d. Leading Indicators Data (Dashboard 4) ---
print("Processing leading indicator data...")
df_leading_risk = pd.concat([monthly_data_raw, weekly_data_raw[['IC4WSA']]], axis=1)
df_leading_risk = df_leading_risk.ffill().reindex(daily_data_raw.index).ffill()
df_leading_risk['BAMLH0A0HYM2'] = daily_data_raw['BAMLH0A0HYM2'].reindex(df_leading_risk.index, method='ffill')

houst_monthly = monthly_data_raw['HOUST'].dropna().asfreq('MS') 
df_leading_risk['HOUST_YOY'] = (houst_monthly.pct_change(12) * 100).reindex(df_leading_risk.index, method='ffill')

claims_data = weekly_data_raw['IC4WSA'].dropna()
claims_52wk_low = claims_data.rolling(window=52).min()
df_leading_risk['CLAIMS_RISE_PCT'] = ((claims_data / claims_52wk_low) - 1) * 100
df_leading_risk['CLAIMS_RISE_PCT'] = df_leading_risk['CLAIMS_RISE_PCT'].reindex(df_leading_risk.index, method='ffill')
df_leading_risk['CLAIMS_52WK_LOW'] = claims_52wk_low.reindex(df_leading_risk.index, method='ffill')
df_leading_risk['IC4WSA'] = claims_data.reindex(df_leading_risk.index, method='ffill')
df_leading_risk['NEWORDER'] = monthly_data_raw['NEWORDER'].reindex(df_leading_risk.index, method='ffill')

df_leading_risk = df_leading_risk.dropna(subset=['HOUST_YOY', 'CLAIMS_RISE_PCT', 'NEWORDER', 'BAMLH0A0HYM2'], how='all')

# --- 4e. Earnings & Consumer Data (Dashboard 5) ---
print("Processing earnings and consumer data...")
df_earnings_consumer = pd.DataFrame(index=daily_data_raw.index)

# Real Retail Sales
rrsfs_monthly = monthly_data_raw['RRSFS'].asfreq('MS')
df_earnings_consumer['RRSFS_YOY'] = (rrsfs_monthly.pct_change(12) * 100).reindex(df_earnings_consumer.index, method='ffill')

# VIX-MOVE Spread
df_earnings_consumer['VIX_MOVE_SPREAD'] = df_market_risk['VIX_MOVE_SPREAD'].reindex(df_earnings_consumer.index, method='ffill')

df_earnings_consumer = df_earnings_consumer.dropna(how='all')


print("All data processing complete.")

# --- 5. DYNAMIC ANALYSIS & STATUS ASSESSMENT ---
print("Analyzing latest data for status dots...")
status_results = {}
latest_values = {}

# Helper function to safely get the last valid value
def get_last_value(series):
    if series.empty:
        return None
    # .dropna()가 비어있을 수 있으므로 확인
    cleaned_series = series.dropna()
    if cleaned_series.empty:
        return None
    return cleaned_series.iloc[-1]

# Get latest values
try:
    latest_values['baa'] = get_last_value(df_market_risk['BAA10Y'])
    latest_values['move'] = get_last_value(df_market_risk['MOVE'])
    latest_values['t10yie'] = get_last_value(df_market_risk['T10YIE'])
    latest_values['vix'] = get_last_value(df_market_risk['VIX'])
    
    latest_values['res_ratio'] = get_last_value(weekly_data_liq['RESERVE_RATIO_PCT'])
    latest_values['elasticity'] = get_last_value(elasticity_df[f'{rolling_window_weeks_list[-1]}-Wk Elasticity'])
        
    latest_values['effr_spread'] = get_last_value(daily_data_liq['EFFR_Spread'])
    latest_values['sofr_spread'] = get_last_value(daily_data_liq['SOFR_Spread'])
    latest_values['hy_spread'] = get_last_value(daily_data_liq['BAMLH0A0HYM2']) 

    latest_values['t10y2y'] = get_last_value(df_global_risk['T10Y2Y'])
    latest_values['nfci'] = get_last_value(df_global_risk['NFCI'])
    latest_values['dollar'] = get_last_value(df_global_risk['DTWEXBGS'])
    dollar_mean = df_global_risk['DTWEXBGS'].mean()
    
    latest_values['houst_yoy'] = get_last_value(df_leading_risk['HOUST_YOY'])
    latest_values['new_orders'] = get_last_value(df_leading_risk['NEWORDER'])
    latest_values['claims_rise'] = get_last_value(df_leading_risk['CLAIMS_RISE_PCT'])

    latest_values['rrsfs_yoy'] = get_last_value(df_earnings_consumer['RRSFS_YOY'])
    latest_values['vix_move_spread'] = get_last_value(df_earnings_consumer['VIX_MOVE_SPREAD'])

    
    # --- Define Status Thresholds ---
    status_results['baa'] = get_status(latest_values['baa'], {'red_high': 3.5, 'yellow_high': 2.8, 'lightgreen_high': 2.0, 'blue_low': 0})
    status_results['move'] = get_status(latest_values['move'], {'red_high': 150, 'yellow_high': 120, 'lightgreen_high': 100, 'blue_low': 0})
    status_results['t10yie'] = get_status(latest_values['t10yie'], {'red_high': 3.5, 'yellow_high': 3.0, 'lightgreen_high': 2.0, 'yellow_low': 1.5, 'red_low': 1.0})
    status_results['vix'] = get_status(latest_values['vix'], {'red_high': 30, 'yellow_high': 20, 'lightgreen_high': 15, 'blue_low': 0})

    status_results['res_ratio'] = get_status(latest_values['res_ratio'], {'red_low': 12, 'yellow_low': 13, 'lightgreen_high': 100, 'blue_low': 13})
    status_results['elasticity'] = get_status(latest_values['elasticity'], {'red_low': 0, 'yellow_low': 10, 'lightgreen_high': 100, 'blue_low': 10})
    status_results['effr_spread'] = get_status(latest_values['effr_spread'], {'red_high': 10, 'yellow_high': 5, 'grey_low': -5, 'blue_low': -100})
    status_results['sofr_spread'] = get_status(latest_values['sofr_spread'], {'red_high': 10, 'yellow_high': 5, 'grey_low': -5, 'blue_low': -100})
    status_results['hy_spread'] = get_status(latest_values['hy_spread'], {'red_high': 6.0, 'yellow_high': 4.5, 'lightgreen_high': 3.5, 'blue_low': 0})

    status_results['t10y2y'] = get_status(latest_values['t10y2y'], {'red_low': 0, 'yellow_low': 0.25, 'lightgreen_high': 100, 'blue_low': 0.25})
    status_results['nfci'] = get_status(latest_values['nfci'], {'red_high': 0, 'yellow_high': -0.15, 'lightgreen_high': -0.5, 'blue_low': -100})
    status_results['dollar'] = get_status(latest_values['dollar'], {'red_high': dollar_mean + 10, 'yellow_high': dollar_mean + 5, 'lightgreen_high': dollar_mean, 'blue_low': 0})

    status_results['houst_yoy'] = get_status(latest_values['houst_yoy'], {'red_low': -20.0, 'yellow_low': 0, 'lightgreen_high': 100, 'blue_low': 0.01})
    status_results['new_orders'] = get_status(latest_values['new_orders'], {'red_low': -20, 'yellow_low': 0, 'lightgreen_high': 100, 'blue_low': 0.01})
    status_results['claims_rise'] = get_status(latest_values['claims_rise'], {'red_high': 40, 'yellow_high': 20, 'lightgreen_high': 10, 'blue_low': -100})
    
    status_results['rrsfs_yoy'] = get_status(latest_values['rrsfs_yoy'], {'red_low': -1, 'yellow_low': 0, 'lightgreen_high': 100, 'blue_low': 0.01})
    status_results['vix_move_spread'] = get_status(latest_values['vix_move_spread'], {'red_low': -20, 'yellow_low': -10, 'lightgreen_high': 100, 'blue_low': -10})


    print("Status assessment complete.")
    
except Exception as e:
    print(f"Error during status assessment: {e}. Check data availability.")
    status_results = {k: ('grey', 'Data Error') for k in [
        'baa', 'move', 't10yie', 'vix', 'res_ratio', 'elasticity', 
        'effr_spread', 'sofr_spread', 'hy_spread', 't10y2y', 'nfci', 'dollar',
        'houst_yoy', 'new_orders', 'claims_rise',
        'rrsfs_yoy', 'vix_move_spread' # Modified
    ]}

# --- 6. PLOTTING FUNCTIONS ---

# Matplotlib 백엔드를 'Agg'로 설정하여 GUI 창 없이 실행
plt.switch_backend('Agg') # <-- GitHub Actions에서 실행 시 이 줄의 주석을 해제하세요.

def plot_market_risk_dashboard(data, status):
    print("Generating Dashboard 1: Macro Fear...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
    fig.suptitle('Dashboard 1: Macro Fear & Risk', fontsize=20, y=0.98)
    
    recessions = [
        (datetime.datetime(2001, 3, 1), datetime.datetime(2001, 11, 1)),
        (datetime.datetime(2007, 12, 1), datetime.datetime(2009, 6, 1)),
        (datetime.datetime(2020, 2, 1), datetime.datetime(2020, 4, 1))
    ]
    
    # Plot 1: Corporate Credit Spreads (BAA10Y)
    ax = axes[0, 0]
    status_color, status_text = status['baa']
    plot_data = data['BAA10Y'].dropna()
    ax.plot(plot_data.index, plot_data, color='red')
    ax.set_title("1. Credit Spreads (BAA10Y)", fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Percent')
    ax.grid(True, linestyle='--', alpha=0.6)
    mean_val = plot_data.mean()
    ax.axhline(mean_val, color='black', linestyle='--', linewidth=0.75, alpha=0.7)
    ax.legend(['Baa Spread', f'Mean: {mean_val:.2f}'], loc='upper left')

    # Plot 2: Bond Market Volatility (MOVE)
    ax = axes[0, 1]
    status_color, status_text = status['move']
    plot_data = data['MOVE'].dropna()
    ax.plot(plot_data.index, plot_data, color='blue')
    ax.set_title("2. Bond Volatility (MOVE)", fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Index Value')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(120, color='black', linestyle='--', linewidth=0.75, alpha=0.7)
    ax.legend(['MOVE Index', 'High Stress (120)'], loc='upper left')

    # Plot 3: Inflation Expectations (T10YIE)
    ax = axes[1, 0]
    status_color, status_text = status['t10yie']
    plot_data = data['T10YIE'].dropna()
    ax.plot(plot_data.index, plot_data, color='green')
    ax.set_title("3. Inflation Expectations (T10YIE)", fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Percent')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(2.0, color='black', linestyle='--', linewidth=0.75, alpha=0.7)
    ax.axhline(3.0, color='black', linestyle='--', linewidth=0.75, alpha=0.7)
    ax.legend(['10Y Breakeven', '2.0% Level', '3.0% Level'], loc='upper left')
    
    # Plot 4: Stock Market Volatility (VIX)
    ax = axes[1, 1]
    status_color, status_text = status['vix']
    plot_data = data['VIX'].dropna()
    ax.plot(plot_data.index, plot_data, color='purple')
    ax.set_title('4. Stock Volatility (VIX)', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('VIX Index')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(30, color='red', linestyle='--', linewidth=1.5, label='30 (Extreme Fear)')
    ax.axhline(20, color='orange', linestyle='--', linewidth=1.0, label='20 (Fear)')
    ax.legend(loc='upper left')
    
    # Common formatting for this figure
    for ax_row in axes:
        for ax in ax_row:
            for start, end in recessions:
                ax.axvspan(start, end, color='grey', alpha=0.2)
            ax.set_xlim(start_date_long, end_date)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlabel('Date')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def plot_liquidity_dashboard(daily_data, weekly_data, elasticity, asset_data, shading, status):
    print("Generating Dashboard 2: Liquidity & Plumbing...")
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle('Dashboard 2: Monetary Analysis & Liquidity', fontsize=24, y=1.0)

    # --- PLOT [0, 0]: FIGURE 5 (Bank Reserve Levels) ---
    ax = axes[0, 0]
    status_color, status_text = status['res_ratio']
    data = weekly_data['RESERVE_RATIO_PCT'].dropna()
    ax.plot(data.index, data, color='blue', label='Reserves / Bank Assets')
    ax.axhspan(12, 13, color='grey', alpha=0.3, label='Reserve Scarcity Zone')
    ax.set_title('5. Bank Reserve Levels', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Reserves as % of Bank Assets', color='blue')
    ax.grid(True, linestyle='--', alpha=0.6)

    if 'RRPONTSYD_B' in daily_data.columns:
        ax_twin = ax.twinx()
        rrp_data = daily_data['RRPONTSYD_B'].dropna()
        ax_twin.fill_between(rrp_data.index, rrp_data, 0, color='gray', alpha=0.3, label='ON RRP Usage')
        ax_twin.set_ylabel('ON RRP Usage ($B)', color='gray')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_twin.get_legend_handles_labels()
        ax.legend(handles=h1 + h2, labels=l1 + l2, loc='upper left', fontsize='small')

    # --- PLOT [0, 1]: FIGURE 6 (System Regime) ---
    ax = axes[0, 1]
    status_color, status_text = status['elasticity']
    ax.set_title('6. System Regime (Elasticity)', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    
    colors = ['green', 'blue', 'red'] # Fast, Medium, Slow
    
    if not elasticity.empty:
        for i, window in enumerate(rolling_window_weeks_list):
            col_name = f'{window}-Wk Elasticity'
            if col_name in elasticity.columns:
                ax.plot(elasticity.index, elasticity[col_name], 
                        color=colors[i], label=f'{window}-Wk Elasticity', 
                        linewidth=(i*1)+1)
    
    ax.set_ylabel('Elasticity (bp / pp change)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize='small')

    # --- PLOT [1, 0]: FIGURE 7 (Unsecured Stress) ---
    ax = axes[1, 0]
    status_color, status_text = status['hy_spread']
    data_effr = daily_data['EFFR_Spread'].dropna()
    data_effr_ma = data_effr.rolling(window=30).mean()
    ax.plot(data_effr.index, data_effr, color='red', label='Daily EFFR Spread', linewidth=0.5, alpha=0.6)
    ax.plot(data_effr_ma.index, data_effr_ma, color='red', label='30-Day MA EFFR Spread', linewidth=2.5, linestyle='--')
    ax.set_title('7. Unsecured & Corp Stress', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('EFFR Spread (Bps)', color='red')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(True, linestyle='--', alpha=0.6)

    if 'BAMLH0A0HYM2' in daily_data.columns:
        ax_twin = ax.twinx()
        hy_spread_data = daily_data['BAMLH0A0HYM2'].dropna()
        ax_twin.plot(hy_spread_data.index, hy_spread_data, color='purple', linestyle='--', label='High-Yield Spread (%)')
        ax_twin.set_ylabel('High-Yield Spread (%)', color='purple')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_twin.get_legend_handles_labels()
        ax.legend(handles=h1 + h2, labels=l1 + l2, loc='upper left', fontsize='small')

    # --- PLOT [1, 1]: FIGURE 8 (Secured Corridor) ---
    ax = axes[1, 1]
    status_color, status_text = status['sofr_spread']
    data_sofr = daily_data['SOFR_Spread'].dropna()
    data_sofr_ma = data_sofr.rolling(window=30).mean()
    ax.plot(daily_data.index, daily_data['DW_Spread'], label='Discount Window (Ceiling)', linestyle='--', color='gray')
    ax.plot(daily_data.index, daily_data['RRP_Spread'], label='ON RRP (Floor)', linestyle='--', color='blue')
    ax.plot(data_sofr.index, data_sofr, label='SOFR (Daily)', linewidth=0.5, color='red', alpha=0.6)
    ax.plot(data_sofr_ma.index, data_sofr_ma, label='SOFR (30-Day MA)', linewidth=2.5, color='red', linestyle='--')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.set_title('8. Secured Market (Corridor)', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Spread (Basis Points)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize='small')

    # Common formatting for this figure
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(start_date_short, end_date)
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.set_xlabel('Date')
            
            if shading['available']:
                bottom, top = ax.get_ylim()
                ax.fill_between(shading['drawdown'].index, bottom, top, where=shading['correction'],
                                facecolor='orange', alpha=0.3, zorder=0)
                ax.fill_between(shading['drawdown'].index, bottom, top, where=shading['decline'],
                                facecolor='red', alpha=0.3, zorder=0)
                ax.set_ylim(bottom, top)
            
            ax.axvspan(datetime.datetime(2019, 9, 16), datetime.datetime(2019, 9, 20),
                       alpha=0.3, color='blue', zorder=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_global_risk_dashboard(data, status):
    print("Generating Dashboard 3: Global & Leading Risk...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Dashboard 3: Global Risk & Recession Indicators', fontsize=20, y=0.98)
    
    recessions = [
        (datetime.datetime(2001, 3, 1), datetime.datetime(2001, 11, 1)),
        (datetime.datetime(2007, 12, 1), datetime.datetime(2009, 6, 1)),
        (datetime.datetime(2020, 2, 1), datetime.datetime(2020, 4, 1))
    ]

    # Plot 9: Yield Curve (T10Y2Y)
    ax = axes[0, 0]
    status_color, status_text = status['t10y2y']
    plot_data = data['T10Y2Y'].dropna()
    ax.plot(plot_data.index, plot_data, color='blue', label='10Y-2Y Spread')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='0 (Inversion Line)')
    ax.fill_between(plot_data.index, plot_data, 0, where=plot_data < 0, color='red', alpha=0.3, label='Yield Curve Inversion')
    ax.set_title('9. Yield Curve (Recession)', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Yield Spread (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    # Plot 10: Financial Conditions (NFCI)
    ax = axes[0, 1]
    status_color, status_text = status['nfci']
    plot_data = data['NFCI'].dropna()
    ax.plot(plot_data.index, plot_data, color='green', label='NFCI Index')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='0 (Tight/Loose Line)')
    ax.fill_between(plot_data.index, 0, plot_data, where=plot_data > 0, color='red', alpha=0.3, label='Tight Financial Conditions')
    ax.set_title('10. Financial Conditions (NFCI)', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Std. Deviations from Average')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    # Plot 11: Dollar Index (DTWEXBGS)
    ax = axes[1, 0]
    status_color, status_text = status['dollar']
    plot_data = data['DTWEXBGS'].dropna()
    ax.plot(plot_data.index, plot_data, color='orange', label='Trade-Weighted Dollar Index')
    mean_val = plot_data.mean()
    ax.axhline(mean_val, color='black', linestyle='--', linewidth=1.0, label=f'Average ({mean_val:.2f})')
    ax.set_title('11. Dollar Index (Global Flow)', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Index (Broad, Goods & Services)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')
    
    # Plot [1, 1]: Empty for Legend
    ax = axes[1, 1]
    ax.axis('off') # Turn off the plot
    
    patch_decline = mpatches.Patch(color='red', alpha=0.3, label='Recession / S&P -20%')
    patch_correction = mpatches.Patch(color='orange', alpha=0.3, label='S&P -10% Correction')
    patch_repo_spike = mpatches.Patch(color='blue', alpha=0.3, label='Repo Spike 2019')
    shading_handles = [patch_decline, patch_correction, patch_repo_spike]
    ax.legend(handles=shading_handles, labels=[h.get_label() for h in shading_handles],
              loc='center', fontsize='large', title='Master Legend')

    # Common formatting
    for ax_row in axes:
        for ax in ax_row:
            if ax == axes[1, 1]: continue 
            for start, end in recessions:
                ax.axvspan(start, end, color='grey', alpha=0.2)
            ax.set_xlim(start_date_long, end_date)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlabel('Date')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_leading_indicators_dashboard(data, daily_data, status):
    print("Generating Dashboard 4: Leading Economic Indicators...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Dashboard 4: Leading Economic Indicators (Recession)', fontsize=20, y=0.98)
    
    recessions = [
        (datetime.datetime(2001, 3, 1), datetime.datetime(2001, 11, 1)),
        (datetime.datetime(2007, 12, 1), datetime.datetime(2009, 6, 1)),
        (datetime.datetime(2020, 2, 1), datetime.datetime(2020, 4, 1))
    ]

    # Plot 12: Housing Starts Year-over-Year
    ax = axes[0, 0]
    status_color, status_text = status['houst_yoy']
    plot_data = data['HOUST_YOY'].dropna()
    ax.plot(plot_data.index, plot_data, color='purple', label='Housing Starts % Change (YoY)')
    ax.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax.axhline(-20.0, color='red', linestyle='--', linewidth=1.5, label='-20% (Recession Signal)')
    ax.fill_between(plot_data.index, plot_data, 0, where=plot_data < 0, color='yellow', alpha=0.3, label='Contraction')
    ax.fill_between(plot_data.index, plot_data, -20.0, where=plot_data < -20.0, color='red', alpha=0.3, label='Recession Signal')
    ax.set_title('12. Housing Starts (YoY)', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Percent Change (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left')

    # Plot 13: Philly Fed New Orders
    ax = axes[0, 1]
    status_color, status_text = status['new_orders']
    plot_data = data['NEWORDER'].dropna()
    ax.plot(plot_data.index, plot_data, color='orange', label='Philly Fed New Orders Index')
    ax.axhline(0, color='black', linestyle='--', linewidth=1.0, label='0 (Growth/Contraction)')
    ax.axhline(-20, color='red', linestyle='--', linewidth=1.5, label='-20 (Recession Signal)')
    ax.fill_between(plot_data.index, plot_data, 0, where=plot_data < 0, color='yellow', alpha=0.3, label='Contraction')
    ax.fill_between(plot_data.index, plot_data, -20, where=plot_data < -20, color='red', alpha=0.3, label='Recession Signal')
    ax.set_title('13. Philly Fed New Orders', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Index (0 = Neutral)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left')

    # Plot 14: Initial Jobless Claims
    ax = axes[1, 0]
    status_color, status_text = status['claims_rise']
    plot_data = data['IC4WSA'].dropna()
    plot_data_low = data['CLAIMS_52WK_LOW'].dropna()
    ax.plot(plot_data.index, plot_data, color='blue', label='Jobless Claims (4-Wk MA)')
    ax.plot(plot_data_low.index, plot_data_low, color='green', linestyle='--', label='52-Week Low')
    ax.set_title('14. Initial Jobless Claims', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Number of Claims')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')
    
    # Plot 15: High-Yield Spread
    ax = axes[1, 1]
    status_color, status_text = status['hy_spread']
    plot_data = daily_data['BAMLH0A0HYM2'].dropna()
    ax.plot(plot_data.index, plot_data, color='purple', linestyle='--', label='High-Yield Spread (%)')
    ax.set_title('15. High-Yield Spread', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('High-Yield Spread (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(4.5, color='orange', linestyle='--', linewidth=1.0, label='4.5 (Warning)')
    ax.axhline(6.0, color='red', linestyle='--', linewidth=1.5, label='6.0 (High Risk)')
    ax.legend(loc='upper left')

    # Common formatting
    for ax_row in axes:
        for ax in ax_row:
            for start, end in recessions:
                ax.axvspan(start, end, color='grey', alpha=0.2)
            ax.set_xlim(start_date_long, end_date)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_xlabel('Date')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
    
def plot_earnings_consumer_dashboard(data, market_data, status):
    print("Generating Dashboard 5: Consumer & Risk Appetite...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) 
    fig.suptitle('Dashboard 5: Consumer Health & Risk Appetite', fontsize=20, y=1.02)
    
    recessions = [
        (datetime.datetime(2001, 3, 1), datetime.datetime(2001, 11, 1)),
        (datetime.datetime(2007, 12, 1), datetime.datetime(2009, 6, 1)),
        (datetime.datetime(2020, 2, 1), datetime.datetime(2020, 4, 1))
    ]

    # --- Plot [0, 0]: FIGURE 16 (Real Retail Sales) ---
    ax = axes[0] # First plot
    status_color, status_text = status['rrsfs_yoy']
    plot_data = data['RRSFS_YOY'].dropna()
    ax.plot(plot_data.index, plot_data, color='orange', label='Real Retail Sales % Change (YoY)')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='0 (Contraction)')
    ax.fill_between(plot_data.index, plot_data, 0, where=plot_data < 0, color='red', alpha=0.3, label='Contraction')
    ax.set_title('16. Real Retail Sales (YoY)', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Percent Change (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left')
    
    # --- Plot [0, 1]: FIGURE 17 (VIX-MOVE Spread) ---
    ax = axes[1] # Second plot
    status_color, status_text = status['vix_move_spread']
    plot_data = market_data['VIX_MOVE_SPREAD'].dropna()
    ax.plot(plot_data.index, plot_data, color='cyan', label='VIX - MOVE Spread')
    ax.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax.axhline(-10, color='orange', linestyle='--', linewidth=1.0, label='-10 (Warning)')
    ax.axhline(-20, color='red', linestyle='--', linewidth=1.5, label='-20 (Systemic Risk)')
    ax.fill_between(plot_data.index, plot_data, 0, where=plot_data < 0, color='yellow', alpha=0.3)
    ax.fill_between(plot_data.index, plot_data, -10, where=plot_data < -10, color='red', alpha=0.3)
    ax.set_title('17. VIX-MOVE Spread (Risk Appetite)', fontsize=14, loc='left')
    ax.set_title(f"● {status_text}", fontsize=12, color=status_color, loc='right')
    ax.set_ylabel('Spread (VIX - MOVE)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left')

    # --- Plot [0, 2]: Master Legend ---
    ax = axes[2] # Third plot
    ax.axis('off')
    patch_decline = mpatches.Patch(color='red', alpha=0.3, label='Recession / S&P -20%')
    patch_correction = mpatches.Patch(color='orange', alpha=0.3, label='S&P -10% Correction')
    patch_repo_spike = mpatches.Patch(color='blue', alpha=0.3, label='Repo Spike 2019')
    shading_handles = [patch_decline, patch_correction, patch_repo_spike]
    ax.legend(handles=shading_handles, labels=[h.get_label() for h in shading_handles],
              loc='center', fontsize='large', title='Master Legend')

    # Common formatting
    for ax in axes[:2]: # Only format the first two plots
        for start, end in recessions:
            ax.axvspan(start, end, color='grey', alpha=0.2)
        ax.set_xlim(start_date_long, end_date)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel('Date')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# --- 7. MAIN EXECUTION BLOCK ---
print("--- Generating All Dashboards ---")

# Pack data for plotting functions
shading_data = {
    'available': dynamic_shading_available,
    'drawdown': drawdown if dynamic_shading_available else None,
    'correction': in_correction if dynamic_shading_available else None,
    'decline': in_decline if dynamic_shading_available else None
}

# --- Create all five figures ---
print("Generating and saving Dashboard 1 (Macro Fear)...")
fig1 = plot_market_risk_dashboard(df_market_risk, status_results)
fig1.savefig('dashboard_1_market_risk.png', dpi=150, bbox_inches='tight')

print("Generating and saving Dashboard 2 (Liquidity)...")
fig2 = plot_liquidity_dashboard(daily_data_liq, weekly_data_liq, elasticity_df, asset_norm, shading_data, status_results)
fig2.savefig('dashboard_2_liquidity.png', dpi=150, bbox_inches='tight')

print("Generating and saving Dashboard 3 (Global Risk)...")
fig3 = plot_global_risk_dashboard(df_global_risk, status_results)
fig3.savefig('dashboard_3_global_risk.png', dpi=150, bbox_inches='tight')

print("Generating and saving Dashboard 4 (Leading Indicators)...")
fig4 = plot_leading_indicators_dashboard(df_leading_risk, daily_data_raw, status_results)
fig4.savefig('dashboard_4_leading_risk.png', dpi=150, bbox_inches='tight')

print("Generating and saving Dashboard 5 (Consumer & Risk Appetite)...")
fig5 = plot_earnings_consumer_dashboard(df_earnings_consumer, df_market_risk, status_results)
fig5.savefig('dashboard_5_earnings_consumer.png', dpi=150, bbox_inches='tight')


# --- Show plots *after* saving ---
# *** MODIFICATION: plt.show() is commented out for automated run ***
print("\nAll dashboards generated and saved.")
print("Displaying figures... (Close all figure windows to continue to the text report)")
# plt.show() # <-- 주석 처리 (GitHub Actions 실행용)

# --- Close figures after user closes them ---
print("Closing figures to save memory...")
plt.close('all') # Close all figures
print("All dashboard images saved successfully.")


# --- 8. AUTOMATED TEXT INTERPRETATION ---
print("\n" + "="*70)
print("--- AUTOMATED DASHBOARD INTERPRETATION ---")
# *** MODIFICATION: Get current time in KST (UTC+9) ***
kst = datetime.timezone(datetime.timedelta(hours=9))
report_time = datetime.datetime.now(kst).strftime('%Y-%m-%d %H:%M KST')
print(f"--- Report Generated on: {report_time} ---")
print("="*70 + "\n")

# Create a list to hold the report lines
report_lines = []
report_lines.append("="*70)
report_lines.append("--- AUTOMATED DASHBOARD INTERPRETATION ---")
report_lines.append(f"--- Report Generated on: {report_time} ---")
report_lines.append("="*70 + "\n")

# Helper function for text formatting
def format_value(key, value):
    if value is None:
        return "N/A"
    if key in ['baa', 't10yie', 'hy_spread', 't10y2y', 'houst_yoy', 'claims_rise', 'cpata_yoy', 'rrsfs_yoy', 'res_ratio']:
        return f"{value:.2f}%"
    if key in ['move', 'vix', 'dollar', 'new_orders', 'vix_move_spread']:
        return f"{value:.2f}"
    if key in ['elasticity', 'nfci']:
        return f"{value:.2f}"
    if key in ['effr_spread', 'sofr_spread']:
        return f"{value:.2f} bps"
    return f"{value}"

try:
    # --- Section 1: Macro & Fear ---
    report_lines.append("--- 1. MACRO RISK & FEAR (Dash 1) ---")
    report_lines.append(f"  Credit Spreads (BAA10Y): {status_results['baa'][1]} (Value: {format_value('baa', latest_values['baa'])})")
    report_lines.append(f"  Stock Volatility (VIX): {status_results['vix'][1]} (Value: {format_value('vix', latest_values['vix'])})")
    report_lines.append(f"  Bond Volatility (MOVE): {status_results['move'][1]} (Value: {format_value('move', latest_values['move'])})")
    report_lines.append(f"  Inflation Expectations: {status_results['t10yie'][1]} (Value: {format_value('t10yie', latest_values['t10yie'])})")

    # --- Section 2: Liquidity & Plumbing ---
    report_lines.append("\n--- 2. LIQUIDITY & SYSTEM PLUMBING (Dash 2) ---")
    report_lines.append(f"  Bank Reserve Ratio: {status_results['res_ratio'][1]} (Value: {format_value('res_ratio', latest_values['res_ratio'])})")
    report_lines.append(f"  System Regime (Elasticity): {status_results['elasticity'][1]} (Value: {format_value('elasticity', latest_values['elasticity'])})")
    report_lines.append(f"  Unsecured Stress (HY Spread): {status_results['hy_spread'][1]} (Value: {format_value('hy_spread', latest_values['hy_spread'])})")
    report_lines.append(f"  Secured Stress (SOFR Spread): {status_results['sofr_spread'][1]} (Value: {format_value('sofr_spread', latest_values['sofr_spread'])})")

    # --- Section 3: Leading & Global ---
    report_lines.append("\n--- 3. LEADING & GLOBAL INDICATORS (Dash 3) ---")
    report_lines.append(f"  Yield Curve (T10Y2Y): {status_results['t10y2y'][1]} (Value: {format_value('t10y2y', latest_values['t10y2y'])})")
    report_lines.append(f"  Financial Conditions (NFCI): {status_results['nfci'][1]} (Value: {format_value('nfci', latest_values['nfci'])})")
    report_lines.append(f"  US Dollar Index: {status_results['dollar'][1]} (Value: {format_value('dollar', latest_values['dollar'])})")

    # --- Section 4: Leading Economic Indicators ---
    report_lines.append("\n--- 4. LEADING ECONOMIC INDICATORS (Dash 4) ---")
    report_lines.append(f"  Housing Starts (YoY %): {status_results['houst_yoy'][1]} (Value: {format_value('houst_yoy', latest_values['houst_yoy'])})")
    report_lines.append(f"  Philly Fed New Orders (PMI): {status_results['new_orders'][1]} (Value: {format_value('new_orders', latest_values['new_orders'])})")
    report_lines.append(f"  Jobless Claims (% from Low): {status_results['claims_rise'][1]} (Value: {format_value('claims_rise', latest_values['claims_rise'])})")
    report_lines.append(f"  High-Yield Spread: {status_results['hy_spread'][1]} (Value: {format_value('hy_spread', latest_values['hy_spread'])})")


    # --- Section 5: Earnings & Consumer ---
    report_lines.append("\n--- 5. CONSUMER & RISK APPETITE (Dash 5) ---")
    # *** MODIFIED: Removed Corp. Profits ***
    report_lines.append(f"  Real Retail Sales (YoY %): {status_results['rrsfs_yoy'][1]} (Value: {format_value('rrsfs_yoy', latest_values['rrsfs_yoy'])})")
    report_lines.append(f"  VIX-MOVE Spread: {status_results['vix_move_spread'][1]} (Value: {format_value('vix_move_spread', latest_values['vix_move_spread'])})")


    # --- Section 6: Overall Summary ---
    report_lines.append("\n" + "-"*70)
    report_lines.append("--- OVERALL SUMMARY ---")
    
    all_colors = [s[0] for s in status_results.values() if s[1] != "Data N/A"]
    red_count = all_colors.count(STATUS_COLORS['red'])
    yellow_count = all_colors.count(STATUS_COLORS['yellow'])
    blue_green_count = len(all_colors) - red_count - yellow_count - all_colors.count(STATUS_COLORS['grey'])

    # *** MODIFIED: Adjusted thresholds for fewer indicators ***
    if red_count >= 4: 
        report_lines.append(f"ALERT: Multiple ({red_count}) high-risk indicators are active. The system is showing significant stress.")
    elif red_count > 0:
        report_lines.append(f"WARNING: {red_count} high-risk indicator(s) active. Monitor the situation closely.")
        if status_results['t10y2y'][0] == STATUS_COLORS['red']:
             report_lines.append("  > The Yield Curve is inverted, signaling a future recession risk.")
        if status_results['houst_yoy'][0] == STATUS_COLORS['red']:
             report_lines.append("  > Housing Starts are down significantly, strongly signaling a future recession.")
        if status_results['nfci'][0] == STATUS_COLORS['red']:
             report_lines.append("  > Financial Conditions are tight, which could slow down the economy.")
    elif yellow_count >= 5:
        report_lines.append("CAUTION: Several indicators are in a warning state. While not a crisis, uncertainty is high.")
    elif blue_green_count > 10:
        report_lines.append("STABLE: The dashboard shows a stable and low-risk environment.")
    else:
        report_lines.append("NEUTRAL: The dashboard shows a mixed environment. Monitor for developing trends.")

    # Specific conflict message
    if 't10y2y' in status_results and 'baa' in status_results and \
       status_results['t10y2y'][0] == STATUS_COLORS['red'] and \
       status_results['baa'][0] != STATUS_COLORS['red']:
        report_lines.append("\nNOTE: A significant conflict exists: Leading indicators (Yield Curve, Housing) signal a recession,")
        report_lines.append("      but current indicators (Credit Spreads) are signaling economic health.")

    # --- 8b. Save and Print Report ---
    report_content = "\n".join(report_lines)
    
    print(report_content)
    
    report_filename = "dashboard_summary_report.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nSuccessfully saved summary report to {report_filename}")

except Exception as e:
    error_message = f"Error generating text interpretation: {e}"
    print(error_message)
    with open("dashboard_summary_report.txt", "w", encoding="utf-8") as f:
        f.write(error_message)

print("\n" + "="*70)
print("--- Master Dashboard Script Finished ---")

# *** MODIFICATION: Add a function to generate README.md ***
def generate_readme(report_file, image_files, status):
    print("Generating README.md...")
    # KST 시간 생성
    kst = datetime.timezone(datetime.timedelta(hours=9))
    report_time = datetime.datetime.now(kst).strftime('%Y-%m-%d %H:%M KST')
    cache_buster = int(time.time()) # 캐시 무력화를 위한 타임스탬프

    # 리포트 파일 읽기
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            report_content = f.read()
    except Exception as e:
        print(f"Warning: Could not read summary report file. Error: {e}")
        report_content = "Summary report is currently being generated..."

    # README 내용 구성
    readme_content = f"""# Monetary Dashboard (Updated Daily)
Last Updated: {report_time}

This repository automatically generates and updates a comprehensive 5-part dashboard of key financial and economic indicators every day. The analysis is performed by `master_dashboard.py` and run via GitHub Actions.

---

### **Automated Summary Report**
<details>
<summary>Click to expand the latest automated analysis</summary>

```
{report_content}
```
</details>

---

### Dashboard 1: Macro Fear & Risk
(Current market sentiment and volatility)
![Dashboard 1: Macro Fear & Risk](dashboard_1_market_risk.png?v={cache_buster})

### Dashboard 2: Monetary Analysis & Liquidity
(Health of the core banking system plumbing)
![Dashboard 2: Monetary Analysis & Liquidity](dashboard_2_liquidity.png?v={cache_buster})

### Dashboard 3: Global Risk & Recession Indicators
(Leading indicators for global risk and recession)
![Dashboard 3: Global Risk & Recession Indicators](dashboard_3_global_risk.png?v={cache_buster})

### Dashboard 4: Leading Economic Indicators
(Real economy leading indicators for recession)
![Dashboard 4: Leading Economic Indicators](dashboard_4_leading_risk.png?v={cache_buster})

### Dashboard 5: Consumer Health & Risk Appetite
(Consumer health and relative risk sentiment)
![Dashboard 5: Consumer & Risk Appetite](dashboard_5_earnings_consumer.png?v={cache_buster})
"""
    
    # README.md 파일 쓰기
    try:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("README.md generated successfully.")
    except Exception as e:
        print(f"Error writing README.md: {e}")

print("\n" + "="*70)
print("--- Master Dashboard Script Finished ---")

# *** 9. NEW: Generate index.html instead of README.md ***
def generate_html_page(report_file, image_files, status):
    print("Generating index.html...")
    # KST 시간 생성
    kst = datetime.timezone(datetime.timedelta(hours=9))
    report_time = datetime.datetime.now(kst).strftime('%Y-%m-%d %H:%M KST')
    cache_buster = int(time.time()) # 캐시 무력화를 위한 타임스탬프

    # 리포트 파일 읽기
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            # \n을 <br>로, 스페이스를 &nbsp;로 HTML에 맞게 변환
            report_content = f.read().replace("\n", "<br>").replace(" ", "&nbsp;")
    except Exception as e:
        print(f"Warning: Could not read summary report file. Error: {e}")
        report_content = "Summary report is currently being generated..."
        
    # HTML 템플릿
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monetary Dashboard (Updated Daily)</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f4f7f6; 
            color: #333; 
            margin: 0; 
            padding: 20px; 
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: auto; 
            background-color: #ffffff;
            border: 1px solid #ddd; 
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
        }}
        header {{ 
            padding: 20px 30px; 
            border-bottom: 2px solid #eee; 
        }}
        header h1 {{ 
            margin: 0; 
            color: #1a1a1a;
        }}
        header p {{ 
            margin: 5px 0 0; 
            font-size: 1.1em; 
            color: #555; 
        }}
        .content {{ 
            padding: 30px; 
        }}
        .summary {{
            background-color: #fdfdfd;
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .summary details {{ 
            cursor: pointer; 
        }}
        .summary summary {{ 
            font-weight: bold; 
            font-size: 1.2em;
            color: #0056b3;
        }}
        .report-content {{
            font-family: 'Courier New', Courier, monospace;
            background-color: #2b2b2b;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre;
            font-size: 0.9em;
            margin-top: 15px;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr; /* 모바일에서는 1열 */
            gap: 30px;
        }}
        .dashboard-item {{
            border: 1px solid #eee;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            background: #fff;
        }}
        .dashboard-item h2 {{
            font-size: 1.5em;
            margin: 0;
            padding: 20px;
            border-bottom: 1px solid #eee;
        }}
        .dashboard-item img {{ 
            width: 100%; 
            height: auto; 
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            display: block;
        }}
        
        /* 2열 레이아웃 (태블릿 이상) */
        @media (min-width: 1024px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
        
        /* 1열 레이아웃 (대시보드 5) */
        .dashboard-item.full-width {{
             grid-column: 1 / -1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Monetary Dashboard (Updated Daily)</h1>
            <p>Last Updated: {report_time}</p>
        </header>
        
        <div class="content">
            <div class="summary">
                <details>
                    <summary>Click to expand the latest automated analysis</summary>
                    <div class="report-content">{report_content}</div>
                </details>
            </div>
            
            <div class="dashboard-grid">
                <div class="dashboard-item">
                    <h2>Dashboard 1: Macro Fear & Risk</h2>
                    <img src="dashboard_1_market_risk.png?v={cache_buster}" alt="Dashboard 1: Macro Fear & Risk">
                </div>
                
                <div class="dashboard-item">
                    <h2>Dashboard 2: Monetary Analysis & Liquidity</h2>
                    <img src="dashboard_2_liquidity.png?v={cache_buster}" alt="Dashboard 2: Monetary Analysis & Liquidity">
                </div>
                
                <div class="dashboard-item">
                    <h2>Dashboard 3: Global Risk & Recession Indicators</h2>
                    <img src="dashboard_3_global_risk.png?v={cache_buster}" alt="Dashboard 3: Global Risk & Recession Indicators">
                </div>
                
                <div class="dashboard-item">
                    <h2>Dashboard 4: Leading Economic Indicators</h2>
                    <img src="dashboard_4_leading_risk.png?v={cache_buster}" alt="Dashboard 4: Leading Economic Indicators">
                </div>
                
                <div class="dashboard-item full-width">
                    <h2>Dashboard 5: Consumer Health & Risk Appetite</h2>
                    <img src="dashboard_5_earnings_consumer.png?v={cache_buster}" alt="Dashboard 5: Consumer & Risk Appetite">
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    # HTML 파일 쓰기
    try:
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html_template)
        print("index.html generated successfully.")
    except Exception as e:
        print(f"Error writing index.html: {e}")

# *** 9. MODIFIED: Call new HTML generation function ***
image_files = [
    'dashboard_1_market_risk.png',
    'dashboard_2_liquidity.png',
    'dashboard_3_global_risk.png',
    'dashboard_4_leading_risk.png',
    'dashboard_5_earnings_consumer.png'
]
generate_html_page('dashboard_summary_report.txt', image_files, status_results) # 'generate_readme' 대신 호출

print("\n--- Master Dashboard Script TRULY Finished ---")
