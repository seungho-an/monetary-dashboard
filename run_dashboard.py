import pandas_datareader.data as web
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings
import time

# --- 0. Suppress Warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# --- 1. Global Settings ---
start_date = datetime.datetime(2018, 1, 1)
end_date = datetime.datetime.now()
rolling_window_weeks = 52

print("--- Running Combined Script ---")
print(f"Data range: {start_date.date()} to {end_date.date()}")
print(f"Rolling elasticity window: {rolling_window_weeks} weeks")

# --- Define Market Event Periods ---
repo_spike_color = 'blue'
repo_spike_2019 = (datetime.datetime(2019, 9, 16), datetime.datetime(2019, 9, 20))

# Colors for dynamic shading
decline_color = 'red'
correction_color = 'orange'

# --- Flags ---
asset_data_fetched = False
dynamic_shading_available = False

try:
    # --- 2A. Master Data Fetch (FRED) ---
    print("\nFetching all required FRED data...")
    daily_series_ids = ['EFFR', 'IORB', 'IOER', 'SOFR', 'RRPONTSYAWARD', 'DPCREDIT',
                        'RRPONTSYD', 'BAMLH0A0HYM2']
    weekly_series_ids = ['WRESBAL', 'TLAACBW027SBOG']
    
    daily_data_raw = web.DataReader(daily_series_ids, 'fred', start_date, end_date)
    weekly_data_raw = web.DataReader(weekly_series_ids, 'fred', start_date, end_date)
    
    # --- 2B. ROBUST ASSET FETCH (Yahoo Finance) ---
    print("Fetching Asset Price data (S&P 500, Nasdaq, Dow, Russell, BTC)...")
    asset_tickers = ['^GSPC', '^IXIC', 'BTC-USD', '^DJI', '^RUT'] 
    asset_prices_full = pd.DataFrame() 
    
    try:
        asset_data_raw = yf.download(asset_tickers, start=start_date, end=end_date)
        
        if asset_data_raw.empty:
            print("\nWARNING: Asset data download was empty. Skipping overlays.")
        else:
            if 'Adj Close' in asset_data_raw.columns:
                asset_prices_full = asset_data_raw['Adj Close']
            elif 'Close' in asset_data_raw.columns:
                print("\nWARNING: 'Adj Close' not found. Falling back to 'Close' prices.")
                asset_prices_full = asset_data_raw['Close']
            else:
                raise KeyError("No valid price columns found.")
            
            asset_prices_full = pd.DataFrame(asset_prices_full).ffill()
            
            downloaded_tickers = [t for t in asset_tickers if t in asset_prices_full.columns]
            if downloaded_tickers:
                asset_prices_full = asset_prices_full[downloaded_tickers]
                asset_data_fetched = True
                print(f"Asset data fetched successfully for: {downloaded_tickers}")
            else:
                 print("\nWARNING: No valid asset ticker data found after processing.")
            
    except Exception as e:
        print(f"\nWARNING: Failed to fetch asset price data. Skipping asset overlays. Error: {e}")

    time.sleep(1) 

    # --- 3A. Master Data Processing (FRED) ---
    print("Processing FRED data...")
    daily_data = daily_data_raw.ffill()
    daily_data['POLICY_RATE'] = daily_data['IORB'].fillna(daily_data['IOER'])
    daily_data['EFFR_Spread'] = (daily_data['EFFR'] - daily_data['POLICY_RATE']) * 100
    daily_data['SOFR_Spread'] = (daily_data['SOFR'] - daily_data['POLICY_RATE']) * 100
    daily_data['RRP_Spread'] = (daily_data['RRPONTSYAWARD'] - daily_data['POLICY_RATE']) * 100
    daily_data['DW_Spread'] = (daily_data['DPCREDIT'] - daily_data['POLICY_RATE']) * 100
    
    if 'RRPONTSYD' in daily_data.columns:
        daily_data['RRPONTSYD_B'] = daily_data['RRPONTSYD'] / 1000
    
    weekly_data_raw['RESERVE_RATIO_PCT'] = (weekly_data_raw['WRESBAL'] / weekly_data_raw['TLAACBW027SBOG']) * 100

    # --- 3B. ROBUST Asset Price Processing ---
    if asset_data_fetched:
        print("Processing and normalizing asset prices...")
        asset_norm = pd.DataFrame(index=asset_prices_full.index)
        
        for col in asset_prices_full.columns:
            first_valid_price = asset_prices_full[col].dropna().iloc[0]
            if pd.notna(first_valid_price) and first_valid_price > 0:
                asset_norm[col] = (asset_prices_full[col] / first_valid_price) * 100
            else:
                asset_norm[col] = pd.NA
        
        asset_norm = asset_norm.fillna(100)
                
        if '^GSPC' in asset_prices_full.columns:
            print("  Calculating dynamic drawdown periods for S&P 500...")
            sp500_prices = asset_prices_full['^GSPC'].dropna() 
            rolling_high = sp500_prices.rolling(window=252, min_periods=1).max()
            drawdown = (sp500_prices - rolling_high) / rolling_high
            
            in_correction = (drawdown <= -0.10) & (drawdown > -0.20)
            in_decline = (drawdown <= -0.20)
            dynamic_shading_available = True
        else:
            print("  WARNING: S&P 500 data not found. Cannot calculate dynamic shading.")
    
    # --- 4. Create Final 2x2 Plot ---
    print("Generating all plots in a 2x2 grid...")
    fig, axes = plt.subplots(2, 2, figsize=(24, 16)) 
    
    year_locator = mdates.YearLocator()
    month_locator = mdates.MonthLocator()
    date_formatter = mdates.DateFormatter('%Y-%m')

    # --- PLOT [0, 0]: FIGURE 1 ---
    ax = axes[0, 0]
    panel_a_data = weekly_data_raw['RESERVE_RATIO_PCT'].dropna()
    ax.plot(panel_a_data.index, panel_a_data, color='blue', label='Reserves / Bank Assets')
    ax.axhspan(12, 13, color='grey', alpha=0.3, label='Reserve Scarcity Zone')
    ax.set_title('Figure 1: Bank Reserve Levels (Liquidity)') 
    ax.set_ylabel('Reserves as % of Bank Assets (Left Axis)', color='blue')
    ax.grid(True, linestyle='--', alpha=0.6)

    if 'RRPONTSYD_B' in daily_data.columns:
        ax1_twin = ax.twinx()
        rrp_data = daily_data['RRPONTSYD_B'].dropna()
        ax1_twin.fill_between(rrp_data.index, rrp_data, 0, color='gray', alpha=0.3, label='ON RRP Usage')
        ax1_twin.set_ylabel('ON RRP Usage ($ Billions, Right Axis)', color='gray')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax1_twin.get_legend_handles_labels()
        ax.legend(handles=h1 + h2, labels=l1 + l2, loc='upper left', fontsize='small')

    # --- PLOT [0, 1]: FIGURE 2 ---
    ax = axes[0, 1]
    print("  Calculating weekly changes for Fig 2...")
    weekly_spread = daily_data['EFFR_Spread'].resample('W-WED').mean()
    weekly_ratio = weekly_data_raw['RESERVE_RATIO_PCT'].resample('W-WED').mean()
    
    final_data_var = pd.concat([weekly_ratio, weekly_spread], axis=1)
    final_data_var.columns = ['RESERVE_RATIO_PCT', 'SPREAD_BPS']
    var_data = final_data_var.dropna()
    var_data_diff = var_data.diff().dropna()
    
    y_ols = var_data_diff['SPREAD_BPS']
    X_ols = sm.add_constant(var_data_diff['RESERVE_RATIO_PCT'])
    
    colors = ['green', 'blue', 'red'] # Fast, Medium, Slow
    windows_to_plot = [13, 26, 52] 
    elasticity_data = {}
    
    for i, window in enumerate(windows_to_plot):
        print(f"  Running {window}-week RollingOLS...")
        rol_model = RollingOLS(y_ols, X_ols, window=window, min_nobs=window)
        rolling_results = rol_model.fit()
        elasticity_series = rolling_results.params['RESERVE_RATIO_PCT'].dropna() 
        elasticity_data[f'{window}-Wk Elasticity (W)'] = elasticity_series
        ax.plot(elasticity_series.index, elasticity_series, 
                 color=colors[i], label=f'{window}-Wk Elasticity', 
                 linewidth= (i*1)+1 )
    
    ax.set_title('Figure 2: System Regime (Rolling Elasticity of Changes)')
    ax.set_ylabel('Elasticity (bp change / pp change)', color='black')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize='small')
    
    # --- PLOT [1, 0]: FIGURE 3 ---
    ax = axes[1, 0]
    panel_b_data = daily_data['EFFR_Spread'].dropna()
    panel_b_ma = panel_b_data.rolling(window=30).mean()
    
    ax.plot(panel_b_data.index, panel_b_data, color='red', 
            label='Daily EFFR Spread', linewidth=0.5, alpha=0.6)
    ax.plot(panel_b_ma.index, panel_b_ma, color='red', 
            label='30-Day MA EFFR Spread', linewidth=2.5, linestyle='--')
    ax.set_title('Figure 3: Unsecured & Corporate Stress') 
    ax.set_ylabel('Spread (Basis Points) (Left Axis)', color='red')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(True, linestyle='--', alpha=0.6)

    if 'BAMLH0A0HYM2' in daily_data.columns:
        ax3_twin = ax.twinx()
        hy_spread_data = daily_data['BAMLH0A0HYM2'].dropna()
        ax3_twin.plot(hy_spread_data.index, hy_spread_data, 
                      color='purple', linestyle='--', label='High-Yield Spread (%)')
        ax3_twin.set_ylabel('High-Yield Spread (Bps, Right Axis)', color='purple')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax3_twin.get_legend_handles_labels()
        ax.legend(handles=h1 + h2, labels=l1 + l2, loc='upper left', fontsize='small')

    # --- PLOT [1, 1]: FIGURE 4 ---
    ax = axes[1, 1]
    plot_data_fig5 = daily_data[['EFFR_Spread', 'SOFR_Spread', 'RRP_Spread', 'DW_Spread']].dropna(how='all')

    effr_spread_ma = plot_data_fig5['EFFR_Spread'].rolling(window=30).mean()
    sofr_spread_ma = plot_data_fig5['SOFR_Spread'].rolling(window=30).mean()

    ax.plot(plot_data_fig5.index, plot_data_fig5['DW_Spread'], 
             label='Discount Window (Ceiling)', linestyle='--', color='gray')
    ax.plot(plot_data_fig5.index, plot_data_fig5['RRP_Spread'], 
             label='ON RRP (Floor)', linestyle='--', color='blue')
    ax.plot(plot_data_fig5.index, plot_data_fig5['EFFR_Spread'], 
             label='EFFR (Daily)', linewidth=0.5, color='green', alpha=0.6)
    ax.plot(plot_data_fig5.index, plot_data_fig5['SOFR_Spread'], 
             label='SOFR (Daily)', linewidth=0.5, color='red', alpha=0.6)
    ax.plot(effr_spread_ma.index, effr_spread_ma, 
             label='EFFR (30-Day MA)', linewidth=2.5, color='green', linestyle='--')
    ax.plot(sofr_spread_ma.index, sofr_spread_ma, 
             label='SOFR (30-Day MA)', linewidth=2.5, color='red', linestyle='--')
             
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.set_title('Figure 4: Secured Market Stress (Repo Corridor)') 
    ax.set_ylabel('Spread (Basis Points) (Left Axis)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize='small')
    
    
    # --- 5. Final Formatting for All Plots ---
    print("  Applying background shading and asset overlays...")
    
    asset_legend_handles = []
    
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            
            bottom, top = ax.get_ylim()
            
            if dynamic_shading_available:
                ax.fill_between(drawdown.index, bottom, top, where=in_correction, 
                                facecolor=correction_color, alpha=0.2, zorder=0)
                ax.fill_between(drawdown.index, bottom, top, where=in_decline, 
                                facecolor=decline_color, alpha=0.2, zorder=0)

            ax.axvspan(repo_spike_2019[0], repo_spike_2019[1], alpha=0.3, color=repo_spike_color, zorder=0)
            
            ax.xaxis.set_major_locator(year_locator)
            ax.xaxis.set_major_formatter(date_formatter)
            ax.xaxis.set_minor_locator(month_locator)
            ax.set_xlim(start_date, end_date)
            ax.set_xlabel('Date')
            
            if asset_data_fetched and i == 0:
                ax_twin = ax.twinx()
                h1, h2, h3, h4, h5 = None, None, None, None, None
                
                if '^GSPC' in asset_norm.columns:
                    h1, = ax_twin.plot(asset_norm.index, asset_norm['^GSPC'], 
                                 label='S&P 500', color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                if '^IXIC' in asset_norm.columns:
                    h2, = ax_twin.plot(asset_norm.index, asset_norm['^IXIC'], 
                                 label='Nasdaq', color='cyan', linestyle=':', linewidth=1.5, alpha=0.7)
                if '^DJI' in asset_norm.columns:
                    h4, = ax_twin.plot(asset_norm.index, asset_norm['^DJI'],
                                 label='Dow Jones', color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
                if '^RUT' in asset_norm.columns:
                    h5, = ax_twin.plot(asset_norm.index, asset_norm['^RUT'],
                                 label='Russell 2000', color='magenta', linestyle=':', linewidth=1.5, alpha=0.7)
                if 'BTC-USD' in asset_norm.columns:
                    h3, = ax_twin.plot(asset_norm.index, asset_norm['BTC-USD'], 
                                 label='BTC', color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
                
                if not asset_legend_handles:
                    asset_legend_handles = [h for h in [h1, h2, h4, h5, h3] if h is not None]
                
                ax_twin.set_ylabel('Normalized Price (Log Scale, Right Axis)')
                ax_twin.set_yscale('log')
                ax_twin.spines['right'].set_visible(True)
                
                if j == 0:
                    ax1_twin.set_ylabel('') 
                    ax1_twin.set_yticklabels([])
                if j == 1:
                    ax.get_legend().remove() 
                    
            ax.set_ylim(bottom, top)

    # --- 6. CREATE THE COMMON LEGEND ---
    print("  Creating common legend...")
    
    patch_decline = mpatches.Patch(color=decline_color, alpha=0.2, label='Decline (-20%)')
    patch_correction = mpatches.Patch(color=correction_color, alpha=0.2, label='Correction (-10%)')
    patch_repo_spike = mpatches.Patch(color=repo_spike_color, alpha=0.3, label='Repo Spike 2019')
    shading_handles = [patch_decline, patch_correction, patch_repo_spike]

    all_handles = asset_legend_handles + shading_handles
    all_labels = [h.get_label() for h in asset_legend_handles] + [h.get_label() for h in shading_handles]
    
    fig.legend(handles=all_handles, labels=all_labels, 
               loc='lower center', ncol=8, fontsize='medium') 
    
    if asset_data_fetched:
        h1_main, l1_main = axes[0, 1].get_legend_handles_labels()
        axes[0, 1].legend(handles=h1_main + asset_legend_handles, 
                       labels=l1_main + [h.get_label() for h in asset_legend_handles], 
                       loc='upper left', fontsize='small')

    
    fig.tight_layout(pad=3.0, rect=[0, 0.05, 1, 0.95]) 
    fig.suptitle('Monetary Analysis Dashboard: Liquidity, Regime, and Stress (from 2018)', fontsize=24, y=1.0)
    
    # *** --- 7. SAVE PLOT TO FILE --- ***
    # Instead of plt.show(), we save the figure
    print("\nSaving dashboard.png...")
    fig.savefig('dashboard.png', dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the plot to free up memory

    
    # --- 8. Generate and Save Table Files ---
    print("Generating final data table...")
    
    # Combine all the data sources
    elasticity_df = pd.DataFrame(elasticity_data)
    combined_df = pd.concat([daily_data, weekly_data_raw, elasticity_df], axis=1)
    
    combined_df['30-Day MA EFFR Spread (Bps)'] = combined_df['EFFR_Spread'].rolling(30).mean()
    combined_df['30-Day MA SOFR Spread (Bps)'] = combined_df['SOFR_Spread'].rolling(30).mean()
    
    weekly_cols = ['RESERVE_RATIO_PCT', '13-Wk Elasticity (W)', '26-Wk Elasticity (W)', '52-Wk Elasticity (W)']
    combined_df[weekly_cols] = combined_df[weekly_cols].ffill()
    
    final_columns_original = [
        'RRPONTSYD_B', 'BAMLH0A0HYM2', 'EFFR_Spread', '30-Day MA EFFR Spread (Bps)',
        'SOFR_Spread', '30-Day MA SOFR Spread (Bps)', 'DW_Spread', 'RRP_Spread',
        'RESERVE_RATIO_PCT', '13-Wk Elasticity (W)', '26-Wk Elasticity (W)', '52-Wk Elasticity (W)'
    ]
    
    table_data = combined_df.reindex(columns=final_columns_original)
    final_table_recent = table_data.iloc[-14:]
    
    final_table_recent = final_table_recent.rename(columns={
        'RRPONTSYD_B': 'ON RRP Usage ($B)', 'BAMLH0A0HYM2': 'High-Yield Spread (%)',
        'EFFR_Spread': 'Daily EFFR Spread (Bps)', 'SOFR_Spread': 'Daily SOFR Spread (Bps)',
        'DW_Spread': 'DW Spread (Bps)', 'RRP_Spread': 'RRP Spread (Bps)',
        'RESERVE_RATIO_PCT': 'Reserves / Bank Assets (W) (%)'
    })
    
    # Save to Excel
    excel_filename = 'dashboard_recent_data.xlsx'
    try:
        final_table_recent.to_excel(excel_filename, float_format="%.2f")
        print(f"Successfully saved recent data to {excel_filename}")
    except Exception as e:
        print(f"\nERROR: Could not save Excel file. Error: {e}")
        print("Make sure you have 'openpyxl' installed: pip install openpyxl")

    # --- 9. NEW: Generate index.html file ---
    print("Generating index.html...")
    
    # Get the HTML for the table
    table_html = final_table_recent.to_html(float_format="%.2f", border=0, classes='dataframe')
    
    # Get the update time (in UTC for the server)
    update_time = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M %Z')
    
    # Simple HTML template with some basic styling
    html_template = f"""
    <html>
    <head>
        <title>Monetary Analysis Dashboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                   background-color: #f4f4f4; color: #333; margin: 0; padding: 20px; }}
            h1, h2 {{ text-align: center; color: #111; }}
            p {{ text-align: center; font-style: italic; color: #555; }}
            .container {{ max-width: 1200px; margin: auto; background-color: #fff;
                          border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            img {{ width: 100%; height: auto; border-radius: 8px 8px 0 0; }}
            .table-container {{ padding: 20px; overflow-x: auto; }}
            table.dataframe {{ width: 100%; border-collapse: collapse; text-align: right; }}
            table.dataframe th {{ background-color: #f8f8f8; padding: 8px; border-bottom: 2px solid #ddd; }}
            table.dataframe td {{ padding: 8px; border-bottom: 1px solid #eee; }}
            table.dataframe tbody tr:hover {{ background-color: #f1f1f1; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Monetary Analysis Dashboard</h1>
            <p>Last updated: {update_time}</p>
            <img src="dashboard.png" alt="Monetary Analysis Dashboard">
            <div class="table-container">
                <h2>Recent Data</h2>
                {table_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    try:
        with open('index.html', 'w', encoding='utf-8') as f:
            f.write(html_template)
        print("Successfully saved index.html")
    except Exception as e:
        print(f"ERROR: Could not save index.html. Error: {e}")
    
    print("\n--- All tasks complete. ---")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure pandas-datareader, matplotlib, statsmodels, yfinance, and openpyxl are installed.")
    print("e.g.: pip install --upgrade pandas-datareader matplotlib statsmodels yfinance openpyxl")