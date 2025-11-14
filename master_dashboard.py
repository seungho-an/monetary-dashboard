import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg') # <-- ADDED: Use 'Agg' backend for non-GUI server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import datetime
from fredapi import Fred  # Use FRED API
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings
import time
import os # <-- Import for GitHub Actions API Key

# --- 0. Suppress Warnings ---
# ... (rest of the script is identical to the one in your context) ...
# ...
# --- 1. STATUS INDICATOR DEFINITIONS ---
# ...
# ... (all functions and processing logic remain the same) ...
# ...
# --- 7. MAIN EXECUTION BLOCK ---
print("--- Generating All Dashboards ---")

# Pack data for plotting functions
shading_data = {
# ... (all plotting logic remains the same) ...
# ...
print("Generating and saving Dashboard 5 (Consumer & Risk Appetite)...")
fig5 = plot_earnings_consumer_dashboard(df_earnings_consumer, df_market_risk, status_results)
fig5.savefig('dashboard_5_earnings_consumer.png', dpi=150, bbox_inches='tight')


# --- Show plots *after* saving ---
print("\nAll dashboards generated and saved.")
# *** MODIFICATION: plt.show() is commented out for server execution ***
# print("Displaying figures... (Close all figure windows to continue to the text report)")
# plt.show() 

# --- Close figures to save memory ---
print("Closing figures to save memory...")
plt.close('all') # Close all figures
# ...
# ... (rest of the script is identical) ...
# ...
print("\n--- Master Dashboard Script TRULY Finished ---")
