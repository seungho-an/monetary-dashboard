This dashboard tries to replicate a couple of figures from Afonso et al. 2025 (https://www.newyorkfed.org/research/staff_reports/sr1019): Figures 2, 4, and 5 using FRED data. Given that FRED has only weekly data for a couple of key variables, figures are not exactly the same. Due to the data limitation and for simplicity, the forecast of reserve elasticity employs the simple OLS approach. The files are updated automatically daily. 

For the measure/interpretation, I asked Gemini to create a document (see below).
## ⚠️ Figure 1: Bank Reserve Levels (Liquidity)
 

What It Measures (Intent): This chart is our high-level "gas tank" for the financial system. It shows the total amount of liquidity (cash) and tells us if the Fed is pumping money in (Quantitative Easing) or draining it out (Quantitative Tightening).

How It's Generated (Logic):

Blue Line (Reserves / Bank Assets): We take total bank reserves (WRESBAL) and divide by total bank assets (TLAACBW027SBOG). We use a percentage, not a raw number, because it shows how much "cash" banks have relative to their size. A high number means banks are very safe and liquid.

Grey Area (ON RRP Usage): We plot the total value of the Fed's Overnight Reverse Repo facility (RRPONTSYD). This represents a pool of "excess cash" held by non-banks (like money market funds) parked at the Fed.

Grey Bar (Scarcity Zone): This is a static bar we draw between 12-13% as a visual guide, based on estimates of where the "reserve scarcity" level might be.

 

## ⚠️ Figure 2: System Regime (Rolling Elasticity)
 

What It Measures (Intent): This is our "fragility" or "regime" indicator. It's the most complex chart and answers the question: "Is the system 'abundant' in cash or 'scarce'?"

Positive Line: The system is abundant. There is so much cash that adding or removing some doesn't impact rates. The system is stable.

Negative Line: The system is scarce. The "buffer" is gone, and any small change in reserves does impact rates. The system is fragile and vulnerable to spikes.

How It's Generated (Logic):

First, we get the weekly change in two data series:

The change in the Reserve Ratio (from Figure 1).

The change in the EFFR Spread (from Figure 3).

We then run a rolling OLS regression (a statistical analysis) to find the correlation between these two changes over time (13, 26, and 52-week windows).

This plot shows the slope of that relationship. A negative slope (negative line) means a decrease in reserves is correlated with an increase in spreads—this is a classic fragile, supply-and-demand "scarce" regime.

 

## 🚨 Figure 3: Unsecured & Corporate Stress
 

What It Measures (Intent): This is our "real-time fire alarm" for trust-based markets. It shows if banks are panicking (by refusing to lend to each other) or if the market is panicking about corporations defaulting.

How It's Generated (Logic):

Red Lines (EFFR Spreads): We calculate EFFR (the actual rate banks charge each other) minus the Policy Rate (the Fed's target). A spike here means banks are charging each other more than the Fed's target, which is a sign of banking system panic. The dashed line is a 30-day average to show "chronic" stress.

Purple Line (High-Yield Spread): We plot the BAMLH0A0HYM2 series directly. This is the "junk bond" spread, or the extra interest investors demand to hold risky corporate bonds. A spike here means investors are panicking about corporate defaults and the real economy.

 

## 🚨  Figure 4: Secured Market Stress (Repo Corridor)
 

What It Measures (Intent): This is our "financial plumbing" alarm. It shows stress in the secured (collateral-based) repo market, which is the essential, core plumbing of the entire financial system.

How It's Generated (Logic):

Floor & Ceiling (Blue/Grey Lines): We create the "policy corridor" by plotting the RRP Rate (the floor) and Discount Window Rate (the ceiling) relative to the main Policy Rate.

Green/Red Lines (Market Rates): We plot the key market rates (EFFR and SOFR) relative to the Policy Rate.

The Goal: In normal times, the market rates (green/red) should trade calmly inside the corridor, near the floor. A "crisis" (like the 2019 Repo Spike) is when a market rate violently spikes above the floor, showing a severe, acute shortage of cash in the system's core plumbing. The dashed lines show the 30-day average of this stress.
