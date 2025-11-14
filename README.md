Monetary Dashboard (Updated Daily)

This repository automatically generates and updates a comprehensive 5-part dashboard of key financial and economic indicators every day using GitHub Actions.

Live Dashboard

The live, interactive dashboard is available at:

https://seungho-an.github.io/monetary-dashboard/

Overview

This project runs the master_dashboard.py script daily to:

Fetch the latest data for 17+ indicators from FRED and Yahoo Finance.

Analyze the data to determine the current risk level ("status dots").

Generate 5 dashboard images.

Generate an automated summary report (dashboard_summary_report.txt).

Build and deploy the index.html page for the live site.
