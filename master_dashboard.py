... (섹션 1부터 8까지 모든 기존 코드) ...
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
