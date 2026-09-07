# AI Financial Health Review Assistant

An analyst-style financial analysis tool that converts standardized company financial data into key financial metrics, material findings, visualizations, and AI-assisted interpretation.

**🚀 Deployed Application:**  
https://dyuksha27-ai-financial-health-review-assistant-app-kasbpe.streamlit.app/

## Features

- Analyzes 5-year Income Statement, Balance Sheet, and Cash Flow data
- Calculates profitability, liquidity, leverage, working capital, and cash flow metrics
- Supports common differences in financial line-item terminology through flexible metric mapping
- Detects material financial trends using a deterministic rules engine
- Generates interactive financial charts
- Uses AI to interpret material findings, identify possible drivers and risks, and suggest areas for further investigation
- Separates deterministic financial calculations from AI-generated interpretation

## Input Format

The application accepts an Excel workbook containing a `Raw_Data` sheet in the following format:

| Statement | Metric | FY2022 | FY2023 | FY2024 | FY2025 | FY2026 |
|---|---|---:|---:|---:|---:|---:|
| Income Statement | Revenue | ... | ... | ... | ... | ... |
| Balance Sheet | Accounts Receivable | ... | ... | ... | ... | ... |
| Cash Flow | Cash Flow from Operations (CFO) | ... | ... | ... | ... | ... |

Financial data must first be extracted from the company's financial statements and converted into this standardized structure.

A Microsoft sample dataset is included in `sample_data/`.

## Analysis Pipeline

Financial Data → Metric Standardization → Financial Analysis → Rules-Based Finding Detection → AI Interpretation

## Tech Stack

Python · Pandas · Streamlit · Plotly · OpenRouter API

## Disclaimer

This project is intended for financial analysis and educational purposes only and does not provide investment advice.
