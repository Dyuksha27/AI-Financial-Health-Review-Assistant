# ============================================================
# AI FINANCIAL HEALTH REVIEW ASSISTANT
# ============================================================

import json
import re
from difflib import get_close_matches

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from plotly.subplots import make_subplots


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Financial Health Review",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Financial Health Review Assistant")

st.caption(
    "Deterministic financial analysis + material finding detection "
    "+ AI-assisted analyst interpretation."
)


# ============================================================
# ALIAS DICTIONARY
# ============================================================

METRIC_ALIASES = {

    # ---------- INCOME STATEMENT ----------

    "Revenue": [
        "revenue",
        "revenues",
        "total revenue",
        "total revenues",
        "net revenue",
        "net revenues",
        "net sales",
        "sales",
        "total sales",
    ],

    "Gross Profit": [
        "gross profit",
        "gross income",
    ],

    "Operating Income": [
        "operating income",
        "operating profit",
        "income from operations",
        "profit from operations",
        "ebit",
    ],

    "Net Income": [
        "net income",
        "net earnings",
        "net profit",
        "profit after tax",
        "profit for the year",
    ],

    # ---------- BALANCE SHEET ----------

    "Cash & Cash Equivalents": [
        "cash and cash equivalents",
        "cash & cash equivalents",
        "cash equivalents",
        "cash",
    ],

    "Accounts Receivable": [
        "accounts receivable",
        "accounts receivable net",
        "trade receivables",
        "trade accounts receivable",
        "receivables",
    ],

    "Inventory": [
        "inventory",
        "inventories",
    ],

    "Accounts Payable": [
        "accounts payable",
        "trade payables",
        "trade accounts payable",
        "payables",
    ],

    "Total Current Assets": [
        "total current assets",
        "current assets",
    ],

    "Total Current Liabilities": [
        "total current liabilities",
        "current liabilities",
    ],

    "Short-Term Debt": [
        "short-term debt",
        "short term debt",
        "short-term borrowings",
        "short term borrowings",
        "current borrowings",
    ],

    "Current Portion of Long-Term Debt": [
        "current portion of long-term debt",
        "current portion of long term debt",
        "current maturities of long-term debt",
        "current maturities of long term debt",
    ],

    "Long-Term Debt": [
        "long-term debt",
        "long term debt",
        "non-current debt",
        "noncurrent debt",
        "long-term borrowings",
        "long term borrowings",
    ],

    "Total Stockholders' Equity": [
        "total stockholders' equity",
        "total stockholders equity",
        "stockholders' equity",
        "shareholders' equity",
        "shareholders equity",
        "total shareholders' equity",
        "total equity",
    ],

    # ---------- CASH FLOW ----------

    "Cash Flow from Operations (CFO)": [
        "cash flow from operations",
        "cash flows from operations",
        "cash from operations",
        "operating cash flow",
        "net cash from operating activities",
        "net cash provided by operating activities",
    ],

    "CapEx": [
        "capex",
        "capital expenditures",
        "capital expenditure",
        "purchases of property plant and equipment",
        "purchases of property, plant and equipment",
        "purchase of property plant and equipment",
        "additions to property plant and equipment",
    ],

    # ---------- EBITDA INPUTS ----------

    "Reported EBITDA": [
        "ebitda",
        "reported ebitda",
        "adjusted ebitda",
    ],

    "D&A": [
        "depreciation and amortization",
        "depreciation & amortization",
        "depreciation and amortisation",
        "d&a",
    ],

    "Depreciation": [
        "depreciation",
        "depreciation expense",
    ],

    "Amortization": [
        "amortization",
        "amortisation",
        "amortization expense",
    ],

    "D&A and Other": [
        "d&a and other",
        "depreciation amortization and other",
        "depreciation, amortization, and other",
        "depreciation amortisation and other",
    ],
}


# ============================================================
# NORMALIZATION / MAPPING
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = value.replace("&", "and")

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


ALIAS_LOOKUP = {}

for standard_metric, aliases in METRIC_ALIASES.items():

    ALIAS_LOOKUP[
        normalize_text(standard_metric)
    ] = standard_metric

    for alias in aliases:

        ALIAS_LOOKUP[
            normalize_text(alias)
        ] = standard_metric


def map_metric_name(metric_name):

    normalized = normalize_text(metric_name)

    if normalized in ALIAS_LOOKUP:

        return (
            ALIAS_LOOKUP[normalized],
            "Exact / Alias",
            1.0
        )

    matches = get_close_matches(
        normalized,
        list(ALIAS_LOOKUP.keys()),
        n=1,
        cutoff=0.88
    )

    if matches:

        matched = matches[0]

        return (
            ALIAS_LOOKUP[matched],
            "Fuzzy",
            0.88
        )

    return (
        metric_name,
        "Unmapped",
        0.0
    )


def standardize_financial_data(df):

    df = df.copy()

    df["Original Metric"] = df["Metric"]

    mapped = df[
        "Original Metric"
    ].apply(map_metric_name)

    df["Metric"] = mapped.apply(
        lambda x: x[0]
    )

    df["Mapping Method"] = mapped.apply(
        lambda x: x[1]
    )

    df["Mapping Confidence"] = mapped.apply(
        lambda x: x[2]
    )

    return df


# ============================================================
# STATEMENT STANDARDIZATION
# ============================================================

def standardize_statement_name(statement):

    value = normalize_text(statement)

    if any(
        phrase in value
        for phrase in [
            "income statement",
            "statement of operations",
            "statement of income",
            "operations",
        ]
    ):
        return "Income Statement"

    if any(
        phrase in value
        for phrase in [
            "balance sheet",
            "financial position",
        ]
    ):
        return "Balance Sheet"

    if any(
        phrase in value
        for phrase in [
            "cash flow",
            "cash flows",
        ]
    ):
        return "Cash Flow"

    return statement


# ============================================================
# DATA CLEANING
# ============================================================

def clean_financial_data(df):

    df = df.copy()

    df["Statement"] = df[
        "Statement"
    ].apply(standardize_statement_name)

    year_columns = [
        col
        for col in df.columns
        if re.fullmatch(
            r"FY\d{4}",
            str(col)
        )
    ]

    for column in year_columns:

        # Handle accounting negatives like (1,234)

        cleaned = (
            df[column]
            .astype(str)
            .str.replace(
                ",",
                "",
                regex=False
            )
            .str.replace(
                "$",
                "",
                regex=False
            )
            .str.strip()
        )

        cleaned = cleaned.str.replace(
            r"^\((.*)\)$",
            r"-\1",
            regex=True
        )

        df[column] = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

    return df


# ============================================================
# YEAR DETECTION
# ============================================================

def get_year_columns(df):

    years = [
        str(col)
        for col in df.columns
        if re.fullmatch(
            r"FY\d{4}",
            str(col)
        )
    ]

    return sorted(
        years,
        key=lambda x: int(x[2:])
    )


# ============================================================
# METRIC LOOKUP
# ============================================================

def get_metric(
    df,
    metric_name,
    statement=None
):

    if statement is None:

        result = df[
            df["Metric"] == metric_name
        ]

    else:

        result = df[
            (df["Metric"] == metric_name)
            &
            (df["Statement"] == statement)
        ]

    if result.empty:

        if statement:

            raise ValueError(
                f"Required metric not found: "
                f"{statement} → {metric_name}"
            )

        raise ValueError(
            f"Required metric not found: {metric_name}"
        )

    return result.iloc[0]


# ============================================================
# EBITDA LOGIC
# ============================================================

def determine_ebitda_input(df):

    available = set(
        df["Metric"].dropna()
    )

    if "Reported EBITDA" in available:

        return {
            "type": "Reported EBITDA",
            "source": "Reported EBITDA",
            "quality": "Reported",
        }

    if "D&A" in available:

        return {
            "type": "EBITDA",
            "source": "D&A",
            "quality": "Calculated",
        }

    if (
        "Depreciation" in available
        and
        "Amortization" in available
    ):

        return {
            "type": "EBITDA",
            "source": "Separate D&A",
            "quality": "Calculated",
        }

    if "D&A and Other" in available:

        return {
            "type": "EBITDA Proxy",
            "source": "D&A and Other",
            "quality": "Proxy",
        }

    return {
        "type": "Unavailable",
        "source": None,
        "quality": "Unavailable",
    }


# ============================================================
# FINANCIAL ENGINE
# ============================================================

def calculate_metrics(df, years):

    # ---------- Income Statement ----------

    revenue = get_metric(
        df,
        "Revenue",
        "Income Statement"
    )

    gross_profit = get_metric(
        df,
        "Gross Profit",
        "Income Statement"
    )

    operating_income = get_metric(
        df,
        "Operating Income",
        "Income Statement"
    )

    net_income = get_metric(
        df,
        "Net Income",
        "Income Statement"
    )

    # ---------- Balance Sheet ----------

    cash = get_metric(
        df,
        "Cash & Cash Equivalents",
        "Balance Sheet"
    )

    ar = get_metric(
        df,
        "Accounts Receivable",
        "Balance Sheet"
    )

    inventory = get_metric(
        df,
        "Inventory",
        "Balance Sheet"
    )

    ap = get_metric(
        df,
        "Accounts Payable",
        "Balance Sheet"
    )

    current_assets = get_metric(
        df,
        "Total Current Assets",
        "Balance Sheet"
    )

    current_liabilities = get_metric(
        df,
        "Total Current Liabilities",
        "Balance Sheet"
    )

    short_term_debt = get_metric(
        df,
        "Short-Term Debt",
        "Balance Sheet"
    )

    current_ltd = get_metric(
        df,
        "Current Portion of Long-Term Debt",
        "Balance Sheet"
    )

    long_term_debt = get_metric(
        df,
        "Long-Term Debt",
        "Balance Sheet"
    )

    equity = get_metric(
        df,
        "Total Stockholders' Equity",
        "Balance Sheet"
    )

    # ---------- Cash Flow ----------

    cfo = get_metric(
        df,
        "Cash Flow from Operations (CFO)",
        "Cash Flow"
    )

    capex = get_metric(
        df,
        "CapEx",
        "Cash Flow"
    )

    ebitda_info = determine_ebitda_input(df)

    ebitda_source_row = None
    depreciation = None
    amortization = None

    if ebitda_info["source"] == "Reported EBITDA":

        ebitda_source_row = get_metric(
            df,
            "Reported EBITDA"
        )

    elif ebitda_info["source"] == "D&A":

        ebitda_source_row = get_metric(
            df,
            "D&A"
        )

    elif ebitda_info["source"] == "D&A and Other":

        ebitda_source_row = get_metric(
            df,
            "D&A and Other"
        )

    elif ebitda_info["source"] == "Separate D&A":

        depreciation = get_metric(
            df,
            "Depreciation"
        )

        amortization = get_metric(
            df,
            "Amortization"
        )


    metrics = [

        "Revenue",
        "Revenue Growth",

        "Gross Profit",
        "Gross Margin",

        "EBIT",
        "EBIT Margin",

        "EBITDA",
        "EBITDA Margin",

        "Net Income",
        "Net Margin",
        "Net Income Growth",

        "Current Ratio",
        "Quick Ratio",

        "Accounts Receivable",
        "AR Growth",

        "Inventory",
        "Inventory Growth",

        "Accounts Payable",
        "AP Growth",

        "Operating Working Capital",

        "Operating Cash Flow",

        "CapEx",
        "CapEx / Revenue",

        "FCF",
        "FCF Margin",
        "FCF Conversion",

        "Cash",

        "Total Debt",
        "Debt / Equity",
        "Debt / EBITDA",

        "Net Debt",
        "Net Debt / EBITDA",

        "ROE",
    ]

    results = {
        metric: {}
        for metric in metrics
    }


    for i, year in enumerate(years):

        # ----------------------------------------------------
        # Raw values
        # ----------------------------------------------------

        results["Revenue"][year] = revenue[year]

        results[
            "Gross Profit"
        ][year] = gross_profit[year]

        results[
            "EBIT"
        ][year] = operating_income[year]

        results[
            "Net Income"
        ][year] = net_income[year]

        results[
            "Accounts Receivable"
        ][year] = ar[year]

        results[
            "Inventory"
        ][year] = inventory[year]

        results[
            "Accounts Payable"
        ][year] = ap[year]

        results[
            "Operating Cash Flow"
        ][year] = cfo[year]

        results[
            "CapEx"
        ][year] = abs(capex[year])

        results[
            "Cash"
        ][year] = cash[year]


        # ----------------------------------------------------
        # Margins
        # ----------------------------------------------------

        results[
            "Gross Margin"
        ][year] = (
            gross_profit[year]
            / revenue[year]
        ) * 100

        results[
            "EBIT Margin"
        ][year] = (
            operating_income[year]
            / revenue[year]
        ) * 100

        results[
            "Net Margin"
        ][year] = (
            net_income[year]
            / revenue[year]
        ) * 100


        # ----------------------------------------------------
        # EBITDA
        # ----------------------------------------------------

        if ebitda_info["source"] == "Reported EBITDA":

            ebitda = (
                ebitda_source_row[year]
            )

        elif ebitda_info["source"] in [
            "D&A",
            "D&A and Other"
        ]:

            ebitda = (
                operating_income[year]
                + ebitda_source_row[year]
            )

        elif ebitda_info["source"] == "Separate D&A":

            ebitda = (
                operating_income[year]
                + depreciation[year]
                + amortization[year]
            )

        else:

            ebitda = np.nan


        results[
            "EBITDA"
        ][year] = ebitda

        results[
            "EBITDA Margin"
        ][year] = (
            (ebitda / revenue[year]) * 100
            if pd.notna(ebitda)
            else np.nan
        )


        # ----------------------------------------------------
        # Liquidity
        # ----------------------------------------------------

        results[
            "Current Ratio"
        ][year] = (
            current_assets[year]
            / current_liabilities[year]
        )

        results[
            "Quick Ratio"
        ][year] = (
            (
                current_assets[year]
                - inventory[year]
            )
            / current_liabilities[year]
        )


        # ----------------------------------------------------
        # Debt
        # ----------------------------------------------------

        total_debt = (
            short_term_debt[year]
            + current_ltd[year]
            + long_term_debt[year]
        )

        results[
            "Total Debt"
        ][year] = total_debt

        results[
            "Debt / Equity"
        ][year] = (
            total_debt
            / equity[year]
        )

        results[
            "Debt / EBITDA"
        ][year] = (
            total_debt / ebitda
            if pd.notna(ebitda)
            and ebitda != 0
            else np.nan
        )


        net_debt = (
            total_debt
            - cash[year]
        )

        results[
            "Net Debt"
        ][year] = net_debt

        results[
            "Net Debt / EBITDA"
        ][year] = (
            net_debt / ebitda
            if pd.notna(ebitda)
            and ebitda != 0
            else np.nan
        )


        # ----------------------------------------------------
        # Working Capital
        # ----------------------------------------------------

        results[
            "Operating Working Capital"
        ][year] = (
            ar[year]
            + inventory[year]
            - ap[year]
        )


        # ----------------------------------------------------
        # Free Cash Flow
        # ----------------------------------------------------

        # Raw CapEx is expected as a cash outflow / negative number.

        fcf = (
            cfo[year]
            + capex[year]
        )

        results[
            "FCF"
        ][year] = fcf

        results[
            "FCF Margin"
        ][year] = (
            fcf
            / revenue[year]
        ) * 100

        results[
            "FCF Conversion"
        ][year] = (
            fcf
            / net_income[year]
        ) * 100

        results[
            "CapEx / Revenue"
        ][year] = (
            abs(capex[year])
            / revenue[year]
        ) * 100


        # ----------------------------------------------------
        # Growth metrics
        # ----------------------------------------------------

        if i == 0:

            results[
                "Revenue Growth"
            ][year] = np.nan

            results[
                "Net Income Growth"
            ][year] = np.nan

            results[
                "AR Growth"
            ][year] = np.nan

            results[
                "Inventory Growth"
            ][year] = np.nan

            results[
                "AP Growth"
            ][year] = np.nan

            results[
                "ROE"
            ][year] = np.nan

        else:

            previous_year = years[i - 1]

            results[
                "Revenue Growth"
            ][year] = (
                (
                    revenue[year]
                    / revenue[previous_year]
                )
                - 1
            ) * 100

            results[
                "Net Income Growth"
            ][year] = (
                (
                    net_income[year]
                    / net_income[previous_year]
                )
                - 1
            ) * 100

            results[
                "AR Growth"
            ][year] = (
                (
                    ar[year]
                    / ar[previous_year]
                )
                - 1
            ) * 100

            results[
                "Inventory Growth"
            ][year] = (
                (
                    inventory[year]
                    / inventory[previous_year]
                )
                - 1
            ) * 100

            results[
                "AP Growth"
            ][year] = (
                (
                    ap[year]
                    / ap[previous_year]
                )
                - 1
            ) * 100


            average_equity = (
                equity[year]
                + equity[previous_year]
            ) / 2

            results[
                "ROE"
            ][year] = (
                net_income[year]
                / average_equity
            ) * 100


    analysis = pd.DataFrame(results).T

    analysis = analysis[years]


    analysis.attrs[
        "ebitda_type"
    ] = ebitda_info["type"]

    analysis.attrs[
        "ebitda_source"
    ] = ebitda_info["source"]

    analysis.attrs[
        "ebitda_quality"
    ] = ebitda_info["quality"]

    return analysis


# ============================================================
# FINDINGS ENGINE
# ============================================================

def generate_findings(
    analysis,
    years
):

    findings = []

    current_year = years[-1]
    previous_year = years[-2]


    # ---------- Revenue Growth ----------

    current = analysis.loc[
        "Revenue Growth",
        current_year
    ]

    previous = analysis.loc[
        "Revenue Growth",
        previous_year
    ]

    change = current - previous

    severity = (
        "High"
        if abs(change) >= 10
        else "Medium"
        if abs(change) >= 5
        else "Low"
    )

    findings.append({

        "Finding":
            "Revenue Growth",

        "Metric":
            "Revenue Growth",

        "Current":
            current,

        "Previous / Comparator":
            previous,

        "Change":
            change,

        "Severity":
            severity,

        "Investigation":
            "Assess drivers of revenue growth and key segment contributions."
    })


    # ---------- Gross Margin ----------

    current = analysis.loc[
        "Gross Margin",
        current_year
    ]

    previous = analysis.loc[
        "Gross Margin",
        previous_year
    ]

    change = current - previous

    severity = (
        "High"
        if abs(change) >= 3
        else "Medium"
        if abs(change) >= 1
        else "Low"
    )

    findings.append({

        "Finding":
            "Margin Compression",

        "Metric":
            "Gross Margin",

        "Current":
            current,

        "Previous / Comparator":
            previous,

        "Change":
            change,

        "Severity":
            severity,

        "Investigation":
            "Investigate cost-of-revenue growth and gross-margin drivers."
    })


    # ---------- Receivables ----------

    current = analysis.loc[
        "AR Growth",
        current_year
    ]

    previous = analysis.loc[
        "AR Growth",
        previous_year
    ]

    change = current - previous

    severity = (
        "High"
        if abs(change) >= 15
        else "Medium"
        if abs(change) >= 7.5
        else "Low"
    )

    findings.append({

        "Finding":
            "Receivables Growth",

        "Metric":
            "AR Growth",

        "Current":
            current,

        "Previous / Comparator":
            previous,

        "Change":
            change,

        "Severity":
            severity,

        "Investigation":
            "Compare receivables growth with revenue growth and review collection trends."
    })


    # ---------- FCF Conversion ----------

    current = analysis.loc[
        "FCF Conversion",
        current_year
    ]

    previous = analysis.loc[
        "FCF Conversion",
        previous_year
    ]

    change = current - previous

    severity = (
        "High"
        if abs(change) >= 15
        else "Medium"
        if abs(change) >= 7.5
        else "Low"
    )

    findings.append({

        "Finding":
            "Cash Conversion",

        "Metric":
            "FCF Conversion",

        "Current":
            current,

        "Previous / Comparator":
            previous,

        "Change":
            change,

        "Severity":
            severity,

        "Investigation":
            "Investigate CapEx and working-capital movements affecting free-cash-flow conversion."
    })


    # ---------- Leverage ----------

    current = analysis.loc[
        "Debt / EBITDA",
        current_year
    ]

    previous = analysis.loc[
        "Debt / EBITDA",
        previous_year
    ]

    change = current - previous

    severity = (
        "High"
        if abs(change) >= 1
        else "Medium"
        if abs(change) >= 0.5
        else "Low"
    )

    findings.append({

        "Finding":
            "Leverage",

        "Metric":
            "Debt / EBITDA",

        "Current":
            current,

        "Previous / Comparator":
            previous,

        "Change":
            change,

        "Severity":
            severity,

        "Investigation":
            "Assess debt trends, repayment capacity, and leverage relative to operating performance."
    })


    # ---------- Earnings vs FCF ----------

    ni_growth = analysis.loc[
        "Net Income Growth",
        current_year
    ]

    current_fcf = analysis.loc[
        "FCF",
        current_year
    ]

    previous_fcf = analysis.loc[
        "FCF",
        previous_year
    ]

    fcf_growth = (
        (
            current_fcf
            - previous_fcf
        )
        / abs(previous_fcf)
    ) * 100

    divergence = (
        ni_growth
        - fcf_growth
    )

    severity = (
        "High"
        if abs(divergence) >= 20
        else "Medium"
        if abs(divergence) >= 10
        else "Low"
    )

    findings.append({

        "Finding":
            "Earnings–Cash Flow Divergence",

        "Metric":
            "Net Income Growth vs FCF Growth",

        "Current":
            ni_growth,

        "Previous / Comparator":
            fcf_growth,

        "Change":
            divergence,

        "Severity":
            severity,

        "Investigation":
            "Investigate CapEx and working-capital movements driving divergence between earnings and FCF."
    })


    # ---------- Investment Intensity ----------

    current = analysis.loc[
        "CapEx / Revenue",
        current_year
    ]

    previous = analysis.loc[
        "CapEx / Revenue",
        previous_year
    ]

    change = current - previous

    severity = (
        "High"
        if abs(change) >= 10
        else "Medium"
        if abs(change) >= 5
        else "Low"
    )

    findings.append({

        "Finding":
            "Investment Intensity",

        "Metric":
            "CapEx / Revenue",

        "Current":
            current,

        "Previous / Comparator":
            previous,

        "Change":
            change,

        "Severity":
            severity,

        "Investigation":
            "Assess drivers of increased capital investment and implications for future cash generation."
    })


    # ---------- AR vs Revenue ----------

    ar_growth = analysis.loc[
        "AR Growth",
        current_year
    ]

    revenue_growth = analysis.loc[
        "Revenue Growth",
        current_year
    ]

    gap = (
        ar_growth
        - revenue_growth
    )

    severity = (
        "High"
        if gap > 10
        else "Medium"
        if gap > 5
        else "Low"
    )

    findings.append({

        "Finding":
            "Receivables vs Sales",

        "Metric":
            "AR Growth vs Revenue Growth",

        "Current":
            ar_growth,

        "Previous / Comparator":
            revenue_growth,

        "Change":
            gap,

        "Severity":
            severity,

        "Investigation":
            "Monitor receivables growth relative to revenue and assess collection trends."
    })


    # ---------- Margin Divergence ----------

    gm_change = (
        analysis.loc[
            "Gross Margin",
            current_year
        ]
        -
        analysis.loc[
            "Gross Margin",
            previous_year
        ]
    )

    ebit_change = (
        analysis.loc[
            "EBIT Margin",
            current_year
        ]
        -
        analysis.loc[
            "EBIT Margin",
            previous_year
        ]
    )

    if (
        gm_change < 0
        and
        ebit_change > 0
    ):

        findings.append({

            "Finding":
                "Margin Divergence",

            "Metric":
                "Gross Margin vs EBIT Margin Change",

            "Current":
                gm_change,

            "Previous / Comparator":
                ebit_change,

            "Change":
                gm_change - ebit_change,

            "Severity":
                "Medium",

            "Investigation":
                "Assess whether operating-expense leverage is offsetting gross-margin pressure."
        })


    return pd.DataFrame(findings)


# ============================================================
# CHART HELPERS
# ============================================================

def value_chart(
    analysis,
    years,
    metric,
    title,
    y_title="$ millions"
):

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=years,
            y=[
                analysis.loc[
                    metric,
                    y
                ]
                for y in years
            ],
            name=metric
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        xaxis_title="Fiscal Year",
        height=390,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


def percentage_chart(
    analysis,
    years,
    metric,
    title
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=[
                analysis.loc[
                    metric,
                    y
                ]
                for y in years
            ],
            mode="lines+markers",
            name=metric
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title="%",
        xaxis_title="Fiscal Year",
        height=390,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


def value_and_percentage_chart(
    analysis,
    years,
    value_metric,
    percentage_metric,
    title,
    value_name=None,
    percentage_name=None
):

    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y":
                        True
                }
            ]
        ]
    )

    fig.add_trace(

        go.Bar(
            x=years,
            y=[
                analysis.loc[
                    value_metric,
                    year
                ]
                for year in years
            ],
            name=(
                value_name
                or value_metric
            )
        ),

        secondary_y=False
    )


    fig.add_trace(

        go.Scatter(
            x=years,
            y=[
                analysis.loc[
                    percentage_metric,
                    year
                ]
                for year in years
            ],
            mode="lines+markers",
            name=(
                percentage_name
                or percentage_metric
            )
        ),

        secondary_y=True
    )


    fig.update_yaxes(
        title_text="$ millions",
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="%",
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        height=410,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


def two_value_chart(
    analysis,
    years,
    metric_1,
    metric_2,
    title
):

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=years,
            y=[
                analysis.loc[
                    metric_1,
                    y
                ]
                for y in years
            ],
            name=metric_1
        )
    )

    fig.add_trace(
        go.Bar(
            x=years,
            y=[
                analysis.loc[
                    metric_2,
                    y
                ]
                for y in years
            ],
            name=metric_2
        )
    )

    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title="$ millions",
        height=390,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


# ============================================================
# AI CONTEXT
# ============================================================

def prepare_ai_context(
    findings,
    company,
    current_year,
    previous_year
):

    material = findings[
        findings[
            "Severity"
        ].isin(
            ["Medium", "High"]
        )
    ]

    ai_findings = []

    for _, row in material.iterrows():

        item = {

            "finding":
                row["Finding"],

            "metric":
                row["Metric"],

            "severity":
                row["Severity"],

            "current":
                round(
                    float(row["Current"]),
                    2
                ),

            "comparator":
                round(
                    float(
                        row[
                            "Previous / Comparator"
                        ]
                    ),
                    2
                ),

            "change":
                round(
                    float(row["Change"]),
                    2
                ),

            "investigation":
                row["Investigation"],
        }

        ai_findings.append(item)

    return {

        "company":
            company,

        "current_year":
            current_year,

        "previous_year":
            previous_year,

        "findings":
            ai_findings,
    }


# ============================================================
# AI REVIEW
# ============================================================

def generate_ai_review(context):

    api_key = st.secrets[
        "OPENROUTER_API_KEY"
    ]

    client = OpenAI(
        base_url=
            "https://openrouter.ai/api/v1",
        api_key=
            api_key
    )

    prompt = f"""
You are a careful financial analyst reviewing company financial performance.

Company: {context["company"]}
Current period: {context["current_year"]}
Comparison period: {context["previous_year"]}

The figures below were already calculated by a deterministic financial engine.

Rules:
- Do not recalculate supplied figures.
- Do not invent financial data.
- Treat possible explanations as hypotheses.
- Clearly separate facts from hypotheses.
- Do not make buy/sell recommendations.
- Do not make unsupported claims about earnings quality,
  liquidity, creditworthiness, valuation, or management.
- Use concise professional analyst language.

For every finding provide:

### Finding
**Fact**
**Why it may matter**
**Possible drivers / hypotheses**
**What to investigate**
**Risks**
**Management questions**
**Potential implications**

Financial findings:

{json.dumps(context["findings"], indent=2)}
"""

    response = (
        client.chat.completions.create(

            model=
                "google/gemini-2.5-flash",

            messages=[

                {
                    "role":
                        "system",

                    "content":
                        (
                            "You are a careful financial analyst. "
                            "Separate facts from hypotheses and never invent financial data."
                        )
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt
                }
            ],

            temperature=
                0.2,

            max_tokens=
                2200
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# SIDEBAR / INPUT
# ============================================================

with st.sidebar:

    st.header(
        "Analysis Input"
    )

    company_name = st.text_input(
        "Company Name",
        value="Microsoft"
    )

    uploaded_file = st.file_uploader(
        "Upload standardized workbook",
        type=[
            "xlsx",
            "xls"
        ]
    )

    st.caption(
        "Required sheet: Raw_Data"
    )

    st.caption(
        "Required columns: Statement, Metric and at least two FYxxxx columns."
    )


# ============================================================
# MAIN APP
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a standardized financial workbook from the sidebar to begin."
    )

    st.markdown(
        """
### How the system works

**Financial Statements → Standardization → Financial Metrics → 
Material Findings → Charts → AI Analyst Review**

The current version expects financial data to first be entered into
the standardized workbook structure. Common financial line-item
terminology is then mapped into the model's canonical metrics.
"""
    )

    st.stop()


try:

    # ========================================================
    # LOAD WORKBOOK
    # ========================================================

    raw_data = pd.read_excel(
        uploaded_file,
        sheet_name="Raw_Data"
    )

    raw_data = raw_data.dropna(
        how="all"
    ).reset_index(
        drop=True
    )


    required_columns = {
        "Statement",
        "Metric"
    }

    if not required_columns.issubset(
        raw_data.columns
    ):

        st.error(
            "Raw_Data must contain Statement and Metric columns."
        )

        st.stop()


    raw_data = clean_financial_data(
        raw_data
    )

    years = get_year_columns(
        raw_data
    )


    if len(years) < 2:

        st.error(
            "At least two FYxxxx columns are required."
        )

        st.stop()


    standardized_data = (
        standardize_financial_data(
            raw_data
        )
    )


    analysis = calculate_metrics(
        standardized_data,
        years
    )


    findings = generate_findings(
        analysis,
        years
    )


    current_year = years[-1]

    previous_year = years[-2]


    material_findings = findings[
        findings[
            "Severity"
        ].isin(
            [
                "Medium",
                "High"
            ]
        )
    ]


    # ========================================================
    # TOP SUMMARY
    # ========================================================

    st.success(
        f"{company_name} successfully processed | "
        f"{years[0]}–{years[-1]}"
    )


    st.subheader(
        f"{current_year} Financial Snapshot"
    )


    col1, col2, col3, col4 = st.columns(4)


    revenue_growth = analysis.loc[
        "Revenue Growth",
        current_year
    ]

    ebit_margin = analysis.loc[
        "EBIT Margin",
        current_year
    ]

    fcf_margin = analysis.loc[
        "FCF Margin",
        current_year
    ]

    debt_ebitda = analysis.loc[
        "Debt / EBITDA",
        current_year
    ]


    col1.metric(
        "Revenue Growth",
        f"{revenue_growth:.2f}%"
    )

    col2.metric(
        "EBIT Margin",
        f"{ebit_margin:.2f}%"
    )

    col3.metric(
        "FCF Margin",
        f"{fcf_margin:.2f}%"
    )

    col4.metric(
        "Debt / EBITDA",
        f"{debt_ebitda:.2f}x"
    )


    # ========================================================
    # EBITDA QUALITY
    # ========================================================

    with st.expander(
        "EBITDA Definition & Data Quality"
    ):

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Classification",
            analysis.attrs[
                "ebitda_type"
            ]
        )

        c2.metric(
            "Source",
            str(
                analysis.attrs[
                    "ebitda_source"
                ]
            )
        )

        c3.metric(
            "Quality",
            analysis.attrs[
                "ebitda_quality"
            ]
        )

        if (
            analysis.attrs[
                "ebitda_quality"
            ]
            == "Proxy"
        ):

            st.warning(
                "EBITDA is presented as a proxy because the source "
                "contains D&A combined with other adjustments."
            )


    # ========================================================
    # DASHBOARD TABS
    # ========================================================

    (
        profitability_tab,
        cashflow_tab,
        working_capital_tab,
        capital_structure_tab,
        findings_tab,
        ai_tab
    ) = st.tabs(
        [
            "📈 Profitability",
            "💵 Cash Flow",
            "🔄 Working Capital",
            "🏦 Capital Structure",
            "⚠️ Findings",
            "🤖 AI Review",
        ]
    )


    # ========================================================
    # PROFITABILITY
    # ========================================================

    with profitability_tab:

        st.subheader(
            "Profitability & Growth"
        )


        c1, c2 = st.columns(2)

        with c1:

            st.plotly_chart(
                value_and_percentage_chart(
                    analysis,
                    years,
                    "Revenue",
                    "Revenue Growth",
                    "Revenue & YoY Growth",
                    "Revenue",
                    "Revenue Growth"
                ),
                use_container_width=True
            )


        with c2:

            ebitda_label = (
                "EBITDA Proxy"
                if analysis.attrs[
                    "ebitda_quality"
                ] == "Proxy"
                else "EBITDA"
            )

            st.plotly_chart(
                value_and_percentage_chart(
                    analysis,
                    years,
                    "EBITDA",
                    "EBITDA Margin",
                    f"{ebitda_label} & Margin",
                    ebitda_label,
                    "EBITDA Margin"
                ),
                use_container_width=True
            )


        c1, c2 = st.columns(2)

        with c1:

            st.plotly_chart(
                value_and_percentage_chart(
                    analysis,
                    years,
                    "Gross Profit",
                    "Gross Margin",
                    "Gross Profit & Gross Margin"
                ),
                use_container_width=True
            )


        with c2:

            st.plotly_chart(
                value_and_percentage_chart(
                    analysis,
                    years,
                    "EBIT",
                    "EBIT Margin",
                    "EBIT & Operating Margin"
                ),
                use_container_width=True
            )


        c1, c2 = st.columns(2)

        with c1:

            st.plotly_chart(
                value_and_percentage_chart(
                    analysis,
                    years,
                    "Net Income",
                    "Net Margin",
                    "Net Income & Net Margin"
                ),
                use_container_width=True
            )


        with c2:

            st.plotly_chart(
                percentage_chart(
                    analysis,
                    years,
                    "ROE",
                    "Return on Equity"
                ),
                use_container_width=True
            )

            st.caption(
                "ROE = Net Income / Average Stockholders' Equity."
            )


    # ========================================================
    # CASH FLOW
    # ========================================================

    with cashflow_tab:

        st.subheader(
            "Cash Generation & Investment"
        )


        c1, c2 = st.columns(2)

        with c1:

            st.plotly_chart(
                value_chart(
                    analysis,
                    years,
                    "FCF",
                    "Free Cash Flow"
                ),
                use_container_width=True
            )


        with c2:

            st.plotly_chart(
                percentage_chart(
                    analysis,
                    years,
                    "FCF Conversion",
                    "FCF Conversion"
                ),
                use_container_width=True
            )

            st.caption(
                "FCF Conversion = Free Cash Flow / Net Income."
            )


        c1, c2 = st.columns(2)

        with c1:

            st.plotly_chart(
                value_chart(
                    analysis,
                    years,
                    "Operating Cash Flow",
                    "Operating Cash Flow"
                ),
                use_container_width=True
            )


        with c2:

            st.plotly_chart(
                value_and_percentage_chart(
                    analysis,
                    years,
                    "CapEx",
                    "CapEx / Revenue",
                    "Capital Expenditure & Investment Intensity",
                    "CapEx",
                    "CapEx / Revenue"
                ),
                use_container_width=True
            )


    # ========================================================
    # WORKING CAPITAL
    # ========================================================

    with working_capital_tab:

        st.subheader(
            "Working Capital"
        )


        c1, c2 = st.columns(2)

        with c1:

            st.plotly_chart(
                value_chart(
                    analysis,
                    years,
                    "Operating Working Capital",
                    "Operating Working Capital"
                ),
                use_container_width=True
            )

            st.caption(
                "Simplified Operating Working Capital = "
                "Accounts Receivable + Inventory − Accounts Payable."
            )


        with c2:

            st.plotly_chart(
                two_value_chart(
                    analysis,
                    years,
                    "Accounts Receivable",
                    "Inventory",
                    "Accounts Receivable & Inventory"
                ),
                use_container_width=True
            )


        st.markdown(
            "#### Working-Capital Growth"
        )


        growth_df = pd.DataFrame(
            {
                "AR Growth":
                    analysis.loc[
                        "AR Growth"
                    ],

                "Inventory Growth":
                    analysis.loc[
                        "Inventory Growth"
                    ],

                "AP Growth":
                    analysis.loc[
                        "AP Growth"
                    ],

                "Revenue Growth":
                    analysis.loc[
                        "Revenue Growth"
                    ],
            }
        )

        st.line_chart(
            growth_df
        )


    # ========================================================
    # CAPITAL STRUCTURE
    # ========================================================

    with capital_structure_tab:

        st.subheader(
            "Liquidity & Capital Structure"
        )


        c1, c2 = st.columns(2)

        with c1:

            st.plotly_chart(
                two_value_chart(
                    analysis,
                    years,
                    "Cash",
                    "Total Debt",
                    "Cash vs Total Debt"
                ),
                use_container_width=True
            )


        with c2:

            st.plotly_chart(
                value_chart(
                    analysis,
                    years,
                    "Net Debt",
                    "Net Debt"
                ),
                use_container_width=True
            )


        c1, c2 = st.columns(2)

        with c1:

            st.plotly_chart(
                percentage_chart(
                    analysis,
                    years,
                    "Net Debt / EBITDA",
                    "Net Debt / EBITDA"
                ),
                use_container_width=True
            )

            st.caption(
                "Displayed in x rather than %. "
                "The chart shows the leverage multiple numerically."
            )


        with c2:

            leverage_df = pd.DataFrame(
                {
                    "Debt / EBITDA":
                        analysis.loc[
                            "Debt / EBITDA"
                        ],

                    "Debt / Equity":
                        analysis.loc[
                            "Debt / Equity"
                        ],

                    "Current Ratio":
                        analysis.loc[
                            "Current Ratio"
                        ],
                }
            )

            st.markdown(
                "#### Leverage & Liquidity Ratios"
            )

            st.line_chart(
                leverage_df
            )


    # ========================================================
    # FINDINGS
    # ========================================================

    with findings_tab:

        st.subheader(
            "Material Financial Findings"
        )


        if material_findings.empty:

            st.success(
                "No Medium or High severity findings were detected."
            )

        else:

            st.dataframe(
                material_findings.round(2),
                use_container_width=True,
                hide_index=True
            )


        with st.expander(
            "View all findings"
        ):

            st.dataframe(
                findings.round(2),
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # AI ANALYST REVIEW
    # ========================================================

    with ai_tab:

        st.subheader(
            "AI Analyst Review"
        )

        st.write(
            "The AI layer receives only the material findings "
            "identified by the deterministic financial engine."
        )

        st.info(
            "AI is used for interpretation, not financial calculations."
        )


        if st.button(
            "Generate AI Analyst Review",
            type="primary"
        ):

            context = prepare_ai_context(
                findings,
                company_name,
                current_year,
                previous_year
            )

            with st.spinner(
                "Generating analyst review..."
            ):

                try:

                    ai_review = (
                        generate_ai_review(
                            context
                        )
                    )

                    st.markdown(
                        ai_review
                    )

                except Exception as error:

                    st.error(
                        "AI review could not be generated."
                    )

                    st.exception(
                        error
                    )




except Exception as error:

    st.error(
        "The workbook could not be processed."
    )

    st.exception(
        error
    )