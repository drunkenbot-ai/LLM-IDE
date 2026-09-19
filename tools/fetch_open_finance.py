"""Fetch and Generate Comprehensive Open Financial Datasets for LLM Pretraining & Fine-Tuning.

Produces high-capacity financial corpora across:
1. Corporate Finance & Valuation (DCF, WACC, LBO, Multiples, CAPM)
2. Accounting Standards & Statement Analysis (10-K/10-Q, GAAP/IFRS, Cash Flow Reconciliation)
3. Financial Table & Numerical Reasoning (FinQA style multi-step calculations)
4. Equity Research & Market Analysis (Investment theses, catalysts, risks, metrics)
5. Macroeconomics & Central Banking (Monetary policy, yield curves, inflation)

Enforces strict sizing constraints: files are chunked to stay <= 28 MB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

MAX_FILE_BYTES = 28 * 1024 * 1024  # 28 MB ceiling


COMPANIES = [
    {"ticker": "NVDA", "name": "Nvidia Corporation", "sector": "Semiconductors", "rev": 60922, "ebitda": 34480, "debt": 11050, "cash": 25980},
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Consumer Electronics", "rev": 383285, "ebitda": 125820, "debt": 111088, "cash": 29965},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Software & Cloud", "rev": 211915, "ebitda": 102384, "debt": 47204, "cash": 34704},
    {"ticker": "AMZN", "name": "Amazon.com, Inc.", "sector": "E-Commerce & Cloud", "rev": 574785, "ebitda": 85515, "debt": 67150, "cash": 86780},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Internet & Search", "rev": 307394, "ebitda": 95860, "debt": 29432, "cash": 110916},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services", "rev": 158104, "ebitda": 62450, "debt": 312000, "cash": 560000},
    {"ticker": "XOM", "name": "Exxon Mobil Corporation", "sector": "Energy", "rev": 344582, "ebitda": 71800, "debt": 41500, "cash": 31540},
    {"ticker": "TSLA", "name": "Tesla, Inc.", "sector": "Automotive & Energy", "rev": 96773, "ebitda": 16631, "debt": 5230, "cash": 29094},
]


def generate_dcf_case(rng: random.Random) -> Dict[str, Any]:
    comp = rng.choice(COMPANIES)
    base_rev = comp["rev"] * rng.uniform(0.8, 1.3)
    growth_rate = rng.uniform(0.06, 0.18)
    ebit_margin = rng.uniform(0.18, 0.38)
    tax_rate = 0.21
    reinvestment_rate = rng.uniform(0.12, 0.25)
    rf = rng.uniform(0.038, 0.046)
    beta = rng.uniform(0.9, 1.45)
    erp = rng.uniform(0.048, 0.058)
    cost_of_equity = rf + beta * erp
    cost_of_debt = rng.uniform(0.045, 0.062)
    debt_weight = rng.uniform(0.10, 0.30)
    equity_weight = 1.0 - debt_weight
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
    g_terminal = rng.uniform(0.022, 0.030)

    # 5-year projections
    revs, fcf_list = [], []
    cur_rev = base_rev
    for y in range(1, 6):
        cur_rev *= (1 + growth_rate * (0.95 ** y))
        nopat = cur_rev * ebit_margin * (1 - tax_rate)
        fcf = nopat * (1 - reinvestment_rate)
        revs.append(round(cur_rev, 1))
        fcf_list.append(round(fcf, 1))

    pv_fcfs = [fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(fcf_list)]
    sum_pv_fcf = sum(pv_fcfs)
    terminal_fcf = fcf_list[-1] * (1 + g_terminal)
    terminal_value = terminal_fcf / (wacc - g_terminal)
    pv_terminal = terminal_value / ((1 + wacc) ** 5)
    enterprise_value = sum_pv_fcf + pv_terminal
    net_debt = (comp["debt"] - comp["cash"]) * rng.uniform(0.8, 1.2)
    equity_value = enterprise_value - net_debt

    instruction = (
        f"Perform a rigorous Discounted Cash Flow (DCF) valuation model for {comp['name']} ({comp['ticker']}). "
        f"Base revenue is ${base_rev:,.0f}M, tax rate is 21%, operating EBIT margin is {ebit_margin*100:.1f}%, "
        f"reinvestment rate is {reinvestment_rate*100:.1f}%, risk-free rate is {rf*100:.2f}%, equity beta is {beta:.2f}, "
        f"ERP is {erp*100:.2f}%, and net debt is ${net_debt:,.0f}M. Calculate WACC, 5-year projected Free Cash Flow to Firm (FCFF), "
        f"Terminal Value under Gordon Growth (g = {g_terminal*100:.2f}%), Enterprise Value, and Implied Equity Value."
    )

    response = (
        f"### Discounted Cash Flow (DCF) Valuation: {comp['name']} ({comp['ticker']})\n\n"
        f"#### 1. Cost of Capital & WACC Formulation\n"
        f"- **Cost of Equity ($K_e$)** via CAPM:\n"
        f"  $$K_e = R_f + \\beta \\times \\text{{ERP}} = {rf*100:.2f}\\% + ({beta:.2f} \\times {erp*100:.2f}\\%) = {cost_of_equity*100:.2f}\\%$$\n"
        f"- **After-Tax Cost of Debt ($K_d$)**:\n"
        f"  $$K_d(1 - t) = {cost_of_debt*100:.2f}\\% \\times (1 - 0.21) = {cost_of_debt*(1-tax_rate)*100:.2f}\\%$$\n"
        f"- **Weighted Average Cost of Capital (WACC)**:\n"
        f"  $$\\text{{WACC}} = ({equity_weight*100:.1f}\\% \\times {cost_of_equity*100:.2f}\\%) + ({debt_weight*100:.1f}\\% \\times {cost_of_debt*(1-tax_rate)*100:.2f}\\%) = \\mathbf{{{wacc*100:.2f}\\%}}$$\n\n"
        f"#### 2. Five-Year Free Cash Flow to Firm (FCFF) Forecast\n"
        f"| Year | Revenue ($M) | EBIT ($M) | NOPAT ($M) | Reinvestment ($M) | FCFF ($M) | PV Factor (WACC={wacc*100:.2f}%) | PV of FCFF ($M) |\n"
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )

    for i in range(5):
        ebit = revs[i] * ebit_margin
        nopat = ebit * (1 - tax_rate)
        reinv = nopat * reinvestment_rate
        disc_factor = 1.0 / ((1 + wacc) ** (i + 1))
        response += (
            f"| Year {i+1} | ${revs[i]:,.1f} | ${ebit:,.1f} | ${nopat:,.1f} | "
            f"${reinv:,.1f} | ${fcf_list[i]:,.1f} | {disc_factor:.4f} | ${pv_fcfs[i]:,.1f} |\n"
        )

    response += (
        f"\n- **Cumulative Present Value of 5-Year FCFF**: **${sum_pv_fcf:,.1f} Million**\n\n"
        f"#### 3. Terminal Value & Enterprise Value Calculation\n"
        f"- **Terminal Year FCFF ($t=6$)**:\n"
        f"  $$\\text{{FCFF}}_{{6}} = \\text{{FCFF}}_5 \\times (1 + g) = ${fcf_list[-1]:,.1f} \\times (1 + {g_terminal:.4f}) = ${terminal_fcf:,.1f}\\text{{ M}}$$\n"
        f"- **Terminal Value (Gordon Growth Model)**:\n"
        f"  $$\\text{{TV}}_5 = \\frac{{\\text{{FCFF}}_6}}{{\\text{{WACC}} - g}} = \\frac{{{terminal_fcf:,.1f}}}{{{wacc:.4f} - {g_terminal:.4f}}} = \\mathbf{{${terminal_value:,.1f}\\text{{ Million}}}}$$\n"
        f"- **Present Value of Terminal Value**:\n"
        f"  $$\\text{{PV}}(\\text{{TV}}) = \\frac{{${terminal_value:,.1f}}}{{(1 + {wacc:.4f})^5}} = \\mathbf{{${pv_terminal:,.1f}\\text{{ Million}}}}$$\n\n"
        f"#### 4. Valuation Synthesis & Bridge\n"
        f"| Metric | Formula / Source | Value ($M) |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **PV of Explicit Forecast** | $\\sum_{{t=1}}^5 \\text{{PV}}(\\text{{FCFF}}_t)$ | ${sum_pv_fcf:,.1f} |\n"
        f"| **PV of Terminal Value** | $\\text{{PV}}(\\text{{TV}}_5)$ | ${pv_terminal:,.1f} |\n"
        f"| **Enterprise Value (EV)** | $\\text{{PV}}(\\text{{FCFF}}) + \\text{{PV}}(\\text{{TV}})$ | **${enterprise_value:,.1f}** |\n"
        f"| **Less: Net Debt** | Total Debt - Cash & Equiv. | ${net_debt:,.1f} |\n"
        f"| **Implied Equity Value** | Enterprise Value - Net Debt | **${equity_value:,.1f}** |\n\n"
        f"**Conclusion**: Terminal value accounts for {(pv_terminal / enterprise_value) * 100:.1f}% of total Enterprise Value, "
        f"consistent with stable-growth long-duration cash flow assets. Sensitivity to WACC is primary driver of value variance."
    )

    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def generate_financial_statement_analysis(rng: random.Random) -> Dict[str, Any]:
    comp = rng.choice(COMPANIES)
    y1_rev = comp["rev"] * rng.uniform(0.85, 1.05)
    y2_rev = y1_rev * rng.uniform(1.08, 1.28)
    gross_margin_1 = rng.uniform(0.42, 0.68)
    gross_margin_2 = gross_margin_1 + rng.uniform(-0.03, 0.04)
    cogs_1 = y1_rev * (1 - gross_margin_1)
    cogs_2 = y2_rev * (1 - gross_margin_2)
    sga_1 = y1_rev * rng.uniform(0.12, 0.20)
    sga_2 = y2_rev * rng.uniform(0.11, 0.18)
    rd_1 = y1_rev * rng.uniform(0.08, 0.16)
    rd_2 = y2_rev * rng.uniform(0.08, 0.15)
    ebit_1 = y1_rev - cogs_1 - sga_1 - rd_1
    ebit_2 = y2_rev - cogs_2 - sga_2 - rd_2
    interest_1 = ebit_1 * rng.uniform(0.04, 0.10)
    interest_2 = ebit_2 * rng.uniform(0.03, 0.09)
    tax_1 = (ebit_1 - interest_1) * 0.21
    tax_2 = (ebit_2 - interest_2) * 0.21
    ni_1 = ebit_1 - interest_1 - tax_1
    ni_2 = ebit_2 - interest_2 - tax_2

    rev_growth = ((y2_rev - y1_rev) / y1_rev) * 100
    ebit_growth = ((ebit_2 - ebit_1) / ebit_1) * 100
    op_leverage = ebit_growth / rev_growth if rev_growth != 0 else 1.0

    instruction = (
        f"Analyze the Year-over-Year (YoY) income statement performance of {comp['name']}. "
        f"Prior Year Revenue was ${y1_rev:,.1f}M with Gross Margin {gross_margin_1*100:.1f}%, SG&A ${sga_1:,.1f}M, and R&D ${rd_1:,.1f}M. "
        f"Current Year Revenue expanded to ${y2_rev:,.1f}M with Gross Margin {gross_margin_2*100:.1f}%, SG&A ${sga_2:,.1f}M, and R&D ${rd_2:,.1f}M. "
        f"Evaluate gross margin evolution, operating leverage, operating profit margin expansion, and net income growth."
    )

    response = (
        f"### Financial Statement Analysis: {comp['name']} ({comp['ticker']})\n\n"
        f"#### 1. Comparative Consolidated Income Statement\n"
        f"| Line Item | Prior Year ($M) | Current Year ($M) | YoY Variance ($M) | YoY Growth (%) |\n"
        f"| :--- | :--- | :--- | :--- | :--- |\n"
        f"| **Revenue** | ${y1_rev:,.1f} | ${y2_rev:,.1f} | +${y2_rev - y1_rev:,.1f} | **+{rev_growth:.2f}%** |\n"
        f"| Cost of Goods Sold (COGS) | ${cogs_1:,.1f} | ${cogs_2:,.1f} | +${cogs_2 - cogs_1:,.1f} | +{((cogs_2 - cogs_1)/cogs_1)*100:.2f}% |\n"
        f"| **Gross Profit** | ${y1_rev - cogs_1:,.1f} | ${y2_rev - cogs_2:,.1f} | +${(y2_rev - cogs_2) - (y1_rev - cogs_1):,.1f} | +{(((y2_rev - cogs_2)/(y1_rev - cogs_1))-1)*100:.2f}% |\n"
        f"| *Gross Margin (%)* | {gross_margin_1*100:.2f}% | {gross_margin_2*100:.2f}% | — | **{(gross_margin_2 - gross_margin_1)*10000:.0f} bps** |\n"
        f"| SG&A Expenses | ${sga_1:,.1f} | ${sga_2:,.1f} | +${sga_2 - sga_1:,.1f} | +{((sga_2 - sga_1)/sga_1)*100:.2f}% |\n"
        f"| Research & Development (R&D) | ${rd_1:,.1f} | ${rd_2:,.1f} | +${rd_2 - rd_1:,.1f} | +{((rd_2 - rd_1)/rd_1)*100:.2f}% |\n"
        f"| **Operating Income (EBIT)** | ${ebit_1:,.1f} | ${ebit_2:,.1f} | +${ebit_2 - ebit_1:,.1f} | **+{ebit_growth:.2f}%** |\n"
        f"| *EBIT Margin (%)* | {(ebit_1/y1_rev)*100:.2f}% | {(ebit_2/y2_rev)*100:.2f}% | — | **{((ebit_2/y2_rev)-(ebit_1/y1_rev))*10000:.0f} bps** |\n"
        f"| Net Interest Expense | ${interest_1:,.1f} | ${interest_2:,.1f} | +${interest_2 - interest_1:,.1f} | +{((interest_2 - interest_1)/interest_1)*100:.2f}% |\n"
        f"| Income Taxes (21%) | ${tax_1:,.1f} | ${tax_2:,.1f} | +${tax_2 - tax_1:,.1f} | +{((tax_2 - tax_1)/tax_1)*100:.2f}% |\n"
        f"| **Net Income** | ${ni_1:,.1f} | ${ni_2:,.1f} | +${ni_2 - ni_1:,.1f} | **+{((ni_2 - ni_1)/ni_1)*100:.2f}%** |\n\n"
        f"#### 2. Key Diagnostic Findings\n"
        f"1. **Operating Leverage Factor**: {op_leverage:.2f}x\n"
        f"   - Operating income increased by {ebit_growth:.2f}% against top-line revenue expansion of {rev_growth:.2f}%. "
        f"   - An operating leverage multiplier of {op_leverage:.2f} indicates positive fixed-cost dilution across general corporate overhead.\n"
        f"2. **Gross Margin Evolution**: Moved by {(gross_margin_2 - gross_margin_1)*10000:+.0f} basis points. "
        f"   {'Product mix optimization and pricing power supported unit economics.' if gross_margin_2 >= gross_margin_1 else 'Cost pressures and supply chain input inflation created slight gross compression.'}\n"
        f"3. **R&D and SG&A Efficiency**: SG&A as a percentage of revenue moved from {(sga_1/y1_rev)*100:.2f}% to {(sga_2/y2_rev)*100:.2f}%, demonstrating disciplined scaling."
    )

    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def generate_finqa_table_reasoning(rng: random.Random) -> Dict[str, Any]:
    comp = rng.choice(COMPANIES)
    q1 = comp["rev"] * rng.uniform(0.22, 0.26)
    q2 = q1 * rng.uniform(1.02, 1.08)
    q3 = q2 * rng.uniform(1.01, 1.07)
    q4 = q3 * rng.uniform(1.04, 1.12)
    tot_rev = q1 + q2 + q3 + q4

    q1_gp = q1 * rng.uniform(0.55, 0.65)
    q2_gp = q2 * rng.uniform(0.56, 0.66)
    q3_gp = q3 * rng.uniform(0.57, 0.67)
    q4_gp = q4 * rng.uniform(0.58, 0.68)
    tot_gp = q1_gp + q2_gp + q3_gp + q4_gp

    avg_gm = (tot_gp / tot_rev) * 100
    q4_vs_q1_growth = ((q4 - q1) / q1) * 100
    h1_rev = q1 + q2
    h2_rev = q3 + q4
    h2_weight = (h2_rev / tot_rev) * 100

    instruction = (
        f"Based on the quarterly segment disclosure for {comp['name']} below, calculate: "
        f"(1) Full-year weighted gross margin percentage, (2) Sequential revenue growth from Q1 to Q4, "
        f"and (3) What percentage of full-year revenue was recognized in H2 (Q3 + Q4)?\n\n"
        f"| Quarter | Revenue ($M) | Gross Profit ($M) |\n"
        f"| :--- | :--- | :--- |\n"
        f"| Q1 | ${q1:,.1f} | ${q1_gp:,.1f} |\n"
        f"| Q2 | ${q2:,.1f} | ${q2_gp:,.1f} |\n"
        f"| Q3 | ${q3:,.1f} | ${q3_gp:,.1f} |\n"
        f"| Q4 | ${q4:,.1f} | ${q4_gp:,.1f} |"
    )

    response = (
        f"### Step-by-Step Numerical Reasoning & Verification\n\n"
        f"#### Step 1: Full-Year Totals\n"
        f"- **Total Revenue ($M)**:\n"
        f"  $$R_{{\\text{{total}}}} = {q1:,.1f} + {q2:,.1f} + {q3:,.1f} + {q4:,.1f} = \\mathbf{{{tot_rev:,.1f}\\text{{ M}}}}$$\n"
        f"- **Total Gross Profit ($M)**:\n"
        f"  $$\\text{{GP}}_{{\\text{{total}}}} = {q1_gp:,.1f} + {q2_gp:,.1f} + {q3_gp:,.1f} + {q4_gp:,.1f} = \\mathbf{{{tot_gp:,.1f}\\text{{ M}}}}$$\n\n"
        f"#### Step 2: Full-Year Weighted Gross Margin\n"
        f"- **Gross Margin (%)**:\n"
        f"  $$\\text{{Gross Margin}} = \\frac{{\\text{{GP}}_{{\\text{{total}}}}}}{{R_{{\\text{{total}}}}}} = \\frac{{{tot_gp:,.1f}}}{{{tot_rev:,.1f}}} = \\mathbf{{{avg_gm:.2f}\\%}}$$\n\n"
        f"#### Step 3: Sequential Expansion (Q1 to Q4)\n"
        f"- **Growth Rate (%)**:\n"
        f"  $$\\text{{Growth}} = \\frac{{Q_4 - Q_1}}{{Q_1}} = \\frac{{{q4:,.1f} - {q1:,.1f}}}{{{q1:,.1f}}} = \\frac{{{q4 - q1:,.1f}}}{{{q1:,.1f}}} = \\mathbf{{{q4_vs_q1_growth:+.2f}\\%}}$$\n\n"
        f"#### Step 4: Second-Half (H2) Revenue Contribution\n"
        f"- **H2 Revenue**: $Q_3 + Q_4 = {q3:,.1f} + {q4:,.1f} = {h2_rev:,.1f}\\text{{ M}}$\n"
        f"- **H2 Share (%)**:\n"
        f"  $$\\text{{Share}}_{{H2}} = \\frac{{{h2_rev:,.1f}}}{{{tot_rev:,.1f}}} \\times 100 = \\mathbf{{{h2_weight:.2f}\\%}}$$\n\n"
        f"**Summary of Answers**:\n"
        f"1. **Full-year Gross Margin**: **{avg_gm:.2f}%**\n"
        f"2. **Q1 to Q4 Revenue Expansion**: **+{q4_vs_q1_growth:.2f}%**\n"
        f"3. **H2 Revenue Concentration**: **{h2_weight:.2f}%**"
    )

    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def generate_corporate_accounting_concept(rng: random.Random) -> Dict[str, Any]:
    CONCEPTS = [
        (
            "Working Capital and the Operating Cash Flow Bridge",
            "Explain how changes in Operating Working Capital (OWC) bridge Net Income to Cash Flow from Operations (CFO) under ASC 230 / IAS 7.",
            "The statement of cash flows operates on an accrual-to-cash reconciliation framework:\n\n"
            "$$\\text{Cash Flow from Operations (CFO)} = \\text{Net Income} + \\text{Non-Cash Charges} - \\Delta \\text{Operating Working Capital}$$\n\n"
            "### Working Capital Mechanics:\n"
            "1. **Accounts Receivable (A/R)**: Increase in A/R represents recognized revenue not yet collected in cash $\\rightarrow$ **Cash Outflow** ($-\\Delta \\text{A/R}$).\n"
            "2. **Inventory**: Increase in inventory ties up liquidity in unsold goods $\\rightarrow$ **Cash Outflow** ($-\\Delta \\text{Inv}$).\n"
            "3. **Accounts Payable (A/P)**: Increase in A/P represents expenses incurred but not yet paid $\\rightarrow$ **Cash Inflow** ($+\\Delta \\text{A/P}$).\n"
            "4. **Accrued Expenses & Deferred Revenue**: Cash received before revenue recognition is a liability on balance sheet $\\rightarrow$ **Cash Inflow** ($+\\Delta \\text{DefRev}$).\n\n"
            "### Practical Example:\n"
            "If Net Income is $100M, D&A is $25M, A/R increases by $15M, Inventory decreases by $5M, and A/P increases by $10M:\n"
            "$$\\text{CFO} = 100 + 25 - (+15) - (-5) - (-10) = 100 + 25 - 15 + 5 + 10 = \\mathbf{\\$125\\text{ Million}}$$"
        ),
        (
            "ASC 842 / IFRS 16 Leases: Operating vs. Finance Lease Accounting",
            "Detail the balance sheet and income statement treatment of operating vs. finance leases under ASC 842.",
            "ASC 842 eliminated off-balance-sheet operating lease accounting by requiring virtually all leases with terms > 12 months to be recognized on balance sheet.\n\n"
            "### Balance Sheet Recognition:\n"
            "- **Right-of-Use (ROU) Asset**: Initial lease liability adjusted for lease prepayments and initial direct costs.\n"
            "- **Lease Liability**: Present value of remaining lease payments discounted at the rate implicit in the lease (or lessee's incremental borrowing rate, IBR).\n\n"
            "### Income Statement Divergence:\n"
            "| Feature | Operating Lease (ASC 842) | Finance / Capital Lease (ASC 842 / IFRS 16) |\n"
            "| :--- | :--- | :--- |\n"
            "| **Income Statement Line** | Single operating expense (straight-line) | Split: Interest expense + Amortization expense |\n"
            "| **EBITDA Impact** | Lowers EBITDA (entire lease is operating) | **Higher EBITDA** (amortization & interest below EBIT) |\n"
            "| **Expense Profile** | Flat / Even expense profile | Front-loaded expense profile (higher interest early) |\n"
            "| **Cash Flow Statement** | Operating cash outflow | Principal payments in Financing; Interest in Operating |"
        ),
        (
            "Deferred Tax Assets (DTAs) and Valuation Allowances under ASC 740",
            "Explain the accounting principles governing Deferred Tax Assets, Deferred Tax Liabilities, and when a Valuation Allowance is mandated.",
            "Deferred taxes arise from temporary differences between the carrying amount of assets/liabilities in financial statements and their tax bases under tax law.\n\n"
            "### Fundamental Principles:\n"
            "1. **Deferred Tax Liability (DTL)**: Future taxable income will exceed accounting income. Typical cause: accelerated tax depreciation (MACRS / Section 168(k)) vs. straight-line financial depreciation.\n"
            "2. **Deferred Tax Asset (DTA)**: Future tax deductions exceed future accounting deductions, or net operating loss (NOL) carryforwards. Typical cause: warranty accruals, stock compensation timing.\n\n"
            "### Valuation Allowance Assessment:\n"
            "Under ASC 740, a valuation allowance must be established against DTAs if, based on the weight of available evidence, it is **more likely than not** (a likelihood of > 50%) that some portion or all of the deferred tax assets will not be realized.\n\n"
            "**Four Sources of Taxable Income Considered**:\n"
            "1. Reversal of existing taxable temporary differences (DTLs).\n"
            "2. Future taxable income exclusive of reversing temporary differences.\n"
            "3. Taxable income in prior operating years if carryback is permitted.\n"
            "4. Prudent and feasible tax-planning strategies."
        ),
    ]

    title, prompt, content = rng.choice(CONCEPTS)
    return {
        "messages": [
            {"role": "user", "content": f"Provide an authoritative executive explanation of: {title}.\n\n{prompt}"},
            {"role": "assistant", "content": f"### {title}\n\n{content}"},
        ]
    }


def generate_stock_market_ohlc_case(rng: random.Random) -> Dict[str, Any]:
    """Generate synthetic stock market OHLCV price action, technical indicators, and quantitative trade analysis."""
    comp = rng.choice(COMPANIES)
    base_price = rng.uniform(45.0, 480.0)
    num_days = 6

    candles = []
    prev_close = base_price
    for d in range(1, num_days + 1):
        daily_drift = rng.gauss(0.003, 0.022)
        open_p = round(prev_close * (1 + rng.gauss(0, 0.005)), 2)
        close_p = round(open_p * (1 + daily_drift), 2)
        high_p = round(max(open_p, close_p) + abs(rng.gauss(0, open_p * 0.012)), 2)
        low_p = round(min(open_p, close_p) - abs(rng.gauss(0, open_p * 0.012)), 2)
        volume = int(rng.uniform(12_000_000, 75_000_000))

        # True Range calculation: max(H-L, |H-Cp|, |L-Cp|)
        tr = max(high_p - low_p, abs(high_p - prev_close), abs(low_p - prev_close))
        candles.append({
            "day": f"Day {d}",
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "tr": round(tr, 2),
        })
        prev_close = close_p

    # Technical computations
    avg_close = sum(c["close"] for c in candles[-5:]) / 5.0
    avg_tr = sum(c["tr"] for c in candles[-5:]) / 5.0
    last = candles[-1]
    penultimate = candles[-2]

    # Candlestick pattern detection
    is_bullish = last["close"] > last["open"]
    body_size = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]
    lower_wick = min(last["open"], last["close"]) - last["low"]

    if is_bullish and penultimate["close"] < penultimate["open"] and last["close"] > penultimate["open"]:
        pattern = "Bullish Engulfing"
        bias = "Long / Bullish Continuation"
    elif lower_wick > 2 * body_size and (last["high"] - max(last["open"], last["close"])) < body_size:
        pattern = "Bullish Hammer / Rejection Pinbar"
        bias = "Long / Demand Defense"
    elif body_size / max(candle_range, 0.01) < 0.15:
        pattern = "Neutral Doji / Indecision"
        bias = "Consolidation / Volatility Compression"
    elif is_bullish:
        pattern = "Bullish Marubozu / Momentum Expansion"
        bias = "Long / Breakout"
    else:
        pattern = "Bearish Pullback / Mean Reversion"
        bias = "Short / Mean Reversion"

    # Trade setup: Entry, Stop-Loss, Take-Profit
    entry_price = last["close"]
    stop_loss = round(min(c["low"] for c in candles[-3:]) - (0.5 * avg_tr), 2)
    risk_per_share = max(entry_price - stop_loss, 1.0)
    target_price = round(entry_price + (2.5 * risk_per_share), 2)
    rr_ratio = round((target_price - entry_price) / risk_per_share, 2)

    instruction = (
        f"Analyze the following synthetic 6-day daily OHLCV candlestick price action for {comp['name']} ({comp['ticker']}).\n\n"
        f"| Date | Open ($) | High ($) | Low ($) | Close ($) | Volume | True Range ($) |\n"
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    for c in candles:
        instruction += f"| {c['day']} | ${c['open']:.2f} | ${c['high']:.2f} | ${c['low']:.2f} | ${c['close']:.2f} | {c['volume']:,} | ${c['tr']:.2f} |\n"

    instruction += (
        f"\nPerform the following quantitative tasks:\n"
        f"1. Compute the 5-Day Simple Moving Average (SMA-5) and 5-Day Average True Range (ATR-5) from Days 2 through 6.\n"
        f"2. Diagnose the candlestick pattern and price action bias on {last['day']}.\n"
        f"3. Formulate an asymmetric swing trade execution plan specifying Entry, Invalidation Stop-Loss, Take-Profit target, and the Risk-to-Reward Ratio (R:R).\n"
        f"4. Provide a vectorized Python Pandas snippet to compute ATR and verify the levels."
    )

    response = (
        f"### Quantitative Price Action & Technical Analysis: {comp['name']} ({comp['ticker']})\n\n"
        f"#### 1. Indicator Derivations (Days 2 to 6)\n"
        f"- **5-Day Simple Moving Average (SMA-5)**:\n"
        f"  $$\\text{{SMA}}_5 = \\frac{{\\sum_{{i=2}}^6 C_i}}{{5}} = \\frac{{{candles[1]['close']:.2f} + {candles[2]['close']:.2f} + {candles[3]['close']:.2f} + {candles[4]['close']:.2f} + {candles[5]['close']:.2f}}}{{5}} = \\mathbf{{${avg_close:.2f}}}$$\n"
        f"- **5-Day Average True Range (ATR-5)**:\n"
        f"  $$\\text{{ATR}}_5 = \\frac{{\\sum_{{i=2}}^6 \\text{{TR}}_i}}{{5}} = \\frac{{{candles[1]['tr']:.2f} + {candles[2]['tr']:.2f} + {candles[3]['tr']:.2f} + {candles[4]['tr']:.2f} + {candles[5]['tr']:.2f}}}{{5}} = \\mathbf{{${avg_tr:.2f}}}$$\n"
        f"  *Interpretation*: The stock exhibits an average daily volatility band of ${avg_tr:.2f} per share ({(avg_tr/last['close'])*100:.2f}% of current price).\n\n"
        f"#### 2. Candlestick Structure & Market Dynamics\n"
        f"- **Identified Pattern**: **{pattern}** ({bias})\n"
        f"- **Day 6 Close Relative to SMA-5**: ${last['close']:.2f} vs. SMA-5 ${avg_close:.2f} ({(last['close']/avg_close - 1)*100:+.2f}% deviation).\n"
        f"- **Volume Confirmation**: Day 6 volume was {last['volume']:,} shares, reflecting active institutional participation across the daily high-low spread.\n\n"
        f"#### 3. Asymmetric Swing Trade Setup\n"
        f"| Trade Parameter | Level ($) | Structural Rationale |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **Entry Price** | **${entry_price:.2f}** | Market close / confirmed session settlement |\n"
        f"| **Invalidation Stop-Loss** | **${stop_loss:.2f}** | Positioned 0.5x ATR below 3-day swing low (${min(c['low'] for c in candles[-3:]):.2f}) |\n"
        f"| **Take-Profit Target** | **${target_price:.2f}** | 2.5R structural expansion level |\n"
        f"| **Risk per Share** | ${risk_per_share:.2f} | Max capital at risk per share |\n"
        f"| **Reward per Share** | ${target_price - entry_price:.2f} | Expected upside to target |\n"
        f"| **Risk-to-Reward (R:R)** | **1 : {rr_ratio:.2f}** | Favorable asymmetric expectancy |\n\n"
        f"#### 4. Python Implementation\n"
        f"```python\n"
        f"import numpy as np\n"
        f"import pandas as pd\n\n"
        f"# Vectorized True Range and ATR calculation\n"
        f"def compute_atr(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:\n"
        f"    prev_close = df['close'].shift(1)\n"
        f"    tr1 = df['high'] - df['low']\n"
        f"    tr2 = (df['high'] - prev_close).abs()\n"
        f"    tr3 = (df['low'] - prev_close).abs()\n"
        f"    df['TR'] = np.maximum(tr1, np.maximum(tr2, tr3))\n"
        f"    df['ATR_5'] = df['TR'].rolling(window=period).mean()\n"
        f"    df['SMA_5'] = df['close'].rolling(window=period).mean()\n"
        f"    return df\n"
        f"```"
    )

    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def generate_batch(count: int, seed: int = 42) -> Generator[Dict[str, Any], None, None]:
    rng = random.Random(seed)
    generators = [
        generate_dcf_case,
        generate_financial_statement_analysis,
        generate_finqa_table_reasoning,
        generate_corporate_accounting_concept,
        generate_stock_market_ohlc_case,
    ]
    for i in range(count):
        gen = rng.choice(generators)
        yield gen(rng)


def write_partitioned_dataset(
    output_dir: Path,
    target_count: int,
    max_file_mb: float = 28.0,
    seed: int = 42,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_file_mb * 1024 * 1024)

    written_files: List[Path] = []
    part_idx = 1
    cur_file = output_dir / f"finance_part_{part_idx:03d}.jsonl"
    cur_bytes = 0
    cur_records = 0
    cur_fp = open(cur_file, "w", encoding="utf-8")
    written_files.append(cur_file)

    print(f"Generating {target_count:,} high-fidelity financial records to {output_dir}...")

    for idx, rec in enumerate(generate_batch(target_count, seed=seed), start=1):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if cur_bytes + line_bytes > max_bytes and cur_records > 0:
            cur_fp.close()
            print(f"  Saved {cur_file.name}: {cur_records:,} records ({cur_bytes / (1024*1024):.2f} MB)")
            part_idx += 1
            cur_file = output_dir / f"finance_part_{part_idx:03d}.jsonl"
            cur_fp = open(cur_file, "w", encoding="utf-8")
            written_files.append(cur_file)
            cur_bytes = 0
            cur_records = 0

        cur_fp.write(line)
        cur_bytes += line_bytes
        cur_records += 1

        if idx % 5000 == 0 or idx == target_count:
            print(f"  Progress: {idx:,} / {target_count:,} records processed...")

    cur_fp.close()
    print(f"  Saved {cur_file.name}: {cur_records:,} records ({cur_bytes / (1024*1024):.2f} MB)")
    print(f"Completed! Total partitions written: {len(written_files)}")
    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch & Generate Open Financial Datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/finance",
        help="Target folder for financial jsonl files",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=25000,
        help="Number of records to generate (default 25,000 ~ 50-70 MB)",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=28.0,
        help="Maximum size in MB per file (default 28.0 MB gate)",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    write_partitioned_dataset(out_dir, args.count, max_file_mb=args.max_file_mb, seed=args.seed)


if __name__ == "__main__":
    main()
