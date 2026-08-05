# Crypto Liquidity and Credit Risk Appetite

This repository contains the data, Python code, and machine-readable results for the seminar paper:

**Crypto Liquidity and Credit Risk Appetite: Evidence from Stablecoin Supply and Other Crypto-Market Indicators**

Author: Jardén Joe Berger  
University of Basel, Faculty of Business and Economics  
Course: Blockchain, Smart Contracts & DeFi  
Submission date: 5 August 2026

## Research question

The project examines whether stablecoin liquidity and crypto derivatives-positioning indicators contain predictive information about subsequent changes in US high-yield and investment-grade credit spreads.

The sample covers 1 January 2020 through 31 December 2025.

## Main findings

The primary analysis finds limited unadjusted evidence that funding rates and stablecoin-supply growth precede credit-spread changes. However, none of the twelve primary crypto-to-credit relationships survives the Benjamini–Hochberg false-discovery-rate adjustment.

Evidence is stronger in the reverse direction: selected changes in credit spreads precede movements in stablecoin supply, open interest, and trading volume.

All estimated VAR systems satisfy the stability condition, but all reject residual whiteness. The results should therefore be interpreted as predictive timing relationships rather than causal effects, and the reported statistical evidence remains subject to diagnostic limitations.

Because no forward relationship passed the predefined screening rule, the conditional second-stage predictive regressions were not estimated.

## Repository structure

```text
Coding/
├── data/
│   └── Raw source data used in the analysis
└── code/
    ├── data_table.py
    ├── data_table.csv
    ├── var.py
    └── var_results/
        └── Machine-readable VAR results and diagnostics
