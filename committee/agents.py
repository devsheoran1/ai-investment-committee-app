"""
agents.py — Brief §11 deliverable. The four agents the brief describes, mapped
onto our committee (each lives in committee/ so teammates can own one file each):

    Brief agent                      Our agent(s)
    -------------------------------  ---------------------------------------------
    Data Agent (The Analyst)         data.py + screener.py tools, Macro Strategist,
                                     Equity Research, Quant, Sentiment
    Strategy Agent (The Trader)      Portfolio Manager (+ optimiser, analytics.plan_orders)
    Risk Agent (The Validator)       Risk Manager (veto) + CIO (final sign-off)
    Execution Agent                  Trader (rule-based) -> database.execute_trade
"""
from committee import (ROSTER, Agent, ChiefInvestmentOfficer, EquityResearchAnalyst,  # noqa: F401
                       MacroStrategist, PortfolioManager, QuantAnalyst, RiskManager,
                       SentimentAnalyst, Trader)

BRIEF_ROLES = {
    "Data Agent": ["Macro Strategist", "Equity Research Analyst", "Quant Analyst", "Sentiment Analyst"],
    "Strategy Agent": ["Portfolio Manager"],
    "Risk Agent": ["Risk Manager", "CIO"],
    "Execution Agent": ["Trader"],
}
