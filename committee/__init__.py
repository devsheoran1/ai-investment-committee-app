"""The 8-member investment committee (7 LLM agents + the rule-based Trader)."""
from .cio import ChiefInvestmentOfficer
from .devils_advocate import DevilsAdvocate
from .equity_research import EquityResearchAnalyst
from .macro_strategist import MacroStrategist
from .portfolio_manager import PortfolioManager
from .quant_analyst import QuantAnalyst
from .reflector import ReflectionAnalyst
from .risk_manager import RiskManager
from .sentiment_analyst import SentimentAnalyst
from .trader import Trader
from .base import Agent

ROSTER = [
    ChiefInvestmentOfficer, MacroStrategist, EquityResearchAnalyst, QuantAnalyst,
    SentimentAnalyst, PortfolioManager, DevilsAdvocate, RiskManager, Trader,
]
