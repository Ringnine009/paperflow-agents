"""The PaperFlow agent team."""

from paperflow.agents.critic import CriticAgent
from paperflow.agents.reader import ReaderAgent
from paperflow.agents.researcher import ResearcherAgent
from paperflow.agents.synthesizer import SynthesizerAgent

TEAM = [ResearcherAgent, ReaderAgent, CriticAgent, SynthesizerAgent]

__all__ = ["ResearcherAgent", "ReaderAgent", "CriticAgent", "SynthesizerAgent", "TEAM"]
