"""
Base Agent Class - All agents inherit from this
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime
from utils.logger import AgentLogger

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = AgentLogger(agent_name)
        self.execution_time = 0
        self.status = "initialized"
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute agent logic - must be implemented by subclasses"""
        pass
    
    def get_execution_log(self) -> Dict[str, Any]:
        """Return execution log"""
        return self.logger.get_log()
    
    def save_report(self, filename: str):
        """Save execution report"""
        self.logger.save_log(filename)
    
    def _mark_status(self, status: str):
        """Update agent status"""
        self.status = status
        self.logger.logger.info(f"Status: {status}")
