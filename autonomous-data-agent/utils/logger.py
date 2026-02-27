"""
Logging utilities for agent reasoning and execution tracking
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class AgentLogger:
    """Tracks agent reasoning and decisions for explainability"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(agent_name)
        self.decisions = []
        self.execution_log = {
            'agent': agent_name,
            'started_at': datetime.now().isoformat(),
            'decisions': [],
            'errors': [],
            'duration': 0
        }
    
    def log_decision(self, column: str, decision: str, reasoning: str, confidence: float):
        """Log a decision with reasoning"""
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'column': column,
            'decision': decision,
            'reasoning': reasoning,
            'confidence': confidence
        }
        self.decisions.append(decision_record)
        self.execution_log['decisions'].append(decision_record)
        self.logger.info(f"[{column}] {decision} (confidence: {confidence:.2f}) - {reasoning}")
    
    def log_error(self, error_msg: str, column: str = None):
        """Log errors encountered"""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'column': column,
            'error': error_msg
        }
        self.execution_log['errors'].append(error_record)
        self.logger.error(f"Error: {error_msg}")
    
    def get_log(self) -> Dict[str, Any]:
        """Return execution log"""
        return self.execution_log
    
    def save_log(self, filename: str):
        """Save log to JSON file"""
        self.execution_log['completed_at'] = datetime.now().isoformat()
        filepath = Path(f"data/reports/{filename}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.execution_log, f, indent=2)
        self.logger.info(f"Log saved to {filepath}")
