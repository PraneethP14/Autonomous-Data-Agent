"""
Agent Configuration File
Controls thresholds, strategies, and parameters for all agents
"""

class AgentConfig:
    """Central configuration for all agents"""
    
    # Data Profiling Agent Config
    MISSING_VALUE_THRESHOLD = 0.5  # Drop column if >50% missing
    DUPLICATE_THRESHOLD = 0.1  # Flag if >10% duplicates
    
    # Outlier Detection
    IQR_MULTIPLIER = 1.5  # Standard IQR multiplier
    
    # Cleaning Strategy Agent Config
    IMPUTATION_METHODS = {
        'numeric': ['mean', 'median', 'forward_fill'],
        'categorical': ['mode', 'forward_fill', 'drop'],
        'datetime': ['forward_fill', 'drop']
    }
    
    # Validation Agent Config
    QUALITY_SCORE_THRESHOLD = 70  # Minimum acceptable quality score
    MIN_IMPROVEMENT_REQUIRED = 5  # %
    
    # Learning Agent Config
    LEARNING_DB_PATH = "storage/learning_history.json"
    MAX_HISTORY_RECORDS = 1000
    
    # General
    EXECUTION_TIMEOUT = 300  # seconds
    LOG_LEVEL = "INFO"
