"""
AGENT 1: Data Profiling Agent
Analyzes raw datasets to detect data quality issues
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent
from utils.data_helpers import DataAnalyzer
from configs.agent_config import AgentConfig

class DataProfilingAgent(BaseAgent):
    """
    Profiles a DataFrame and generates a comprehensive quality report.
    Detects: missing values, duplicates, outliers, invalid dates, etc.
    """
    
    def __init__(self):
        super().__init__("DataProfilingAgent")
        self.profile_report = {}
        self.df = None
    
    def execute(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main execution method (optimized - no full copy)
        
        Args:
            df: Raw pandas DataFrame
            
        Returns:
            Profile report with detected issues
        """
        self._mark_status("running")
        start_time = datetime.now()
        
        try:
            # Don't copy - use reference instead for very large datasets
            self.df = df
            self.logger.logger.info(f"Profiling dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # For large datasets (>100k rows), use sampling for faster profiling
            profile_df = self.df
            sample_size = None
            if len(self.df) > 100000:
                sample_size = min(50000, int(len(self.df) * 0.5))
                profile_df = self.df.sample(n=sample_size, random_state=42)
                self.logger.logger.info(f"Using sample ({sample_size} rows) for faster profiling")
            
            # Run profiling checks
            profile = {
                'timestamp': datetime.now().isoformat(),
                'dataset_shape': {
                    'rows': df.shape[0],
                    'columns': df.shape[1]
                },
                'sample_size': sample_size,
                'columns': self._profile_columns(profile_df),
                'issues': self._detect_issues(profile_df),
                'quality_metrics': self._calculate_quality_metrics(profile_df)
            }
            
            self.profile_report = profile
            self._mark_status("completed")
            
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_time = duration
            self.logger.logger.info(f"Profiling completed in {duration:.2f}s")
            
            return {
                'status': 'success',
                'profile': profile,
                'execution_time': duration
            }
            
        except Exception as e:
            self.logger.log_error(str(e))
            self._mark_status("failed")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _profile_columns(self, profile_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Profile each column in the dataset"""
        columns_profile = {}
        
        for col in profile_df.columns:
            series = profile_df[col]
            col_type = DataAnalyzer.infer_column_type(series)
            
            # Get base statistics
            stats = DataAnalyzer.get_column_stats(series, col_type)
            
            # Additional checks for specific types (skip expensive outlier detection for very large datasets)
            issues = []
            
            if col_type == 'datetime':
                is_valid, msg = DataAnalyzer.validate_date_format(series)
                if not is_valid:
                    issues.append(f"Invalid dates: {msg}")
            
            elif col_type == 'numeric':
                # Only do outlier detection on smaller series (skip for >500k points)
                if len(series) < 500000:
                    outlier_mask = DataAnalyzer.detect_outliers_iqr(
                        series.dropna(), 
                        AgentConfig.IQR_MULTIPLIER
                    )
                    outlier_pct = (outlier_mask.sum() / len(series) * 100) if len(series) > 0 else 0
                    stats['outlier_count'] = int(outlier_mask.sum())
                    stats['outlier_pct'] = float(outlier_pct)
                    
                    if outlier_pct > 5:
                        issues.append(f"High outliers detected: {outlier_pct:.2f}%")
                else:
                    stats['outlier_count'] = 0
                    stats['outlier_pct'] = 0.0
            
            stats['issues'] = issues
            columns_profile[col] = stats
            
            # Log the profile
            self.logger.log_decision(
                column=col,
                decision=f"Profiled ({col_type})",
                reasoning=f"Type: {col_type}, Missing: {stats['missing_pct']:.2f}%, Unique: {stats['unique_count']}",
                confidence=0.95
            )
        
        return columns_profile
    
    def _detect_issues(self, profile_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect data quality issues in the dataset"""
        issues = {}
        
        # Check for duplicates (use hash for faster detection on large datasets)
        if len(profile_df) > 10000:
            # For large datasets, use hash-based duplicate detection (faster)
            duplicate_pct = (profile_df.drop_duplicates().shape[0] / profile_df.shape[0] * 100)
            duplicate_pct = 100 - duplicate_pct
        else:
            duplicate_pct = DataAnalyzer.get_duplicate_stats(profile_df)
        
        if duplicate_pct > 0:
            issues['duplicates'] = {
                'count': int(profile_df.duplicated().sum()),
                'percentage': float(duplicate_pct),
                'severity': 'high' if duplicate_pct > 10 else 'medium' if duplicate_pct > 1 else 'low'
            }
            self.logger.logger.warning(f"Duplicates found: {duplicate_pct:.2f}%")
        
        # Check for columns with too much missing data
        missing_stats = DataAnalyzer.get_missing_stats(profile_df)
        high_missing_cols = {col: pct for col, pct in missing_stats.items() 
                             if pct > AgentConfig.MISSING_VALUE_THRESHOLD * 100}
        
        if high_missing_cols:
            issues['high_missing_columns'] = high_missing_cols
            self.logger.logger.warning(f"Columns with >50% missing: {list(high_missing_cols.keys())}")
        
        # Check for empty columns
        empty_cols = [col for col in profile_df.columns if profile_df[col].isnull().all()]
        if empty_cols:
            issues['empty_columns'] = empty_cols
            self.logger.logger.warning(f"Empty columns found: {empty_cols}")
        
        return issues
    
    def _calculate_quality_metrics(self, profile_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall data quality metrics"""
        total_cells = profile_df.shape[0] * profile_df.shape[1]
        missing_cells = profile_df.isnull().sum().sum()
        
        # Quality score (0-100)
        # Starts at 100, deduct for issues
        quality_score = 100.0
        
        # Deduct for missing values
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        quality_score -= min(missing_ratio * 100, 20)  # Max -20 for missing
        
        # Deduct for duplicates
        if len(profile_df) > 10000:
            duplicate_pct = (profile_df.drop_duplicates().shape[0] / profile_df.shape[0] * 100)
            duplicate_pct = 100 - duplicate_pct
        else:
            duplicate_pct = DataAnalyzer.get_duplicate_stats(profile_df)
        quality_score -= min(duplicate_pct, 15)  # Max -15 for duplicates
        
        # Check for non-unique column names
        if len(profile_df.columns) != len(set(profile_df.columns)):
            quality_score -= 5
        
        quality_score = max(quality_score, 0)  # Floor at 0
        
        metrics = {
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_ratio': float(missing_ratio),
            'duplicate_rows': int(profile_df.duplicated().sum()),
            'duplicate_ratio': float(duplicate_pct),
            'quality_score': float(quality_score)
        }
        
        self.logger.logger.info(f"Dataset Quality Score: {quality_score:.2f}/100")
        
        return metrics
    
    def get_profile(self) -> Dict[str, Any]:
        """Return the profile report"""
        return self.profile_report
