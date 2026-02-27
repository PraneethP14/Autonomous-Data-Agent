"""
AGENT 4: Validation & Quality Agent
Compares raw vs cleaned data, validates consistency, and calculates quality score.
Returns PASS/FAIL verdict and suggests retry strategies if needed.
"""
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.data_helpers import DataAnalyzer
from configs.agent_config import AgentConfig

class ValidationAgent(BaseAgent):
    """
    Validates the cleaned data and ensures quality improvements.
    Produces a comprehensive quality report.
    """
    
    def __init__(self):
        super().__init__("ValidationAgent")
        self.validation_report = {}
    
    def execute(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                transformation_log: list) -> Dict[str, Any]:
        """
        Main execution method
        
        Args:
            original_df: Original raw DataFrame
            cleaned_df: Cleaned DataFrame
            transformation_log: Log of transformations applied
            
        Returns:
            Validation report with quality score and PASS/FAIL verdict
        """
        self._mark_status("running")
        start_time = datetime.now()
        
        try:
            self.logger.logger.info("Starting data validation...")
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'original_shape': original_df.shape,
                'cleaned_shape': cleaned_df.shape,
                'schema_validation': self._validate_schema(original_df, cleaned_df),
                'completeness_analysis': self._analyze_completeness(original_df, cleaned_df),
                'duplication_analysis': self._analyze_duplicates(original_df, cleaned_df),
                'transformation_validation': self._validate_transformations(original_df, cleaned_df, transformation_log),
                'quality_metrics': self._calculate_quality_metrics(original_df, cleaned_df),
                'verdict': None,
                'recommendations': []
            }
            
            # Determine verdict
            verdict = self._determine_verdict(report)
            report['verdict'] = verdict
            
            # Add recommendations if needed
            if verdict['status'] != 'PASS':
                report['recommendations'] = self._generate_recommendations(report)
            
            self.validation_report = report
            self._mark_status("completed")
            
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_time = duration
            
            self.logger.logger.info(f"Validation completed in {duration:.2f}s")
            self.logger.logger.info(f"Verdict: {verdict['status']} (Quality Score: {report['quality_metrics']['cleaned_quality_score']:.2f})")
            
            return {
                'status': 'success',
                'report': report,
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
    
    def _validate_schema(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that schema is consistent"""
        schema_info = {
            'original_columns': len(original_df.columns),
            'cleaned_columns': len(cleaned_df.columns),
            'columns_removed': len(original_df.columns) - len(cleaned_df.columns),
            'columns_removed_list': list(set(original_df.columns) - set(cleaned_df.columns)),
            'dtypes_consistent': True,
            'issues': []
        }
        
        # Check for data type changes
        for col in cleaned_df.columns:
            if col in original_df.columns:
                if original_df[col].dtype != cleaned_df[col].dtype:
                    schema_info['dtypes_consistent'] = False
                    schema_info['issues'].append(f"{col}: dtype changed from {original_df[col].dtype} to {cleaned_df[col].dtype}")
        
        self.logger.log_decision(
            column='[SCHEMA]',
            decision='SCHEMA_VALIDATED',
            reasoning=f"Removed {schema_info['columns_removed']} columns. Schema consistency: {schema_info['dtypes_consistent']}",
            confidence=0.95
        )
        
        return schema_info
    
    def _analyze_completeness(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data reduction"""
        
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        original_missing_pct = (original_missing / (original_df.shape[0] * original_df.shape[1])) * 100
        cleaned_missing_pct = (cleaned_missing / (cleaned_df.shape[0] * cleaned_df.shape[1])) * 100
        
        completeness = {
            'original_missing_cells': int(original_missing),
            'cleaned_missing_cells': int(cleaned_missing),
            'cells_imputed': int(original_missing - cleaned_missing),
            'original_missing_pct': float(original_missing_pct),
            'cleaned_missing_pct': float(cleaned_missing_pct),
            'improvement_pct': float(original_missing_pct - cleaned_missing_pct)
        }
        
        self.logger.log_decision(
            column='[COMPLETENESS]',
            decision='COMPLETENESS_ANALYZED',
            reasoning=f"Missing data reduced from {original_missing_pct:.2f}% to {cleaned_missing_pct:.2f}% ({completeness['improvement_pct']:.2f}% improvement)",
            confidence=0.95
        )
        
        return completeness
    
    def _analyze_duplicates(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate reduction (optimized for large datasets)"""
        
        # For very large datasets, estimate duplicates using sampling rather than full scan
        if len(original_df) > 1000000:
            # Use approximate duplicate detection
            sample_size = min(100000, len(original_df))
            original_sample = original_df.sample(n=sample_size, random_state=42)
            cleaned_sample = cleaned_df.sample(n=sample_size, random_state=42) if len(cleaned_df) > sample_size else cleaned_df
            
            original_dupes = int(original_sample.duplicated().sum() * (len(original_df) / sample_size))
            cleaned_dupes = int(cleaned_sample.duplicated().sum() * (len(cleaned_df) / sample_size)) if len(cleaned_df) > 0 else 0
        else:
            original_dupes = original_df.duplicated().sum()
            cleaned_dupes = cleaned_df.duplicated().sum()
        
        original_dupe_pct = (original_dupes / len(original_df)) * 100 if len(original_df) > 0 else 0
        cleaned_dupe_pct = (cleaned_dupes / len(cleaned_df)) * 100 if len(cleaned_df) > 0 else 0
        
        duplicates = {
            'original_duplicate_rows': int(original_dupes),
            'cleaned_duplicate_rows': int(cleaned_dupes),
            'duplicates_removed': int(original_dupes - cleaned_dupes),
            'original_duplicate_pct': float(original_dupe_pct),
            'cleaned_duplicate_pct': float(cleaned_dupe_pct)
        }
        
        if duplicates['duplicates_removed'] > 0:
            self.logger.log_decision(
                column='[DUPLICATES]',
                decision='DUPLICATES_REMOVED',
                reasoning=f"Removed {duplicates['duplicates_removed']} duplicate rows ({original_dupe_pct:.2f}% → {cleaned_dupe_pct:.2f}%)",
                confidence=0.95
            )
        
        return duplicates
    
    def _validate_transformations(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                                   transformation_log: list) -> Dict[str, Any]:
        """Validate that transformations were applied correctly"""
        
        validation = {
            'total_transformations': len(transformation_log),
            'successful_transformations': len([t for t in transformation_log if t.get('status') == 'success']),
            'failed_transformations': len([t for t in transformation_log if t.get('status') == 'failed']),
            'skipped_transformations': len([t for t in transformation_log if t.get('status') == 'skipped']),
            'issues': []
        }
        
        # Check for consistency
        if cleaned_df.isnull().all().any():
            validation['issues'].append("Cleaned data contains completely empty columns")
        
        if len(cleaned_df) > len(original_df):
            validation['issues'].append("ERROR: Cleaned data has more rows than original!")
        
        if len(cleaned_df.columns) > len(original_df.columns):
            validation['issues'].append("ERROR: Cleaned data has more columns than original!")
        
        self.logger.log_decision(
            column='[TRANSFORMATIONS]',
            decision='TRANSFORMATIONS_VALIDATED',
            reasoning=f"Applied {validation['successful_transformations']} successful transformations, {validation['failed_transformations']} failed",
            confidence=0.90
        )
        
        return validation
    
    def _calculate_quality_metrics(self, original_df: pd.DataFrame, 
                                    cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        
        # Original quality score
        original_quality = self._calc_quality_score(original_df)
        
        # Cleaned quality score
        cleaned_quality = self._calc_quality_score(cleaned_df)
        
        # Calculate improvements
        quality_improvement = cleaned_quality - original_quality
        
        completeness_original = 100 - (original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1]) * 100)
        completeness_cleaned = 100 - (cleaned_df.isnull().sum().sum() / (cleaned_df.shape[0] * cleaned_df.shape[1]) * 100) if cleaned_df.shape[0] * cleaned_df.shape[1] > 0 else 100
        
        metrics = {
            'original_quality_score': float(original_quality),
            'cleaned_quality_score': float(cleaned_quality),
            'quality_improvement_points': float(quality_improvement),
            'original_completeness_pct': float(completeness_original),
            'cleaned_completeness_pct': float(completeness_cleaned),
            'data_retention_pct': float((len(cleaned_df) / len(original_df)) * 100) if len(original_df) > 0 else 100
        }
        
        self.logger.log_decision(
            column='[QUALITY]',
            decision='QUALITY_CALCULATED',
            reasoning=f"Quality score: {original_quality:.2f} → {cleaned_quality:.2f} ({quality_improvement:+.2f} points)",
            confidence=0.95
        )
        
        return metrics
    
    def _calc_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate quality score for a DataFrame (0-100) - optimized for large datasets"""
        if df.shape[0] == 0:
            return 0
        
        score = 100.0
        
        # Deduct for missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        score -= min(missing_ratio * 100, 20)
        
        # Deduct for duplicates (use sampling for very large datasets)
        if len(df) > 1000000:
            sample_size = min(100000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42)
            duplicate_pct = (df_sample.duplicated().sum() / len(df_sample)) * 100
        else:
            duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        
        score -= min(duplicate_pct, 15)
        
        # Deduct for potential issues
        if len(df.columns) != len(set(df.columns)):
            score -= 5
        
        return max(score, 0)
    
    def _determine_verdict(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if cleaning passed or failed"""
        
        quality_score = report['quality_metrics']['cleaned_quality_score']
        improvement = report['quality_metrics']['quality_improvement_points']
        completeness = report['quality_metrics']['cleaned_completeness_pct']
        
        # Verdict logic
        verdict_status = 'PASS'
        reasons = []
        confidence = 0.95
        
        # Check threshold
        if quality_score < AgentConfig.QUALITY_SCORE_THRESHOLD:
            verdict_status = 'FAIL'
            reasons.append(f"Quality score {quality_score:.2f} below threshold {AgentConfig.QUALITY_SCORE_THRESHOLD}")
            confidence = 0.85
        
        # Check minimum improvement
        if improvement < AgentConfig.MIN_IMPROVEMENT_REQUIRED:
            verdict_status = 'FAIL'
            reasons.append(f"Improvement {improvement:.2f}% below required {AgentConfig.MIN_IMPROVEMENT_REQUIRED}%")
            confidence = 0.80
        
        # Warning if completeness is still low
        if completeness < 85:
            verdict_status = 'WARN'
            reasons.append(f"Completeness {completeness:.2f}% is still low")
            confidence = 0.75
        
        return {
            'status': verdict_status,
            'quality_score': quality_score,
            'improvement_points': improvement,
            'completeness_pct': completeness,
            'confidence': confidence,
            'reasons': reasons if reasons else ['All checks passed']
        }
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> list:
        """Generate retry recommendations if validation fails"""
        
        recommendations = []
        
        if report['completeness_analysis']['improvement_pct'] < 10:
            recommendations.append("Consider more aggressive imputation strategies")
        
        if report['quality_metrics']['cleaned_completeness_pct'] < 80:
            recommendations.append("Drop more rows with critical missing values")
        
        if report['duplication_analysis']['cleaned_duplicate_pct'] > 1:
            recommendations.append("Check for complex duplicate patterns (case sensitivity, whitespace)")
        
        if report['transformation_validation']['failed_transformations'] > 0:
            recommendations.append("Fix failed transformations before retry")
        
        return recommendations
    
    def get_report(self) -> Dict[str, Any]:
        """Return the validation report"""
        return self.validation_report
