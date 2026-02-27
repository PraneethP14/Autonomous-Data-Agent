"""
AGENT 2: Cleaning Strategy Agent
Decides cleaning actions autonomously based on profiling data and reasoning rules.
THIS IS THE DECISION-MAKING HEART OF THE SYSTEM.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.data_helpers import (
    DataAnalyzer, ImbalanceDetector, DataQualityScorer, 
    MemoryOptimizer, FeatureEngineer, CleaningLogger
)
from configs.agent_config import AgentConfig

class CleaningStrategyAgent(BaseAgent):
    """
    Autonomous decision-making agent that creates a cleaning plan.
    Uses reasoning rules (not just hardcoded logic) to decide:
    - Which columns to drop
    - Which columns to impute (and what method)
    - How to handle outliers
    - Which rows to drop
    """
    
    def __init__(self):
        super().__init__("CleaningStrategyAgent")
        self.cleaning_plan = {}
        self.reasoning_log = []
    
    def execute(self, profile: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main execution method
        
        Args:
            profile: Profile report from DataProfilingAgent
            df: Original DataFrame
            
        Returns:
            Cleaning plan with decisions and reasoning
        """
        self._mark_status("running")
        start_time = datetime.now()
        
        try:
            self.logger.logger.info("Generating cleaning strategy...")
            
            plan = {
                'timestamp': datetime.now().isoformat(),
                'dataset_shape': profile['dataset_shape'],
                'actions': [],
                'summary': {},
                'reasoning': self.reasoning_log
            }
            
            # Analyze each column and decide action
            for col_name, col_profile in profile['columns'].items():
                action = self._decide_column_action(col_name, col_profile, df)
                plan['actions'].append(action)
                
                self.logger.log_decision(
                    column=col_name,
                    decision=action['action'],
                    reasoning=action['reasoning'],
                    confidence=action['confidence_score']
                )
            
            # Decide on row-level actions
            row_actions = self._decide_row_actions(profile, df)
            plan['row_actions'] = row_actions
            
            # Decide on categorical encoding
            encoding_actions = self._decide_encoding_actions(df, plan['actions'])
            plan['encoding_actions'] = encoding_actions
            
            # Decide on scaling/normalization
            scaling_actions = self._decide_scaling_actions(df, plan['actions'])
            plan['scaling_actions'] = scaling_actions
            
            # Calculate data quality before cleaning
            quality_before = DataQualityScorer.calculate_overall_quality_score(df)
            plan['quality_metrics'] = {'before_cleaning': quality_before}
            
            # Detect imbalances and propose handling
            imbalance_actions = self._decide_imbalance_handling(df)
            plan['imbalance_actions'] = imbalance_actions
            
            # Optimize memory for large datasets
            if len(df) > 10000:
                memory_info = MemoryOptimizer.estimate_memory(df)
                plan['memory_optimization'] = {
                    'current_memory_mb': memory_info['mb'],
                    'recommend_optimization': memory_info['mb'] > 100  # Flag if > 100MB
                }
            
            # Decide on auto feature engineering
            feature_engineering = self._decide_feature_engineering(df)
            plan['feature_engineering'] = feature_engineering
            
            # Calculate expected improvement
            plan['summary'] = self._estimate_improvement(plan, profile)
            
            self.cleaning_plan = plan
            self._mark_status("completed")
            
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_time = duration
            self.logger.logger.info(f"Strategy generation completed in {duration:.2f}s")
            
            return {
                'status': 'success',
                'plan': plan,
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
    
    def _decide_column_action(self, col_name: str, col_profile: Dict[str, Any], 
                              df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decide action for a single column using reasoning rules
        
        Reasoning Rules Applied:
        1. If >50% missing → DROP (column is too sparse)
        2. If >20% missing AND numeric → IMPUTE with median (robust to outliers)
        3. If >20% missing AND categorical → IMPUTE with mode (most common)
        4. If >20% missing AND datetime → DROP (risky to impute dates)
        5. If no missing → KEEP as-is
        6. If 5-20% missing → FORWARD_FILL or IMPUTE
        """
        
        missing_pct = col_profile['missing_pct']
        col_type = col_profile['type']
        unique_count = col_profile['unique_count']
        issues = col_profile.get('issues', [])
        
        # RULE 1: Too much missing data - DROP
        if missing_pct > AgentConfig.MISSING_VALUE_THRESHOLD * 100:
            reasoning = f"Column has {missing_pct:.1f}% missing (>{AgentConfig.MISSING_VALUE_THRESHOLD*100}%). Too sparse to impute reliably."
            self.reasoning_log.append({
                'rule': 'HIGH_MISSING_DROP',
                'column': col_name,
                'reasoning': reasoning
            })
            return {
                'column': col_name,
                'action': 'DROP',
                'reason': 'too_sparse',
                'reasoning': reasoning,
                'confidence_score': 0.95,
                'affected_rows': int(df[col_name].isnull().sum())
            }
        
        # RULE 2 & 3: 5-20% missing - IMPUTE
        if 5 <= missing_pct <= 20:
            if col_type == 'numeric':
                method = 'median'
                reasoning = f"Numeric column with {missing_pct:.1f}% missing. Using MEDIAN imputation (robust to outliers)."
                self.reasoning_log.append({
                    'rule': 'NUMERIC_IMPUTE_MEDIAN',
                    'column': col_name,
                    'reasoning': reasoning
                })
                return {
                    'column': col_name,
                    'action': 'IMPUTE',
                    'method': method,
                    'reasoning': reasoning,
                    'confidence_score': 0.90,
                    'affected_rows': int(df[col_name].isnull().sum()),
                    'value': float(df[col_name].median()) if not df[col_name].isna().all() else 0
                }
            
            elif col_type == 'categorical':
                method = 'mode'
                mode_val = df[col_name].mode()[0] if len(df[col_name].mode()) > 0 else 'UNKNOWN'
                reasoning = f"Categorical column with {missing_pct:.1f}% missing. Using MODE imputation ('{mode_val}')."
                self.reasoning_log.append({
                    'rule': 'CATEGORICAL_IMPUTE_MODE',
                    'column': col_name,
                    'reasoning': reasoning
                })
                return {
                    'column': col_name,
                    'action': 'IMPUTE',
                    'method': method,
                    'reasoning': reasoning,
                    'confidence_score': 0.85,
                    'affected_rows': int(df[col_name].isnull().sum()),
                    'value': str(mode_val)
                }
            
            elif col_type == 'datetime':
                reasoning = f"DateTime column with {missing_pct:.1f}% missing. Safer to DROP rows than impute dates."
                self.reasoning_log.append({
                    'rule': 'DATETIME_DROP_ROWS',
                    'column': col_name,
                    'reasoning': reasoning
                })
                return {
                    'column': col_name,
                    'action': 'DROP_ROWS',
                    'target': col_name,
                    'reasoning': reasoning,
                    'confidence_score': 0.88,
                    'affected_rows': int(df[col_name].isnull().sum())
                }
        
        # RULE 4: < 5% missing - FORWARD FILL
        if 0 < missing_pct < 5:
            reasoning = f"Minor missing data ({missing_pct:.1f}%). Using forward fill to preserve temporal continuity."
            self.reasoning_log.append({
                'rule': 'MINOR_MISSING_FORWARD_FILL',
                'column': col_name,
                'reasoning': reasoning
            })
            return {
                'column': col_name,
                'action': 'FORWARD_FILL',
                'reasoning': reasoning,
                'confidence_score': 0.92,
                'affected_rows': int(df[col_name].isnull().sum())
            }
        
        # RULE 5: No missing - KEEP
        reasoning = f"No missing values. Column quality is good, keeping as-is."
        return {
            'column': col_name,
            'action': 'KEEP',
            'reasoning': reasoning,
            'confidence_score': 1.0,
            'affected_rows': 0
        }
    
        return row_actions
    
    def _decide_encoding_actions(self, df: pd.DataFrame, cleaning_actions: list) -> Dict[str, Any]:
        """
        Decide categorical encoding strategy for each categorical column
        
        Rules:
        1. Columns marked for DROP in cleaning → skip encoding
        2. Categorical columns with ≤10 unique values → ONE-HOT encode
        3. Categorical columns with >10 unique values → LABEL encode
        4. No missing values after cleaning → safe to encode
        """
        encoding_actions = {
            'actions': []
        }
        
        # Get columns that will be kept after cleaning
        dropped_cols = set([a['column'] for a in cleaning_actions if a['action'] == 'DROP'])
        
        for col in df.columns:
            if col in dropped_cols:
                continue
            
            # Check if column is categorical
            try:
                col_type = pd.api.types.infer_dtype(df[col], skipna=True)
                is_categorical = col_type in ['object', 'category', 'string', 'mixed']
            except:
                is_categorical = False
            
            if is_categorical and df[col].dtype == 'object':
                unique_count = df[col].nunique()
                
                # Skip if mostly NaN (should be dropped anyway)
                if df[col].isnull().sum() / len(df) > 0.9:
                    continue
                
                # Decide encoding method
                if unique_count <= 10:
                    encoding_method = 'one_hot'
                    reasoning = f"Categorical column with {unique_count} unique values. Using ONE-HOT encoding for interpretability."
                else:
                    encoding_method = 'label'
                    reasoning = f"Categorical column with {unique_count} unique values. Using LABEL encoding to reduce dimensionality."
                
                action = {
                    'column': col,
                    'action': 'ENCODE',
                    'method': encoding_method,
                    'reasoning': reasoning,
                    'confidence_score': 0.90,
                    'unique_values': unique_count
                }
                encoding_actions['actions'].append(action)
                
                self.logger.log_decision(
                    column=col,
                    decision=f"ENCODE ({encoding_method.upper()})",
                    reasoning=reasoning,
                    confidence=0.90
                )
                self.reasoning_log.append({
                    'rule': f'CATEGORICAL_ENCODE_{encoding_method.upper()}',
                    'column': col,
                    'reasoning': reasoning
                })
        
        return encoding_actions
    
    def _decide_scaling_actions(self, df: pd.DataFrame, cleaning_actions: list) -> Dict[str, Any]:
        """
        Decide scaling/normalization strategy for numeric columns
        
        Rules:
        1. Only scale columns that will be kept after cleaning
        2. Numeric columns with high variance → STANDARDIZE (mean=0, std=1)
        3. Numeric columns with bounded range → MIN-MAX normalize (0-1)
        4. Use STANDARDIZATION by default for ML compatibility
        """
        scaling_actions = {
            'actions': [],
            'scale_method': 'standard'  # default to standardization
        }
        
        # Get columns that will be kept after cleaning
        dropped_cols = set([a['column'] for a in cleaning_actions if a['action'] == 'DROP'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return scaling_actions
        
        # Determine scaling method
        # Use StandardScaler for better ML model performance
        scale_method = 'standard'
        method_reasoning = "Using StandardScaler (z-score normalization) for optimal ML model performance. Transforms data to mean 0 and std 1."
        
        for col in numeric_cols:
            if col in dropped_cols:
                continue
            
            # Skip columns that are mostly missing
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 90:
                continue
            
            # Check if column has meaningful variance
            col_std = df[col].std()
            if col_std > 0:  # Only scale if there's variance
                action = {
                    'column': col,
                    'action': 'SCALE',
                    'method': scale_method,
                    'reasoning': method_reasoning,
                    'confidence_score': 0.95,
                    'mean': float(df[col].mean()),
                    'std': float(col_std)
                }
                scaling_actions['actions'].append(action)
        
        if scaling_actions['actions']:
            self.logger.log_decision(
                column='[NUMERIC]',
                decision=f"SCALE ({scale_method.upper()})",
                reasoning=method_reasoning,
                confidence=0.95
            )
            self.reasoning_log.append({
                'rule': 'NUMERIC_SCALE_STANDARD',
                'column': '[ALL_NUMERIC]',
                'reasoning': method_reasoning
            })
        
        return scaling_actions
    
    def _decide_row_actions(self, profile: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decide row-level actions (drop duplicates, etc.)
        """
        row_actions = {
            'actions': []
        }
        
        # Check duplicates
        duplicate_info = profile['issues'].get('duplicates')
        if duplicate_info and duplicate_info['percentage'] > 1:
            action = {
                'action': 'DROP_DUPLICATES',
                'reasoning': f"Dataset has {duplicate_info['count']} duplicate rows ({duplicate_info['percentage']:.2f}%). Removing all duplicates.",
                'confidence_score': 0.95,
                'affected_rows': duplicate_info['count']
            }
            row_actions['actions'].append(action)
            self.logger.log_decision(
                column='[ROWS]',
                decision='DROP_DUPLICATES',
                reasoning=action['reasoning'],
                confidence=action['confidence_score']
            )
        
        # Check for fully empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            action = {
                'action': 'DROP_EMPTY_ROWS',
                'reasoning': f"Found {empty_rows} completely empty rows. Removing them.",
                'confidence_score': 0.99,
                'affected_rows': empty_rows
            }
            row_actions['actions'].append(action)
            self.logger.log_decision(
                column='[ROWS]',
                decision='DROP_EMPTY_ROWS',
                reasoning=action['reasoning'],
                confidence=action['confidence_score']
            )
        
        return row_actions
    
    def _estimate_improvement(self, plan: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate how much the data will improve after cleaning"""
        summary = {
            'total_actions': len(plan['actions']),
            'drop_columns': len([a for a in plan['actions'] if a['action'] == 'DROP']),
            'impute_actions': len([a for a in plan['actions'] if a['action'] == 'IMPUTE']),
            'keep_columns': len([a for a in plan['actions'] if a['action'] == 'KEEP']),
            'row_actions': len(plan['row_actions']['actions']),
            'encoding_actions': len(plan['encoding_actions']['actions']),
            'scaling_actions': len(plan['scaling_actions']['actions']),
            'estimated_missing_reduction': f"~{sum([a.get('affected_rows', 0) for a in plan['actions'] if a['action'] == 'IMPUTE'])} cells",
            'estimated_duplicate_reduction': plan['row_actions']['actions'][0]['affected_rows'] if plan['row_actions']['actions'] else 0,
            'categorical_columns_to_encode': sum(1 for a in plan['encoding_actions']['actions']),
            'numeric_columns_to_scale': sum(1 for a in plan['scaling_actions']['actions'])
        }
        return summary
    
    def _decide_column_action_adaptive(self, col_name: str, col_profile: Dict[str, Any], 
                                       df: pd.DataFrame) -> Dict[str, Any]:
        """
        Adaptive missing value handling based on data distribution
        
        Uses multiple factors instead of just % missing:
        - Data type
        - Distribution (skewness, kurtosis)
        - Correlation with other columns
        - Business context (optional)
        """
        missing_pct = col_profile['missing_pct']
        col_type = col_profile['type']
        
        # Start with basic rules
        if missing_pct > AgentConfig.MISSING_VALUE_THRESHOLD * 100:
            return {
                'column': col_name,
                'action': 'DROP',
                'reason': 'too_sparse',
                'reasoning': f"Column has {missing_pct:.1f}% missing - too sparse",
                'confidence_score': 0.95,
                'adaptive': False
            }
        
        if col_type == 'numeric' and 5 <= missing_pct <= 20:
            # Adaptive: check distribution
            skewness = df[col_name].skew()
            
            # Skewed data: use median (more robust)
            if abs(skewness) > 1.0:
                method = 'median'
                reasoning = f"Numeric column with {missing_pct:.1f}% missing and skewness {skewness:.2f}. MEDIAN is more robust for skewed data."
            else:
                method = 'median'
                reasoning = f"Numeric column with {missing_pct:.1f}% missing. Using MEDIAN."
            
            return {
                'column': col_name,
                'action': 'IMPUTE',
                'method': method,
                'reasoning': reasoning,
                'confidence_score': 0.90,
                'adaptive': True,
                'skewness': float(skewness)
            }
        
        # Default behavior for other cases
        return {
            'column': col_name,
            'action': 'KEEP',
            'reasoning': 'Column quality is acceptable',
            'confidence_score': 1.0,
            'adaptive': False
        }
    
    def _decide_imbalance_handling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect class imbalance and recommend handling
        
        Returns actions for imbalanced categorical columns
        """
        imbalance_actions = {
            'actions': [],
            'imbalanced_columns': []
        }
        
        # Check categorical columns for imbalance
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() <= 2:  # Binary or categorical
                imbalance_info = ImbalanceDetector.detect_categorical_imbalance(df[col], threshold=0.75)
                
                if imbalance_info['is_imbalanced']:
                    imbalance_actions['imbalanced_columns'].append({
                        'column': col,
                        'info': imbalance_info
                    })
                    
                    action = {
                        'column': col,
                        'action': 'FLAG_IMBALANCE',
                        'severity': imbalance_info.get('severity', 'moderate'),
                        'imbalance_ratio': imbalance_info.get('imbalance_ratio'),
                        'reasoning': f"Column '{col}' is {'highly ' if imbalance_info.get('severity') == 'high' else ''}imbalanced. Top class: {imbalance_info['top_class_ratio']:.1%}, Minority: {imbalance_info['minority_class_ratio']:.1%}",
                        'recommendation': 'Apply stratified sampling or class weighting in ML models',
                        'confidence_score': 0.85
                    }
                    imbalance_actions['actions'].append(action)
                    
                    self.logger.log_decision(
                        column=col,
                        decision='FLAG_IMBALANCE',
                        reasoning=action['reasoning'],
                        confidence=action['confidence_score']
                    )
        
        # Check numeric columns for skewness
        for col in df.select_dtypes(include=[np.number]).columns:
            skew_info = ImbalanceDetector.detect_numeric_skewness(df[col])
            
            if skew_info['is_skewed']:
                action = {
                    'column': col,
                    'action': 'FLAG_SKEWNESS',
                    'skewness': skew_info['skewness'],
                    'direction': skew_info['skew_direction'],
                    'reasoning': f"Column '{col}' is highly skewed ({skew_info['skewness']:.2f}). Consider log/box-cox transformation.",
                    'recommendation': 'Apply log transformation or other scaling',
                    'confidence_score': 0.80
                }
                imbalance_actions['actions'].append(action)
        
        return imbalance_actions
    
    def _decide_feature_engineering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decide if and how to apply auto feature engineering
        
        Returns: Feature engineering plan
        """
        fe_decision = {
            'recommended': False,
            'reason': '',
            'actions': []
        }
        
        # Check if feature engineering should be applied
        if FeatureEngineer.should_engineer_features(df):
            fe_decision['recommended'] = True
            fe_decision['reason'] = f"Dataset has {len(df.columns)} columns and {len(df)} rows. Good candidate for feature engineering."
            
            # List potential features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            dt_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            actions = []
            
            if len(numeric_cols) > 0:
                actions.append({
                    'type': 'numeric_transformations',
                    'count': len(numeric_cols),
                    'description': f'Create squared terms and log transform for {len(numeric_cols)} numeric columns'
                })
            
            if len(numeric_cols) >= 2:
                actions.append({
                    'type': 'interactions',
                    'count': 1,
                    'description': f'Create interaction features between top numeric columns'
                })
            
            if len(dt_cols) > 0:
                actions.append({
                    'type': 'datetime_features',
                    'count': len(dt_cols),
                    'description': f'Extract temporal features (year, month, day, dow, is_weekend) from {len(dt_cols)} datetime columns'
                })
            
            if len(text_cols) > 0:
                actions.append({
                    'type': 'text_features',
                    'count': len(text_cols),
                    'description': f'Extract text length and word count from {len(text_cols)} text columns'
                })
            
            fe_decision['actions'] = actions
            fe_decision['feature_count_before'] = len(df.columns)
            fe_decision['estimated_feature_count_after'] = len(df.columns) + len(numeric_cols) * 2 + len(dt_cols) * 5 + len(text_cols) * 2
            
            self.logger.log_decision(
                column='[FEATURES]',
                decision='AUTO_FEATURE_ENGINEERING',
                reasoning=fe_decision['reason'],
                confidence=0.80
            )
        else:
            fe_decision['reason'] = f"Dataset too large ({len(df)} rows) or too many features ({len(df.columns)}). Skipping auto feature engineering."
        
        return fe_decision
    
    def get_plan(self) -> Dict[str, Any]:
        """Return the cleaning plan"""
        return self.cleaning_plan
