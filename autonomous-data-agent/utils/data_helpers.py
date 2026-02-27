"""
Data manipulation and analysis helpers
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import pickle
from pathlib import Path

class DataAnalyzer:
    """Utilities for data analysis and profiling"""
    
    @staticmethod
    def get_missing_stats(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate missing value statistics"""
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        return missing_pct
    
    @staticmethod
    def get_duplicate_stats(df: pd.DataFrame) -> float:
        """Calculate duplicate row percentage"""
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df) * 100) if len(df) > 0 else 0
        return duplicate_pct
    
    @staticmethod
    def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> np.ndarray:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers.values
    
    @staticmethod
    def infer_column_type(series: pd.Series) -> str:
        """Infer the semantic type of a column"""
        dtype = series.dtype
        
        # Numeric
        if dtype in ['int64', 'float64']:
            return 'numeric'
        
        # Datetime
        if dtype == 'datetime64[ns]':
            return 'datetime'
        try:
            pd.to_datetime(series.dropna().head(3))
            return 'datetime'
        except:
            pass
        
        # Categorical (limited unique values)
        if series.nunique() < len(series) * 0.05:
            return 'categorical'
        
        # Default
        return 'text'
    
    @staticmethod
    def validate_date_format(series: pd.Series) -> Tuple[bool, str]:
        """Check if series contains valid dates"""
        try:
            pd.to_datetime(series.dropna())
            return True, "Valid dates"
        except Exception as e:
            return False, f"Invalid date format: {str(e)}"
    
    @staticmethod
    def get_column_stats(series: pd.Series, col_type: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a column"""
        stats = {
            'name': series.name,
            'type': col_type,
            'missing_count': series.isnull().sum(),
            'missing_pct': (series.isnull().sum() / len(series) * 100),
            'unique_count': series.nunique(),
            'distinct_ratio': series.nunique() / len(series) if len(series) > 0 else 0
        }
        
        if col_type == 'numeric':
            stats.update({
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'skewness': series.skew()
            })
        elif col_type == 'categorical':
            stats['mode'] = series.mode().values[0] if len(series.mode()) > 0 else None
            stats['top_values'] = series.value_counts().head(5).to_dict()
        
        return stats

class CategoricalEncoder:
    """Handles categorical encoding (one-hot and label encoding)"""
    
    def __init__(self):
        self.encoders = {}
        self.encoded_columns = {}
    
    @staticmethod
    def should_one_hot_encode(series: pd.Series, max_categories: int = 10) -> bool:
        """Determine if column should be one-hot encoded vs label encoded"""
        unique_count = series.nunique()
        return unique_count <= max_categories
    
    def one_hot_encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        One-hot encode a categorical column
        
        Args:
            df: DataFrame
            column: Column to encode
            
        Returns:
            DataFrame with one-hot encoded column (original column dropped)
        """
        df_result = df.copy()
        
        # Create one-hot encoded columns
        one_hot = pd.get_dummies(df_result[column], prefix=column, drop_first=False)
        df_result = pd.concat([df_result, one_hot], axis=1)
        
        # Store the column names created
        self.encoded_columns[column] = list(one_hot.columns)
        
        # Drop original column
        df_result = df_result.drop(columns=[column])
        
        return df_result
    
    def label_encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Label encode a categorical column
        
        Args:
            df: DataFrame
            column: Column to encode
            
        Returns:
            DataFrame with label encoded column
        """
        df_result = df.copy()
        
        # Create label encoder
        le = LabelEncoder()
        
        # Handle NaN values separately
        mask = df_result[column].notna()
        df_result.loc[mask, column] = le.fit_transform(df_result.loc[mask, column].astype(str))
        
        # Store encoder for future use
        self.encoders[column] = le
        
        return df_result
    
    def get_encoders(self) -> Dict:
        """Return all fitted encoders"""
        return self.encoders
    
    def get_encoded_columns(self) -> Dict:
        """Return mapping of original to encoded column names"""
        return self.encoded_columns


class DataScaler:
    """Handles normalization and scaling of numeric columns"""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize scaler
        
        Args:
            method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        self.method = method
        self.scalers = {}
        self.scaled_columns = []
    
    def fit_and_scale(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Fit scaler on data and transform
        
        Args:
            df: DataFrame
            columns: List of numeric columns to scale (None = all numeric columns)
            
        Returns:
            DataFrame with scaled columns
        """
        df_result = df.copy()
        
        # Auto-detect numeric columns if not specified
        if columns is None:
            columns = df_result.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to only columns that exist
        columns = [col for col in columns if col in df_result.columns]
        
        if not columns:
            return df_result
        
        # Initialize scaler
        if self.method == 'standard':
            scaler = StandardScaler()
        elif self.method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        # Fit and transform
        scaled_values = scaler.fit_transform(df_result[columns])
        df_result[columns] = scaled_values
        
        # Store scaler for each column
        for col in columns:
            self.scalers[col] = scaler
        
        self.scaled_columns = columns
        
        return df_result
    
    def scale(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Scale data using fitted scalers (without fitting)
        
        Args:
            df: DataFrame
            columns: Columns to scale (uses self.scaled_columns if None)
            
        Returns:
            DataFrame with scaled columns
        """
        if columns is None:
            columns = self.scaled_columns
        
        df_result = df.copy()
        
        for col in columns:
            if col in self.scalers:
                scaler = self.scalers[col]
                df_result[[col]] = scaler.transform(df_result[[col]])
        
        return df_result
    
    def get_scalers(self) -> Dict:
        """Return all fitted scalers"""
        return self.scalers
    
    def get_scaling_method(self) -> str:
        """Return the scaling method used"""
        return self.method


class ImbalanceDetector:
    """Detects class imbalance and data skewness"""
    
    @staticmethod
    def detect_categorical_imbalance(series: pd.Series, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Detect imbalance in categorical column
        
        Args:
            series: Categorical series
            threshold: If top class > threshold, consider imbalanced
            
        Returns:
            Dict with imbalance metrics
        """
        if series.isnull().all():
            return {'is_imbalanced': False, 'reason': 'All values are NaN'}
        
        value_counts = series.value_counts(normalize=True)
        
        if len(value_counts) < 2:
            return {'is_imbalanced': False, 'reason': 'Only one unique value'}
        
        top_class_ratio = value_counts.iloc[0]
        minority_class_ratio = value_counts.iloc[-1]
        imbalance_ratio = top_class_ratio / minority_class_ratio if minority_class_ratio > 0 else float('inf')
        
        return {
            'is_imbalanced': top_class_ratio > threshold,
            'top_class_ratio': float(top_class_ratio),
            'minority_class_ratio': float(minority_class_ratio),
            'imbalance_ratio': float(imbalance_ratio) if imbalance_ratio != float('inf') else None,
            'unique_values': len(value_counts),
            'distribution': value_counts.to_dict(),
            'dominant_class': value_counts.index[0],
            'severity': 'high' if imbalance_ratio > 10 else 'moderate' if imbalance_ratio > 3 else 'low'
        }
    
    @staticmethod
    def detect_numeric_skewness(series: pd.Series) -> Dict[str, Any]:
        """
        Detect skewness in numeric column
        
        Args:
            series: Numeric series
            
        Returns:
            Dict with skewness metrics
        """
        if series.isnull().all():
            return {'is_skewed': False, 'reason': 'All values are NaN'}
        
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        return {
            'is_skewed': abs(skewness) > 1.0,
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'skew_direction': 'right' if skewness > 0 else 'left' if skewness < 0 else 'symmetric',
            'mean': float(series.mean()),
            'median': float(series.median()),
            'mean_vs_median_diff': float(abs(series.mean() - series.median())),
            'severity': 'high' if abs(skewness) > 2 else 'moderate' if abs(skewness) > 1 else 'low'
        }


class DataQualityScorer:
    """Scores overall data quality with multiple metrics"""
    
    @staticmethod
    def calculate_completeness(df: pd.DataFrame) -> float:
        """Percentage of non-null values (0-100)"""
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0
        return float(completeness)
    
    @staticmethod
    def calculate_uniqueness(df: pd.DataFrame) -> float:
        """Percentage of unique rows out of total (0-100)"""
        if len(df) == 0:
            return 0.0
        unique_rows = df.drop_duplicates().shape[0]
        uniqueness = (unique_rows / len(df) * 100)
        return float(uniqueness)
    
    @staticmethod
    def calculate_consistency(df: pd.DataFrame) -> float:
        """
        Consistency score based on:
        - Data type uniformity within columns
        - Missing pattern consistency
        
        Returns: 0-100 score
        """
        if df.empty:
            return 0.0
        
        consistency_score = 0.0
        num_cols = len(df.columns)
        
        for col in df.columns:
            try:
                # Check if column has consistent types
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    types = non_null.apply(type).unique()
                    type_consistency = (1 / len(types)) * 100 if len(types) > 0 else 0
                    consistency_score += type_consistency
            except:
                pass
        
        return float(consistency_score / num_cols) if num_cols > 0 else 0.0
    
    @staticmethod
    def calculate_validity(df: pd.DataFrame) -> float:
        """
        Validity score based on:
        - Range constraints (reasonable min/max)
        - Format validation
        - No obvious errors
        
        Returns: 0-100 score
        """
        if df.empty:
            return 0.0
        
        valid_cells = 0
        total_cells = df.shape[0] * df.shape[1]
        
        for col in df.columns:
            for val in df[col]:
                try:
                    if pd.notna(val):
                        # Check for reasonable values (not inf, reasonable ranges)
                        if isinstance(val, (int, float)):
                            if not np.isinf(val):
                                valid_cells += 1
                        else:
                            valid_cells += 1
                    else:
                        pass  # NaN is not counted as invalid
                except:
                    pass
        
        validity = (valid_cells / total_cells * 100) if total_cells > 0 else 0
        return float(validity)
    
    @staticmethod
    def calculate_overall_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive data quality score (0-100)
        
        Weights:
        - Completeness: 40% (most important)
        - Uniqueness: 20%
        - Consistency: 20%
        - Validity: 20%
        """
        completeness = DataQualityScorer.calculate_completeness(df)
        uniqueness = DataQualityScorer.calculate_uniqueness(df)
        consistency = DataQualityScorer.calculate_consistency(df)
        validity = DataQualityScorer.calculate_validity(df)
        
        overall_score = (
            completeness * 0.40 +
            uniqueness * 0.20 +
            consistency * 0.20 +
            validity * 0.20
        )
        
        # Determine quality level
        if overall_score >= 90:
            level = 'EXCELLENT'
        elif overall_score >= 75:
            level = 'GOOD'
        elif overall_score >= 50:
            level = 'FAIR'
        else:
            level = 'POOR'
        
        return {
            'overall_score': float(overall_score),
            'level': level,
            'completeness': float(completeness),
            'uniqueness': float(uniqueness),
            'consistency': float(consistency),
            'validity': float(validity),
            'components': {
                'completeness_weight': 0.40,
                'uniqueness_weight': 0.20,
                'consistency_weight': 0.20,
                'validity_weight': 0.20
            }
        }


class MemoryOptimizer:
    """Optimizes memory usage for large datasets"""
    
    @staticmethod
    def estimate_memory(df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate memory usage of DataFrame"""
        memory_bytes = df.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / (1024 ** 2)
        memory_gb = memory_mb / 1024
        
        return {
            'bytes': int(memory_bytes),
            'mb': float(memory_mb),
            'gb': float(memory_gb),
            'per_column': df.memory_usage(deep=True).to_dict()
        }
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Optimize data types to reduce memory usage
        
        Optimizations:
        - int64 → int32/int16/int8 where possible
        - float64 → float32 where possible
        - object → category for low-cardinality strings
        """
        df_optimized = df.copy()
        original_dtypes = {}
        optimizations = {}
        
        for col in df_optimized.columns:
            original_dtypes[col] = str(df_optimized[col].dtype)
            
            # Optimize integers
            if df_optimized[col].dtype == 'int64':
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                    optimizations[col] = 'int64 → int8'
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                    optimizations[col] = 'int64 → int16'
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
                    optimizations[col] = 'int64 → int32'
            
            # Optimize floats
            elif df_optimized[col].dtype == 'float64':
                df_optimized[col] = df_optimized[col].astype(np.float32)
                optimizations[col] = 'float64 → float32'
            
            # Convert high-frequency strings to category
            elif df_optimized[col].dtype == 'object':
                num_unique = df_optimized[col].nunique()
                num_total = len(df_optimized[col])
                
                # Category if <10% unique values and not mostly strings
                if num_unique / num_total < 0.1 and num_unique < 1000:
                    df_optimized[col] = df_optimized[col].astype('category')
                    optimizations[col] = f'object → category ({num_unique} categories)'
        
        return df_optimized, optimizations
    
    @staticmethod
    def get_memory_reduction(original_memory: float, optimized_memory: float) -> Dict[str, Any]:
        """Calculate memory reduction statistics"""
        reduction_mb = original_memory - optimized_memory
        reduction_pct = (reduction_mb / original_memory * 100) if original_memory > 0 else 0
        
        return {
            'original_mb': float(original_memory),
            'optimized_mb': float(optimized_memory),
            'reduction_mb': float(reduction_mb),
            'reduction_percent': float(reduction_pct)
        }


class FeatureEngineer:
    """Automatically generates useful features from existing columns"""
    
    @staticmethod
    def generate_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Auto-generate features from existing columns
        
        Strategies:
        1. Numeric: squares, logs, interactions
        2. Datetime: extract year, month, day, day_of_week, is_weekend
        3. String length for text columns
        """
        df_engineered = df.copy()
        features_created = {}
        
        # Process numeric columns
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_engineered[col].std() > 0:  # Only for columns with variance
                # Log transformation for skewed data
                if df_engineered[col].skew() > 1.5:
                    new_col = f'{col}_log'
                    df_engineered[new_col] = np.log1p(df_engineered[col].abs())
                    features_created[new_col] = f'Log transformation of {col}'
                
                # Square for potential polynomial relationships
                new_col = f'{col}_squared'
                df_engineered[new_col] = df_engineered[col] ** 2
                features_created[new_col] = f'Square of {col}'
        
        # Interaction features for top numeric columns
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            new_col = f'{col1}_x_{col2}'
            df_engineered[new_col] = df_engineered[col1] * df_engineered[col2]
            features_created[new_col] = f'Interaction: {col1} × {col2}'
        
        # Process datetime columns
        dt_cols = df_engineered.select_dtypes(include=['datetime64']).columns
        for col in dt_cols:
            try:
                df_engineered[f'{col}_year'] = df_engineered[col].dt.year
                features_created[f'{col}_year'] = f'Year from {col}'
                
                df_engineered[f'{col}_month'] = df_engineered[col].dt.month
                features_created[f'{col}_month'] = f'Month from {col}'
                
                df_engineered[f'{col}_day'] = df_engineered[col].dt.day
                features_created[f'{col}_day'] = f'Day from {col}'
                
                df_engineered[f'{col}_dow'] = df_engineered[col].dt.dayofweek
                features_created[f'{col}_dow'] = f'Day of week from {col}'
                
                df_engineered[f'{col}_is_weekend'] = (df_engineered[col].dt.dayofweek >= 5).astype(int)
                features_created[f'{col}_is_weekend'] = f'Is weekend from {col}'
            except:
                pass
        
        # Process text columns
        text_cols = df_engineered.select_dtypes(include=['object']).columns
        for col in text_cols:
            try:
                df_engineered[f'{col}_length'] = df_engineered[col].astype(str).str.len()
                features_created[f'{col}_length'] = f'Text length of {col}'
                
                df_engineered[f'{col}_word_count'] = df_engineered[col].astype(str).str.split().str.len()
                features_created[f'{col}_word_count'] = f'Word count in {col}'
            except:
                pass
        
        return df_engineered, features_created
    
    @staticmethod
    def should_engineer_features(df: pd.DataFrame) -> bool:
        """Determine if feature engineering should be applied"""
        # Only if dataset is reasonably small to avoid explosion of features
        num_features = len(df.columns)
        num_rows = len(df)
        
        # Don't engineer if already many features or very large dataset
        return num_features < 50 and num_rows < 1000000


class CleaningLogger:
    """Creates human-readable, explainable cleaning logs"""
    
    def __init__(self):
        self.log_entries = []
        self.transformations = {}
    
    def add_entry(self, stage: str, action: str, details: Dict[str, Any], 
                  reasoning: str, confidence: float = 1.0):
        """
        Add a log entry with explanations
        
        Args:
            stage: Stage name (PROFILING, STRATEGY, EXECUTION, etc.)
            action: What action was taken
            details: Details about the action
            reasoning: Why this decision was made
            confidence: Confidence in the decision (0-1)
        """
        entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'stage': stage,
            'action': action,
            'details': details,
            'reasoning': reasoning,
            'confidence': float(confidence),
            'confidence_label': self._confidence_label(confidence)
        }
        self.log_entries.append(entry)
    
    @staticmethod
    def _confidence_label(confidence: float) -> str:
        """Convert confidence score to label"""
        if confidence >= 0.95:
            return 'VERY_HIGH'
        elif confidence >= 0.85:
            return 'HIGH'
        elif confidence >= 0.70:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def get_human_readable_log(self) -> str:
        """Generate human-readable log as formatted text"""
        if not self.log_entries:
            return "No transformations logged."
        
        log_text = []
        log_text.append("=" * 80)
        log_text.append("DATA CLEANING & TRANSFORMATION LOG")
        log_text.append("=" * 80)
        
        current_stage = None
        for i, entry in enumerate(self.log_entries, 1):
            # Stage header
            if entry['stage'] != current_stage:
                current_stage = entry['stage']
                log_text.append("")
                log_text.append(f"[STAGE {i}] {current_stage.upper()}")
                log_text.append("-" * 80)
            
            # Entry details
            log_text.append(f"\n{i}. {entry['action']}")
            log_text.append(f"   Confidence: {entry['confidence_label']} ({entry['confidence']:.0%})")
            log_text.append(f"   Reasoning: {entry['reasoning']}")
            
            # Details
            if entry['details']:
                log_text.append(f"   Details:")
                for key, value in entry['details'].items():
                    log_text.append(f"     - {key}: {value}")
        
        log_text.append("")
        log_text.append("=" * 80)
        log_text.append(f"Total Transformations: {len(self.log_entries)}")
        log_text.append("=" * 80)
        
        return "\n".join(log_text)
    
    def get_json_log(self) -> Dict[str, Any]:
        """Return log as structured JSON"""
        return {
            'total_entries': len(self.log_entries),
            'entries': self.log_entries,
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of cleaning"""
        if not self.log_entries:
            return {}
        
        stages = {}
        for entry in self.log_entries:
            stage = entry['stage']
            if stage not in stages:
                stages[stage] = 0
            stages[stage] += 1
        
        avg_confidence = np.mean([e['confidence'] for e in self.log_entries])
        
        return {
            'stages_executed': list(stages.keys()),
            'transformations_per_stage': stages,
            'average_confidence': float(avg_confidence),
            'high_confidence_decisions': len([e for e in self.log_entries if e['confidence'] >= 0.85]),
            'low_confidence_decisions': len([e for e in self.log_entries if e['confidence'] < 0.70])
        }
    
    def clear(self):
        """Clear all log entries"""
        self.log_entries = []