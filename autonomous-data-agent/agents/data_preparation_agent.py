"""
Autonomous Data Preparation Agent
Converts datasets into fully model-ready datasets optimized for ML performance
Implements 9-step strict validation and transformation pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler

class DataPreparationAgent:
    """
    Autonomous Data Preparation Agent following 9-step pipeline
    """
    
    def __init__(self):
        self.preparation_log = []
        self.dataframe = None
        self.target_column = None
        self.initial_shape = None
        self.final_shape = None
        self.problem_type = None
        
    def _print_log(self, message: str, level: str = "INFO"):
        """Print log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if level == "INFO":
            print(f"{timestamp} {message}")
        else:
            print(f"{timestamp} [{level}] {message}")
        
    def add_log_entry(self, step: str, action: str, reasoning: str, confidence: float, details: Dict = None):
        """Add entry to preparation log"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "action": action,
            "reasoning": reasoning,
            "confidence": confidence,
            "details": details or {}
        }
        self.preparation_log.append(entry)
        
        confidence_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.6 else "LOW"
        self._print_log(
            f"[{step}] {action} | Confidence: {confidence_label} ({confidence:.2f}) | {reasoning}"
        )
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        """Execute full 9-step data preparation pipeline"""
        try:
            self._print_log("=" * 80)
            self._print_log("STARTING AUTONOMOUS DATA PREPARATION AGENT")
            self._print_log("=" * 80)
            
            # Load data
            self.dataframe = pd.read_csv(filepath)
            self.initial_shape = self.dataframe.shape
            self._print_log(f"Loaded dataset: {self.initial_shape[0]} rows × {self.initial_shape[1]} columns")
            
            # STEP 1: Dataset Validation
            self._step1_dataset_validation()
            
            # STEP 2: High Missing Value Handling
            self._step2_high_missing_value_handling()
            
            # STEP 3: Low Variance Removal
            self._step3_low_variance_removal()
            
            # STEP 4: Multicollinearity Reduction
            self._step4_multicollinearity_reduction()
            
            # STEP 5: Outlier Stabilization
            self._step5_outlier_stabilization()
            
            # STEP 6: Problem Type Detection
            self._step6_problem_type_detection()
            
            # STEP 7: Class Imbalance Detection
            if self.problem_type == "Classification":
                self._step7_class_imbalance_detection()
            
            # STEP 8: Feature Scaling
            self._step8_feature_scaling()
            
            # STEP 9: Memory Optimization
            self._step9_memory_optimization()
            
            self.final_shape = self.dataframe.shape
            
            # Generate final report
            return self._generate_final_report()
            
        except Exception as e:
            self._print_log(f"ERROR in data preparation: {str(e)}", level="ERROR")
            import traceback
            traceback.print_exc()
            return {"status": "FAILED", "error": str(e)}
    
    def _step1_dataset_validation(self):
        """STEP 1: Dataset Validation"""
        self._print_log("\n" + "="*80)
        self._print_log("STEP 1: DATASET VALIDATION")
        self._print_log("="*80)
        
        rows, cols = self.dataframe.shape
        self.add_log_entry(
            "STEP 1",
            "Dataset dimensions confirmed",
            f"Loaded dataset with {rows} rows and {cols} columns",
            confidence=1.0,
            details={"rows": rows, "columns": cols}
        )
        
        duplicate_count = self.dataframe.duplicated().sum()
        if duplicate_count > 0:
            self._print_log(f"[WARNING] Found {duplicate_count} duplicate rows")
            self.add_log_entry(
                "STEP 1",
                f"Detected {duplicate_count} duplicate rows",
                "Keeping duplicates - possible legitimate records OR data quality issue",
                confidence=0.5,
                details={"duplicate_rows": int(duplicate_count)}
            )
        else:
            self.add_log_entry(
                "STEP 1",
                "No duplicate rows detected",
                "Dataset has unique records",
                confidence=1.0,
                details={"duplicate_rows": 0}
            )
        
        self.target_column = self._detect_target_column()
        self.add_log_entry(
            "STEP 1",
            f"Target column detected: {self.target_column}",
            f"Auto-detected based on column name heuristics",
            confidence=0.8,
            details={"target_column": self.target_column}
        )
    
    def _detect_target_column(self) -> str:
        """Auto-detect target column from dataset"""
        target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'y']
        
        for col in self.dataframe.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in target_keywords):
                return col
        
        return self.dataframe.columns[-1]
    
    def _step2_high_missing_value_handling(self):
        """STEP 2: High Missing Value Handling"""
        self._print_log("\n" + "="*80)
        self._print_log("STEP 2: HIGH MISSING VALUE HANDLING")
        self._print_log("="*80)
        
        dropped_columns = []
        imputed_columns = []
        
        for col in self.dataframe.columns:
            missing_count = self.dataframe[col].isnull().sum()
            missing_pct = (missing_count / len(self.dataframe)) * 100
            
            if missing_pct == 0:
                continue
            
            self._print_log(f"Column '{col}': {missing_pct:.2f}% missing ({missing_count} rows)")
            
            if missing_pct > 35:
                self.dataframe.drop(col, axis=1, inplace=True)
                dropped_columns.append(col)
                self.add_log_entry(
                    "STEP 2",
                    f"Dropped column '{col}'",
                    f"Missing percentage {missing_pct:.2f}% exceeds 35% threshold",
                    confidence=0.95,
                    details={"column": col, "missing_percentage": missing_pct}
                )
            else:
                if self.dataframe[col].dtype in ['int64', 'float64']:
                    skewness = self.dataframe[col].skew()
                    
                    if abs(skewness) > 1:
                        impute_value = self.dataframe[col].median()
                        self.dataframe[col].fillna(impute_value, inplace=True)
                        self.add_log_entry(
                            "STEP 2",
                            f"Imputed '{col}' with MEDIAN",
                            f"Skewness {skewness:.2f} > 1 (skewed distribution), median more robust",
                            confidence=0.9,
                            details={"column": col, "method": "median", "skewness": skewness, "value": float(impute_value)}
                        )
                    else:
                        impute_value = self.dataframe[col].mean()
                        self.dataframe[col].fillna(impute_value, inplace=True)
                        self.add_log_entry(
                            "STEP 2",
                            f"Imputed '{col}' with MEAN",
                            f"Skewness {skewness:.2f} ≤ 1 (normal distribution), mean appropriate",
                            confidence=0.9,
                            details={"column": col, "method": "mean", "skewness": skewness, "value": float(impute_value)}
                        )
                    imputed_columns.append(col)
                else:
                    mode_value = self.dataframe[col].mode()[0] if not self.dataframe[col].mode().empty else "UNKNOWN"
                    self.dataframe[col].fillna(mode_value, inplace=True)
                    self.add_log_entry(
                        "STEP 2",
                        f"Imputed '{col}' with MODE",
                        f"Categorical column - most frequent value '{mode_value}' used",
                        confidence=0.85,
                        details={"column": col, "method": "mode", "value": str(mode_value)}
                    )
                    imputed_columns.append(col)
        
        self.add_log_entry(
            "STEP 2",
            "High missing value handling complete",
            f"Dropped {len(dropped_columns)} columns, imputed {len(imputed_columns)} columns",
            confidence=0.9,
            details={"dropped": dropped_columns, "imputed": imputed_columns}
        )
    
    def _step3_low_variance_removal(self):
        """STEP 3: Low Variance Removal"""
        self._print_log("\n" + "="*80)
        self._print_log("STEP 3: LOW VARIANCE REMOVAL")
        self._print_log("="*80)
        
        removed_columns = []
        variance_threshold = 0.01
        
        for col in self.dataframe.columns:
            if col == self.target_column:
                continue
            
            if self.dataframe[col].dtype in ['int64', 'float64']:
                col_variance = self.dataframe[col].var()
                
                if col_variance < variance_threshold:
                    self.dataframe.drop(col, axis=1, inplace=True)
                    removed_columns.append((col, col_variance))
                    self.add_log_entry(
                        "STEP 3",
                        f"Removed low-variance column '{col}'",
                        f"Variance {col_variance:.6f} < threshold {variance_threshold} (nearly constant)",
                        confidence=0.95,
                        details={"column": col, "variance": col_variance}
                    )
            else:
                unique_ratio = self.dataframe[col].nunique() / len(self.dataframe)
                
                if unique_ratio < 0.01:
                    self.dataframe.drop(col, axis=1, inplace=True)
                    removed_columns.append((col, unique_ratio))
                    self.add_log_entry(
                        "STEP 3",
                        f"Removed low-variance categorical '{col}'",
                        f"Unique ratio {unique_ratio:.4f} < 0.01 (nearly constant)",
                        confidence=0.95,
                        details={"column": col, "unique_ratio": unique_ratio}
                    )
        
        if removed_columns:
            self.add_log_entry(
                "STEP 3",
                f"Low variance removal complete",
                f"Removed {len(removed_columns)} near-zero variance columns",
                confidence=0.95,
                details={"removed": [col for col, _ in removed_columns]}
            )
        else:
            self.add_log_entry(
                "STEP 3",
                "No low-variance columns detected",
                "All columns have sufficient variance",
                confidence=1.0,
                details={"removed": []}
            )
    
    def _step4_multicollinearity_reduction(self):
        """STEP 4: Multicollinearity Reduction"""
        self._print_log("\n" + "="*80)
        self._print_log("STEP 4: MULTICOLLINEARITY REDUCTION")
        self._print_log("="*80)
        
        numeric_cols = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        if len(numeric_cols) < 2:
            self.add_log_entry(
                "STEP 4",
                "Multicollinearity check skipped",
                "Insufficient numeric columns for correlation analysis",
                confidence=1.0,
                details={"action": "skipped"}
            )
            return
        
        corr_matrix = self.dataframe[numeric_cols].corr().abs()
        removed_features = set()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.90:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if col_i not in removed_features and col_j not in removed_features:
                        var_i = self.dataframe[col_i].var()
                        var_j = self.dataframe[col_j].var()
                        
                        if var_i < var_j:
                            removed_col = col_i
                        else:
                            removed_col = col_j
                        
                        removed_features.add(removed_col)
                        high_corr_pairs.append((col_i, col_j, corr_value, removed_col))
                        
                        self.add_log_entry(
                            "STEP 4",
                            f"Removed '{removed_col}' (correlated with '{col_i if removed_col == col_j else col_j}')",
                            f"Correlation {corr_value:.4f} > 0.90 - Removed feature with lower variance",
                            confidence=0.95,
                            details={"column_1": col_i, "column_2": col_j, "correlation": corr_value, "removed": removed_col}
                        )
        
        self.dataframe.drop(list(removed_features), axis=1, inplace=True)
        
        if high_corr_pairs:
            self.add_log_entry(
                "STEP 4",
                "Multicollinearity reduction complete",
                f"Removed {len(removed_features)} highly correlated features",
                confidence=0.95,
                details={"removed_count": len(removed_features), "pairs": [[p[0], p[1], p[2]] for p in high_corr_pairs]}
            )
        else:
            self.add_log_entry(
                "STEP 4",
                "No multicollinearity detected",
                "No feature pairs with correlation > 0.90",
                confidence=1.0,
                details={"action": "no_removal"}
            )
    
    def _step5_outlier_stabilization(self):
        """STEP 5: Outlier Stabilization"""
        self._print_log("\n" + "="*80)
        self._print_log("STEP 5: OUTLIER STABILIZATION")
        self._print_log("="*80)
        
        numeric_cols = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        affected_columns = []
        
        for col in numeric_cols:
            Q1 = self.dataframe[col].quantile(0.25)
            Q3 = self.dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((self.dataframe[col] < lower_bound) | (self.dataframe[col] > upper_bound)).sum()
            
            if outlier_count > 0:
                self.dataframe[col] = self.dataframe[col].clip(lower=lower_bound, upper=upper_bound)
                affected_columns.append(col)
                
                outlier_pct = (outlier_count / len(self.dataframe)) * 100
                self.add_log_entry(
                    "STEP 5",
                    f"Capped outliers in '{col}'",
                    f"IQR Method: Detected {outlier_count} outliers ({outlier_pct:.2f}%), capped at ±1.5×IQR",
                    confidence=0.9,
                    details={
                        "column": col,
                        "outlier_count": int(outlier_count),
                        "outlier_percentage": outlier_pct,
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    }
                )
        
        if affected_columns:
            self.add_log_entry(
                "STEP 5",
                "Outlier stabilization complete",
                f"Capped outliers in {len(affected_columns)} numeric columns (no rows dropped)",
                confidence=0.95,
                details={"affected_columns": affected_columns}
            )
        else:
            self.add_log_entry(
                "STEP 5",
                "No significant outliers detected",
                "All numeric columns within acceptable ranges",
                confidence=0.9,
                details={"affected_columns": []}
            )
    
    def _step6_problem_type_detection(self):
        """STEP 6: Problem Type Detection"""
        self._print_log("\n" + "="*80)
        self._print_log("STEP 6: PROBLEM TYPE DETECTION")
        self._print_log("="*80)
        
        target_unique = self.dataframe[self.target_column].nunique()
        
        if target_unique < 20:
            self.problem_type = "Classification"
            confidence = 0.95 if target_unique <= 10 else 0.85
        else:
            self.problem_type = "Regression"
            confidence = 0.9
        
        self.add_log_entry(
            "STEP 6",
            f"Problem type detected: {self.problem_type}",
            f"Target column has {target_unique} unique values ({'< 20' if target_unique < 20 else '>= 20'})",
            confidence=confidence,
            details={
                "problem_type": self.problem_type,
                "target_unique_values": int(target_unique)
            }
        )
    
    def _step7_class_imbalance_detection(self):
        """STEP 7: Class Imbalance Detection"""
        self._print_log("\n" + "="*80)
        self._print_log("STEP 7: CLASS IMBALANCE DETECTION")
        self._print_log("="*80)
        
        value_counts = self.dataframe[self.target_column].value_counts()
        majority_class_ratio = value_counts.iloc[0] / len(self.dataframe)
        imbalance_percentage = (majority_class_ratio * 100)
        
        self._print_log(f"Class distribution:")
        for class_val, count in value_counts.items():
            pct = (count / len(self.dataframe)) * 100
            self._print_log(f"  {class_val}: {count} samples ({pct:.2f}%)")
        
        if imbalance_percentage > 70:
            self.add_log_entry(
                "STEP 7",
                f"Class imbalance detected",
                f"Majority class: {imbalance_percentage:.2f}% (> 70% threshold)",
                confidence=0.95,
                details={
                    "majority_class_ratio": imbalance_percentage,
                    "recommendation": "Apply SMOTE for oversampling OR class weighting in model"
                }
            )
        else:
            self.add_log_entry(
                "STEP 7",
                "Classes are relatively balanced",
                f"Majority class: {imbalance_percentage:.2f}% (≤ 70%)",
                confidence=0.95,
                details={
                    "majority_class_ratio": imbalance_percentage,
                    "status": "balanced"
                }
            )
    
    def _step8_feature_scaling(self):
        """STEP 8: Feature Scaling"""
        self._print_log("\n" + "="*80)
        self._print_log("STEP 8: FEATURE SCALING")
        self._print_log("="*80)
        
        numeric_cols = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        if len(numeric_cols) == 0:
            self.add_log_entry(
                "STEP 8",
                "Feature scaling skipped",
                "No numeric features to scale",
                confidence=1.0,
                details={"action": "skipped"}
            )
            return
        
        scaler = StandardScaler()
        self.dataframe[numeric_cols] = scaler.fit_transform(self.dataframe[numeric_cols])
        
        self.add_log_entry(
            "STEP 8",
            f"Applied StandardScaler to {len(numeric_cols)} features",
            "Features normalized to mean=0, std=1 (optimal for distance-based ML models)",
            confidence=0.95,
            details={
                "scaler": "StandardScaler",
                "scaled_features": list(numeric_cols),
                "scaling_count": len(numeric_cols)
            }
        )
    
    def _step9_memory_optimization(self):
        """STEP 9: Memory Optimization"""
        self._print_log("\n" + "="*80)
        self._print_log("STEP 9: MEMORY OPTIMIZATION")
        self._print_log("="*80)
        
        initial_memory = self.dataframe.memory_usage(deep=True).sum() / 1024**2
        
        float_cols = self.dataframe.select_dtypes(include=['float64']).columns
        for col in float_cols:
            self.dataframe[col] = self.dataframe[col].astype('float32')
        
        int_cols = self.dataframe.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if self.dataframe[col].min() >= -2147483648 and self.dataframe[col].max() <= 2147483647:
                self.dataframe[col] = self.dataframe[col].astype('int32')
        
        final_memory = self.dataframe.memory_usage(deep=True).sum() / 1024**2
        reduction_pct = ((initial_memory - final_memory) / initial_memory) * 100 if initial_memory > 0 else 0
        
        self.add_log_entry(
            "STEP 9",
            f"Memory optimization complete",
            f"Downcasted float64→float32 and int64→int32 (where safe)",
            confidence=0.95,
            details={
                "initial_memory_mb": round(initial_memory, 2),
                "final_memory_mb": round(final_memory, 2),
                "reduction_percentage": round(reduction_pct, 2),
                "float_cols_downcasted": len(float_cols),
                "int_cols_downcasted": len(int_cols)
            }
        )
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        self._print_log("\n" + "="*80)
        self._print_log("DATA PREPARATION COMPLETE")
        self._print_log("="*80)
        
        report = {
            "status": "SUCCESS",
            "timestamp": datetime.now().isoformat(),
            "dataset_summary": {
                "initial_shape": self.initial_shape,
                "final_shape": self.final_shape,
                "rows_removed": self.initial_shape[0] - self.final_shape[0],
                "columns_removed": self.initial_shape[1] - self.final_shape[1]
            },
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "preparation_steps": len(self.preparation_log),
            "all_transformations": self.preparation_log,
            "summary": self._generate_text_summary()
        }
        
        self._print_log(report['summary'])
        
        return report
    
    def _generate_text_summary(self) -> str:
        """Generate human-readable summary"""
        return f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    DATA PREPARATION SUMMARY REPORT                         ║
╚════════════════════════════════════════════════════════════════════════════╝

Dataset Shape:
  • Initial: {self.initial_shape[0]} rows × {self.initial_shape[1]} columns
  • Final:   {self.final_shape[0]} rows × {self.final_shape[1]} columns
  • Removed: {self.initial_shape[0] - self.final_shape[0]} rows, {self.initial_shape[1] - self.final_shape[1]} columns

Target Analysis:
  • Target Column: {self.target_column}
  • Problem Type: {self.problem_type}
  • Unique Values: {self.dataframe[self.target_column].nunique()}

Preprocessing Steps Applied: {len(self.preparation_log)}
  ✓ Dataset Validation
  ✓ High Missing Value Handling
  ✓ Low Variance Removal
  ✓ Multicollinearity Reduction
  ✓ Outlier Stabilization
  ✓ Problem Type Detection
  {'✓ Class Imbalance Detection' if self.problem_type == 'Classification' else ''}
  ✓ Feature Scaling
  ✓ Memory Optimization

Status: READY FOR MACHINE LEARNING
────────────────────────────────────────────────────────────────────────────
"""
    
    def save_report(self, output_path: str):
        """Save preparation report to JSON"""
        report = self._generate_final_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self._print_log(f"Report saved to: {output_path}")
        return report
    
    def save_prepared_data(self, output_path: str):
        """Save prepared dataset to CSV"""
        self.dataframe.to_csv(output_path, index=False)
        self._print_log(f"Prepared dataset saved to: {output_path}")
