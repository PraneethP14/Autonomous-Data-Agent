"""
Agent Orchestrator
Coordinates all 5 agents to execute the full autonomous data cleaning pipeline.
THIS IS THE MAIN COORDINATOR OF THE ENTIRE SYSTEM.
"""
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import json
from pathlib import Path

from agents import (
    DataProfilingAgent,
    CleaningStrategyAgent,
    CleaningExecutionAgent,
    ValidationAgent,
    LearningAgent
)

class AgentOrchestrator:
    """
    Main orchestrator that runs the complete autonomous data cleaning pipeline.
    
    Pipeline Flow:
    1. DataProfilingAgent → Profile raw data
    2. CleaningStrategyAgent → Decide cleaning strategy
    3. CleaningExecutionAgent → Execute cleaning
    4. ValidationAgent → Validate & score results
    5. LearningAgent → Learn from outcomes
    """
    
    def __init__(self):
        self.agents = {
            'profiling': DataProfilingAgent(),
            'strategy': CleaningStrategyAgent(),
            'execution': CleaningExecutionAgent(),
            'validation': ValidationAgent(),
            'learning': LearningAgent()
        }
        self.pipeline_log = {
            'started_at': None,
            'completed_at': None,
            'duration': 0,
            'stages': {},
            'final_verdict': None,
            'status': 'not_started'
        }
    
    def run_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the full autonomous cleaning pipeline
        
        Args:
            df: Raw pandas DataFrame
            
        Returns:
            Complete pipeline result with cleaned data and reports
        """
        self.pipeline_log['started_at'] = datetime.now().isoformat()
        self.pipeline_log['status'] = 'running'
        
        print("\n" + "="*80)
        print("[*] AUTONOMOUS DATA CLEANING PIPELINE STARTED")
        print("="*80 + "\n")
        
        try:
            # STAGE 1: Data Profiling
            print("[STAGE 1] DATA PROFILING")
            print("-" * 80)
            profiling_result = self._run_stage('profiling', self.agents['profiling'].execute, df)
            if profiling_result['status'] != 'success':
                return self._pipeline_failed(profiling_result)
            
            profile = profiling_result['result']['profile']
            print(f"[+] Profile complete: {profile['dataset_shape']['rows']} rows, {profile['dataset_shape']['columns']} columns")
            print(f"    Quality Score: {profile['quality_metrics']['quality_score']:.2f}/100\n")
            
            # STAGE 2: Strategy Generation
            print("[STAGE 2] CLEANING STRATEGY")
            print("-" * 80)
            strategy_result = self._run_stage('strategy', self.agents['strategy'].execute, profile, df)
            if strategy_result['status'] != 'success':
                return self._pipeline_failed(strategy_result)
            
            plan = strategy_result['result']['plan']
            print(f"[+] Strategy complete: {plan['summary']['total_actions']} column actions")
            print(f"    - Drop: {plan['summary']['drop_columns']}")
            print(f"    - Impute: {plan['summary']['impute_actions']}")
            print(f"    - Row actions: {plan['summary']['row_actions']}\n")
            
            # STAGE 3: Execution
            print("[STAGE 3] DATA CLEANING EXECUTION")
            print("-" * 80)
            execution_result = self._run_stage(
                'execution', 
                self.agents['execution'].execute, 
                df, 
                plan
            )
            if execution_result['status'] != 'success':
                return self._pipeline_failed(execution_result)
            
            cleaned_df = execution_result['result']['cleaned_df']
            trans_log = execution_result['result']['log']
            print(f"[+] Cleaning complete: {df.shape} -> {cleaned_df.shape}")
            print(f"    - Rows removed: {execution_result['result']['rows_removed']}")
            print(f"    - Columns removed: {execution_result['result']['columns_removed']}")
            print(f"    - Transformations: {execution_result['result']['transformations']}\n")
            
            # STAGE 4: Validation
            print("[STAGE 4] VALIDATION & QUALITY ASSESSMENT")
            print("-" * 80)
            validation_result = self._run_stage(
                'validation',
                self.agents['validation'].execute,
                df,
                cleaned_df,
                trans_log
            )
            if validation_result['status'] != 'success':
                return self._pipeline_failed(validation_result)
            
            validation_report = validation_result['result']['report']
            verdict = validation_report['verdict']
            metrics = validation_report['quality_metrics']
            
            print(f"[+] Validation complete: {verdict['status']}")
            print(f"    - Quality: {metrics['original_quality_score']:.2f} -> {metrics['cleaned_quality_score']:.2f}")
            print(f"    - Improvement: {metrics['quality_improvement_points']:+.2f} points")
            print(f"    - Completeness: {metrics['cleaned_completeness_pct']:.2f}%")
            print(f"    - Data Retention: {metrics['data_retention_pct']:.2f}%\n")
            
            # STAGE 5: Learning
            print("[STAGE 5] FEEDBACK & LEARNING")
            print("-" * 80)
            learning_result = self._run_stage(
                'learning',
                self.agents['learning'].execute,
                validation_report,
                plan['actions']
            )
            if learning_result['status'] != 'success':
                print(f"[!] Learning stage warning: {learning_result['result']['error']}")
            else:
                recommendations = learning_result['result']['recommendations']
                print(f"[+] Learning complete: Generated {len(recommendations)} recommendations")
                for i, rec in enumerate(recommendations[:2], 1):
                    print(f"    {i}. {rec['recommendation']}")
            
            print("\n" + "="*80)
            print("[+] PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80 + "\n")
            
            # Prepare final result
            final_result = {
                'status': 'success',
                'pipeline_status': 'COMPLETED',
                'verdict': verdict['status'],
                'original_shape': df.shape,
                'cleaned_shape': cleaned_df.shape,
                'quality_metrics': metrics,
                'cleaned_df': cleaned_df,
                'reports': {
                    'profiling': profile,
                    'strategy': plan,
                    'transformation_log': trans_log,
                    'validation': validation_report,
                    'learning': learning_result['result'] if learning_result['status'] == 'success' else None
                },
                'agent_logs': {
                    'profiling': self.agents['profiling'].get_execution_log(),
                    'strategy': self.agents['strategy'].get_execution_log(),
                    'execution': self.agents['execution'].get_execution_log(),
                    'validation': self.agents['validation'].get_execution_log(),
                    'learning': self.agents['learning'].get_execution_log()
                }
            }
            
            self.pipeline_log['status'] = 'completed'
            self.pipeline_log['final_verdict'] = verdict['status']
            self.pipeline_log['completed_at'] = datetime.now().isoformat()
            final_result['pipeline_log'] = self.pipeline_log
            
            return final_result
            
        except Exception as e:
            print(f"\n[-] PIPELINE FAILED: {str(e)}")
            return self._pipeline_failed({'status': 'error', 'error': str(e)})
    
    def _run_stage(self, stage_name: str, stage_func, *args) -> Dict[str, Any]:
        """Run a pipeline stage with error handling"""
        try:
            result = stage_func(*args)
            self.pipeline_log['stages'][stage_name] = {
                'status': result['status'],
                'duration': result.get('execution_time', 0)
            }
            return {'status': result['status'], 'result': result}
        except Exception as e:
            self.pipeline_log['stages'][stage_name] = {
                'status': 'error',
                'error': str(e)
            }
            return {'status': 'error', 'result': {'error': str(e)}}
    
    def _pipeline_failed(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pipeline failure"""
        self.pipeline_log['status'] = 'failed'
        self.pipeline_log['completed_at'] = datetime.now().isoformat()
        
        return {
            'status': 'error',
            'pipeline_status': 'FAILED',
            'error': error.get('error', 'Unknown error'),
            'pipeline_log': self.pipeline_log,
            'cleaned_df': None
        }
    
    def save_results(self, result: Dict[str, Any], output_dir: str = 'data/reports'):
        """Save pipeline results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save cleaned data
            if result['cleaned_df'] is not None:
                cleaned_path = output_path / f"cleaned_data_{timestamp}.csv"
                result['cleaned_df'].to_csv(cleaned_path, index=False)
                print(f"[+] Cleaned data saved: {cleaned_path}")
            
            # Save validation report
            if 'reports' in result and result['reports']['validation']:
                report_path = output_path / f"validation_report_{timestamp}.json"
                with open(report_path, 'w') as f:
                    # Replace DataFrames with strings for JSON serialization
                    json.dump(self._make_serializable(result['reports']['validation']), f, indent=2)
                print(f"[+] Validation report saved: {report_path}")
            
            # Save agent decision logs
            if 'agent_logs' in result:
                logs_path = output_path / f"agent_logs_{timestamp}.json"
                with open(logs_path, 'w') as f:
                    json.dump(self._make_serializable(result['agent_logs']), f, indent=2)
                print(f"[+] Agent logs saved: {logs_path}")
            
            print(f"\n[+] All outputs saved to: {output_path}")
            
        except Exception as e:
            print(f"[!] Error saving results: {str(e)}")
    
    @staticmethod
    def _make_serializable(obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: AgentOrchestrator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [AgentOrchestrator._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, (float, int)):
            if isinstance(obj, float) and (obj != obj or obj == float('inf')):  # NaN or inf
                return None
            return obj
        elif obj is None:
            return None
        else:
            return str(obj)
