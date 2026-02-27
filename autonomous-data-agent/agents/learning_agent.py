"""
AGENT 5: Feedback & Learning Agent
Stores past decisions and outcomes, learns which strategies work best.
Uses feedback to improve future recommendations.
"""
import json
import sqlite3
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from agents.base_agent import BaseAgent
from configs.agent_config import AgentConfig

class LearningAgent(BaseAgent):
    """
    Learns from past cleaning decisions and their outcomes.
    Tracks success rates and recommends best strategies for future runs.
    """
    
    def __init__(self):
        super().__init__("LearningAgent")
        self.history = []
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage for learning history"""
        storage_path = Path(AgentConfig.LEARNING_DB_PATH)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not storage_path.exists():
            # Create empty history file
            with open(storage_path, 'w') as f:
                json.dump([], f)
            self.logger.logger.info(f"Initialized learning storage at {storage_path}")
    
    def execute(self, validation_report: Dict[str, Any], 
                cleaning_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main execution method (optimized for speed)
        
        Args:
            validation_report: Report from validation agent
            cleaning_decisions: Decisions made by strategy agent
            
        Returns:
            Learning summary and recommendations for future runs
        """
        self._mark_status("running")
        start_time = datetime.now()
        
        try:
            self.logger.logger.info("Learning from cleaning outcomes...")
            
            # For large processing jobs, skip file I/O to speed up
            skip_storage = len(cleaning_decisions) > 10000
            
            if not skip_storage:
                # Record this run
                learning_record = self._create_learning_record(validation_report, cleaning_decisions)
                self._store_learning(learning_record)
                record_id = learning_record['id']
            else:
                self.logger.logger.info("Skipping learning storage for large dataset (performance optimization)")
                record_id = None
            
            # Generate recommendations (simplified for speed)
            recommendations = self._generate_simple_recommendations(validation_report)
            
            result = {
                'status': 'success',
                'learning_record_id': record_id,
                'outcome_summary': {
                    'verdict': validation_report['verdict']['status'],
                    'quality_improvement': validation_report['quality_metrics']['quality_improvement_points'],
                    'successful_strategies': len([d for d in cleaning_decisions if d.get('action') != 'KEEP'])
                },
                'recommendations': recommendations,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
            self._mark_status("completed")
            self.logger.logger.info(f"Learning completed. Generated {len(recommendations)} recommendations")
            
            return result
            
        except Exception as e:
            self.logger.log_error(str(e))
            self._mark_status("failed")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _create_learning_record(self, validation_report: Dict[str, Any], 
                                 cleaning_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a learning record from this run"""
        
        record = {
            'id': int(datetime.now().timestamp() * 1000),
            'timestamp': datetime.now().isoformat(),
            'verdict': validation_report['verdict']['status'],
            'quality_score_before': validation_report['quality_metrics']['original_quality_score'],
            'quality_score_after': validation_report['quality_metrics']['cleaned_quality_score'],
            'quality_improvement': validation_report['quality_metrics']['quality_improvement_points'],
            'completeness_improvement': (
                validation_report['quality_metrics']['cleaned_completeness_pct'] - 
                validation_report['quality_metrics']['original_completeness_pct']
            ),
            'data_retention_pct': validation_report['quality_metrics']['data_retention_pct'],
            'decisions_made': [],
            'successful': validation_report['verdict']['status'] == 'PASS'
        }
        
        # Store decision details
        decision_summary = {}
        for decision in cleaning_decisions:
            action = decision.get('action')
            if action not in decision_summary:
                decision_summary[action] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'success_rate': 0
                }
            decision_summary[action]['count'] += 1
            decision_summary[action]['avg_confidence'] += decision.get('confidence_score', 0)
        
        # Calculate averages
        for action in decision_summary:
            decision_summary[action]['avg_confidence'] /= decision_summary[action]['count']
        
        record['decision_summary'] = decision_summary
        
        self.logger.logger.info(f"Created learning record: {record['id']}")
        
        return record
    
    def _store_learning(self, record: Dict[str, Any]):
        """Store learning record to persistence layer"""
        
        storage_path = Path(AgentConfig.LEARNING_DB_PATH)
        
        try:
            # Load existing history
            with open(storage_path, 'r') as f:
                history = json.load(f)
            
            # Add new record
            history.append(record)
            
            # Keep only recent history
            if len(history) > AgentConfig.MAX_HISTORY_RECORDS:
                history = history[-AgentConfig.MAX_HISTORY_RECORDS:]
            
            # Save back
            with open(storage_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.logger.logger.info(f"Stored learning record. Total records: {len(history)}")
            
        except Exception as e:
            self.logger.log_error(f"Failed to store learning: {str(e)}")
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns from historical data"""
        
        storage_path = Path(AgentConfig.LEARNING_DB_PATH)
        
        try:
            with open(storage_path, 'r') as f:
                history = json.load(f)
            
            if not history:
                return {'status': 'insufficient_data', 'message': 'No historical data yet'}
            
            # Calculate success rate
            successful_runs = len([r for r in history if r.get('successful')])
            total_runs = len(history)
            success_rate = (successful_runs / total_runs) * 100 if total_runs > 0 else 0
            
            # Calculate average quality improvement
            improvements = [r.get('quality_improvement', 0) for r in history]
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0
            
            # Calculate average data retention
            retentions = [r.get('data_retention_pct', 100) for r in history]
            avg_retention = sum(retentions) / len(retentions) if retentions else 100
            
            # Analyze decision effectiveness
            all_decisions = {}
            for record in history:
                for action, stats in record.get('decision_summary', {}).items():
                    if action not in all_decisions:
                        all_decisions[action] = {
                            'count': 0,
                            'successful_runs': 0,
                            'avg_confidence': 0,
                            'avg_improvement': 0
                        }
                    all_decisions[action]['count'] += stats['count']
                    if record['successful']:
                        all_decisions[action]['successful_runs'] += stats['count']
                    all_decisions[action]['avg_confidence'] += stats['avg_confidence']
                    all_decisions[action]['avg_improvement'] += record.get('quality_improvement', 0)
            
            # Calculate decision success rates
            for action in all_decisions:
                if all_decisions[action]['count'] > 0:
                    all_decisions[action]['success_rate'] = (
                        all_decisions[action]['successful_runs'] / all_decisions[action]['count'] * 100
                    )
                    all_decisions[action]['avg_confidence'] /= all_decisions[action]['count']
                    all_decisions[action]['avg_improvement'] /= all_decisions[action]['count']
            
            analysis = {
                'status': 'success',
                'total_historical_runs': total_runs,
                'successful_runs': successful_runs,
                'success_rate_pct': success_rate,
                'avg_quality_improvement': avg_improvement,
                'avg_data_retention_pct': avg_retention,
                'decision_effectiveness': all_decisions
            }
            
            self.logger.log_decision(
                column='[LEARNING]',
                decision='PATTERNS_ANALYZED',
                reasoning=f"Historical success rate: {success_rate:.1f}%, Avg improvement: {avg_improvement:.2f} points",
                confidence=0.95
            )
            
            return analysis
            
        except Exception as e:
            self.logger.log_error(f"Error analyzing patterns: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategy recommendations based on patterns"""
        
        recommendations = []
        
        if analysis.get('status') != 'success':
            return [{'recommendation': 'Insufficient historical data. Continue collecting patterns.', 'priority': 'low'}]
        
        # Recommend based on success rate
        if analysis['success_rate_pct'] < 50:
            recommendations.append({
                'recommendation': 'Cleaning strategies need improvement. Consider more conservative thresholds.',
                'priority': 'high',
                'rationale': f"Current success rate: {analysis['success_rate_pct']:.1f}%"
            })
        
        # Recommend most effective strategies
        decision_effectiveness = analysis.get('decision_effectiveness', {})
        best_performing = max(
            decision_effectiveness.items(),
            key=lambda x: x[1]['success_rate'] if x[1]['count'] > 5 else 0,
            default=None
        )
        
        if best_performing and best_performing[1]['count'] > 5:
            recommendations.append({
                'recommendation': f"Strategy '{best_performing[0]}' has highest success rate ({best_performing[1]['success_rate']:.1f}%). Consider prioritizing this.",
                'priority': 'medium',
                'rationale': f"Effective in {best_performing[1]['successful_runs']} of {best_performing[1]['count']} cases"
            })
        
        # Identify underperforming strategies
        worst_performing = min(
            decision_effectiveness.items(),
            key=lambda x: x[1]['success_rate'] if x[1]['count'] > 3 else 100,
            default=None
        )
        
        if worst_performing and worst_performing[1]['count'] > 3 and worst_performing[1]['success_rate'] < 50:
            recommendations.append({
                'recommendation': f"Strategy '{worst_performing[0]}' has low success rate ({worst_performing[1]['success_rate']:.1f}%). Consider alternatives.",
                'priority': 'medium',
                'rationale': f"Failed in {worst_performing[1]['count'] - worst_performing[1]['successful_runs']} out of {worst_performing[1]['count']} cases"
            })
        
        # Data retention warning
        if analysis['avg_data_retention_pct'] < 70:
            recommendations.append({
                'recommendation': 'Average data retention is low. More data is being lost during cleaning.',
                'priority': 'medium',
                'rationale': f"Average retention: {analysis['avg_data_retention_pct']:.1f}%"
            })
        
        if not recommendations:
            recommendations.append({
                'recommendation': 'Current strategies are performing well. Continue as-is.',
                'priority': 'low',
                'rationale': f"Success rate: {analysis['success_rate_pct']:.1f}%, Avg improvement: {analysis['avg_quality_improvement']:.2f} points"
            })
        
        self.logger.logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def _generate_simple_recommendations(self, validation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations quickly without file I/O (for large datasets)"""
        recommendations = []
        
        quality_improvement = validation_report['quality_metrics']['quality_improvement_points']
        verdict = validation_report['verdict']['status']
        completeness = validation_report['quality_metrics']['cleaned_completeness_pct']
        
        # Recommendation based on quality improvement
        if quality_improvement > 15:
            recommendations.append({
                'recommendation': f'Excellent quality improvement (+{quality_improvement:.1f} points). Current strategy is highly effective.',
                'priority': 'low',
                'rationale': f"Quality score increased significantly from {validation_report['quality_metrics']['original_quality_score']:.1f} to {validation_report['quality_metrics']['cleaned_quality_score']:.1f}"
            })
        elif quality_improvement < 5:
            recommendations.append({
                'recommendation': 'Modest quality improvement. Consider adjusting cleaning thresholds.',
                'priority': 'medium',
                'rationale': f"Quality improvement was only {quality_improvement:.1f} points"
            })
        
        # Recommendation based on completeness
        if completeness < 70:
            recommendations.append({
                'recommendation': 'Data completeness is low. Review imputation strategies.',
                'priority': 'high',
                'rationale': f"Only {completeness:.1f}% of data cells are complete"
            })
        
        # Recommendation based on verdict
        if verdict != 'PASS':
            recommendations.append({
                'recommendation': 'Cleaning result did not pass validation. Review strategy and thresholds.',
                'priority': 'high',
                'rationale': f"Verdict: {verdict}"
            })
        
        if not recommendations:
            recommendations.append({
                'recommendation': 'Cleaning process completed successfully. Continue with current approach.',
                'priority': 'low',
                'rationale': f"Quality: {validation_report['quality_metrics']['cleaned_quality_score']:.1f}/100, Completeness: {completeness:.1f}%"
            })
        
        return recommendations
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent learning history"""
        
        storage_path = Path(AgentConfig.LEARNING_DB_PATH)
        
        try:
            with open(storage_path, 'r') as f:
                history = json.load(f)
            
            return history[-limit:]  # Return last N records
            
        except Exception as e:
            self.logger.log_error(f"Error retrieving history: {str(e)}")
            return []
