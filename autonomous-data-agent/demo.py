"""
Demo script for the Autonomous Data Cleaning Agent
Shows how to use the system programmatically
"""
import pandas as pd
from pathlib import Path
import json
import sys
import os

# Fix encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from agents.orchestrator import AgentOrchestrator
from generate_samples import create_sample_datasets

def main():
    """Run a complete demo"""
    
    print("\n" + "="*80)
    print("[*] AUTONOMOUS DATA CLEANING AGENT - DEMO")
    print("="*80)
    
    # Step 1: Generate sample datasets
    print("\n[*] STEP 1: Generating sample datasets...")
    ecom_path, med_path = create_sample_datasets()
    
    # Step 2: Load a sample dataset
    print("\n[*] STEP 2: Loading sample dataset...")
    df = pd.read_csv(ecom_path)
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData info:")
    print(df.info())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Step 3: Run the agent pipeline
    print("\n[*] STEP 3: Running Autonomous Agent Pipeline...")
    orchestrator = AgentOrchestrator()
    result = orchestrator.run_pipeline(df)
    
    # Step 4: Display results
    print("\n[*] STEP 4: Analyzing Results...")
    
    if result['status'] == 'success':
        cleaned_df = result['cleaned_df']
        metrics = result['quality_metrics']
        verdict = result['verdict']
        
        print(f"\n[+] Pipeline Status: SUCCESS")
        print(f"\n[+] VERDICT: {verdict}")
        print(f"\n[+] Quality Metrics:")
        print(f"    Quality Score: {metrics['original_quality_score']:.2f} -> {metrics['cleaned_quality_score']:.2f}")
        print(f"    Improvement: {metrics['quality_improvement_points']:+.2f} points")
        print(f"    Completeness: {metrics['original_completeness_pct']:.2f}% -> {metrics['cleaned_completeness_pct']:.2f}%")
        print(f"    Data Retention: {metrics['data_retention_pct']:.2f}%")
        
        print(f"\n[+] Shape Improvement:")
        print(f"    Original: {result['original_shape']}")
        print(f"    Cleaned: {result['cleaned_shape']}")
        print(f"    Rows removed: {result['original_shape'][0] - result['cleaned_shape'][0]}")
        print(f"    Columns removed: {result['original_shape'][1] - result['cleaned_shape'][1]}")
        
        print(f"\nCleaned data (first 5 rows):")
        print(cleaned_df.head())
        
        print(f"\nCleaned data info:")
        print(cleaned_df.info())
        
        # Step 5: Save results
        print("\n[*] STEP 5: Saving Results...")
        orchestrator.save_results(result)
        
        # Show agent decision logs
        print("\n[*] STEP 6: Agent Decision Logs Summary...")
        agent_logs = result.get('agent_logs', {})
        
        for agent_name, log in agent_logs.items():
            if log and log.get('decisions'):
                print(f"\n{agent_name.upper()} Agent Decisions:")
                for decision in log['decisions'][:3]:  # Show first 3
                    print(f"  - [{decision['column']}] {decision['decision']}")
                    print(f"    Reasoning: {decision['reasoning']}")
                    print(f"    Confidence: {decision['confidence']:.2f}")
        
        # Show learning insights
        if 'learning' in result['reports'] and result['reports']['learning']:
            learning = result['reports']['learning']
            print(f"\n[+] Learning Insights:")
            if 'recommendations' in learning:
                for i, rec in enumerate(learning['recommendations'][:3], 1):
                    print(f"  {i}. {rec['recommendation']}")
    
    else:
        print(f"\n[-] Pipeline Error: {result.get('error')}")
    
    print("\n" + "="*80)
    print("[+] Demo Complete!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
