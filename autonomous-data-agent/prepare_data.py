#!/usr/bin/env python3
"""
Execute Data Preparation Agent on CSV files
Converts any dataset into fully model-ready format
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.data_preparation_agent import DataPreparationAgent

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Data Preparation Agent - Convert datasets to ML-ready format"
    )
    parser.add_argument(
        "input_file",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save prepared dataset (default: data/reports/prepared_data.csv)",
        default=None
    )
    parser.add_argument(
        "-r", "--report",
        help="Path to save preparation report (default: data/reports/preparation_report.json)",
        default=None
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Set default output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"data/reports/prepared_data_{timestamp}.csv"
    report_file = args.report or f"data/reports/preparation_report_{timestamp}.json"
    
    # Create output directories
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AUTONOMOUS DATA PREPARATION AGENT")
    print("=" * 80)
    print(f"Input File:  {args.input_file}")
    print(f"Output File: {output_file}")
    print(f"Report File: {report_file}")
    print("=" * 80)
    print()
    
    # Execute preparation agent
    agent = DataPreparationAgent()
    
    try:
        report = agent.execute(args.input_file)
        
        if report["status"] == "SUCCESS":
            # Save prepared data
            agent.save_prepared_data(output_file)
            
            # Save report
            agent.save_report(report_file)
            
            print()
            print("=" * 80)
            print("✓ DATA PREPARATION SUCCESSFUL")
            print("=" * 80)
            print(f"✓ Prepared dataset saved to: {output_file}")
            print(f"✓ Report saved to: {report_file}")
            print()
            
            # Print summary
            summary = report["dataset_summary"]
            print(f"Initial shape: {summary['initial_shape'][0]} rows × {summary['initial_shape'][1]} columns")
            print(f"Final shape:   {summary['final_shape'][0]} rows × {summary['final_shape'][1]} columns")
            print(f"Rows removed: {summary['rows_removed']}")
            print(f"Columns removed: {summary['columns_removed']}")
            print()
            print(f"Problem Type: {report['problem_type']}")
            print(f"Transformation Steps: {report['preparation_steps']}")
            print()
            
            sys.exit(0)
        else:
            print(f"✗ Data preparation failed: {report.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
