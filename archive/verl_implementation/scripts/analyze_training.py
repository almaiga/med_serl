#!/usr/bin/env python3
"""
Training Metrics Analysis Tool
Extracts and visualizes key metrics from VERL training logs (WandB runs)
Follows the Hugging Face GRPO blog analysis patterns.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

try:
    import wandb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("❌ Missing dependencies. Install: pip install wandb pandas numpy matplotlib seaborn")
    sys.exit(1)


class TrainingMetricsAnalyzer:
    """Analyze training metrics from WandB runs."""
    
    def __init__(self, project_name: str = "medserl-selfplay", entity: Optional[str] = None):
        """
        Initialize analyzer.
        
        Args:
            project_name: WandB project name
            entity: WandB entity (username/team)
        """
        self.project = project_name
        self.entity = entity
        self.api = wandb.Api()
        
    def fetch_run_history(self, run_name: str, limit: int = None) -> pd.DataFrame:
        """
        Fetch complete history from a WandB run.
        
        Args:
            run_name: Run name or ID
            limit: Max rows to fetch (None = all)
            
        Returns:
            DataFrame with all logged metrics
        """
        print(f"📥 Fetching metrics from run: {run_name}")
        
        # Build project path
        project_path = f"{self.entity}/{self.project}" if self.entity else self.project
        
        try:
            run = self.api.run(f"{project_path}/{run_name}")
        except Exception as e:
            print(f"❌ Error fetching run: {e}")
            return None
        
        # Convert history to DataFrame
        history = []
        for row in run.history():
            history.append(row)
            if limit and len(history) >= limit:
                break
        
        df = pd.DataFrame(history)
        print(f"✅ Fetched {len(df)} data points")
        return df
    
    def analyze_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Compute key metrics statistics.
        
        Args:
            df: DataFrame with training history
            
        Returns:
            Dict with analysis results
        """
        analysis = {}
        
        # System Health
        if 'actor_rollout_ref_reward/actor_score' in df.columns:
            rewards = df['actor_rollout_ref_reward/actor_score'].dropna()
            analysis['actor_score'] = {
                'mean': rewards.mean(),
                'max': rewards.max(),
                'min': rewards.min(),
                'final': rewards.iloc[-1] if len(rewards) > 0 else None,
                'improvement': rewards.iloc[-1] - rewards.iloc[0] if len(rewards) > 1 else None
            }
        
        # KL Divergence (Policy Drift)
        if 'actor_rollout_ref_reward/kl_loss' in df.columns:
            kl = df['actor_rollout_ref_reward/kl_loss'].dropna()
            analysis['kl_divergence'] = {
                'mean': kl.mean(),
                'max': kl.max(),
                'min': kl.min(),
                'final': kl.iloc[-1] if len(kl) > 0 else None,
                'stable': kl.max() < 0.01  # Should stay bounded
            }
        
        # Loss metrics
        for loss_key in ['actor_loss', 'critic_loss', 'loss']:
            if loss_key in df.columns:
                loss = df[loss_key].dropna()
                analysis[loss_key] = {
                    'mean': loss.mean(),
                    'max': loss.max(),
                    'min': loss.min(),
                    'final': loss.iloc[-1] if len(loss) > 0 else None,
                    'converged': loss.std() < loss.mean() * 0.1 if len(loss) > 0 else None
                }
        
        # Gradient norm
        if 'actor_rollout_ref_actor/grad_norm' in df.columns:
            grad = df['actor_rollout_ref_actor/grad_norm'].dropna()
            analysis['gradient_norm'] = {
                'mean': grad.mean(),
                'max': grad.max(),
                'min': grad.min(),
                'spikes': (grad > grad.quantile(0.95)).sum(),  # Count outliers
                'stable': grad.max() < grad.mean() * 10  # No explosive growth
            }
        
        # Training progress
        if 'global_step' in df.columns:
            analysis['training_progress'] = {
                'total_steps': df['global_step'].max(),
                'rows_logged': len(df)
            }
        
        return analysis
    
    def print_report(self, run_name: str, analysis: Dict) -> None:
        """Print formatted analysis report."""
        print("\n" + "="*70)
        print(f"📊 TRAINING ANALYSIS REPORT: {run_name}")
        print("="*70 + "\n")
        
        if 'actor_score' in analysis:
            print("🎯 REWARD PROGRESSION (Alignment with Objective)")
            print("   " + "-"*50)
            score = analysis['actor_score']
            print(f"   Start:       {score['min']:.4f}")
            print(f"   End:         {score['final']:.4f}")
            print(f"   Improvement: {score['improvement']:.4f} ✅" if score['improvement'] > 0 else f"   Improvement: {score['improvement']:.4f} ⚠️")
            print(f"   Max:         {score['max']:.4f}\n")
        
        if 'kl_divergence' in analysis:
            print("🔀 POLICY DRIFT (KL Divergence)")
            print("   " + "-"*50)
            kl = analysis['kl_divergence']
            print(f"   Mean:        {kl['mean']:.6f}")
            print(f"   Max:         {kl['max']:.6f}")
            print(f"   Final:       {kl['final']:.6f}")
            status = "✅ Stable" if kl['stable'] else "⚠️ Warning: High drift"
            print(f"   Status:      {status}\n")
        
        if 'gradient_norm' in analysis:
            print("📈 GRADIENT STABILITY")
            print("   " + "-"*50)
            grad = analysis['gradient_norm']
            print(f"   Mean:        {grad['mean']:.6f}")
            print(f"   Max:         {grad['max']:.6f}")
            print(f"   Outliers:    {grad['spikes']} spikes detected")
            status = "✅ Stable" if grad['stable'] else "⚠️ Warning: Potential instability"
            print(f"   Status:      {status}\n")
        
        if 'actor_loss' in analysis:
            print("💧 ACTOR LOSS")
            print("   " + "-"*50)
            loss = analysis['actor_loss']
            print(f"   Mean:        {loss['mean']:.6f}")
            print(f"   Final:       {loss['final']:.6f}")
            print(f"   Converged:   {'✅ Yes' if loss['converged'] else '⚠️ No'}\n")
        
        if 'critic_loss' in analysis:
            print("🎪 CRITIC LOSS")
            print("   " + "-"*50)
            loss = analysis['critic_loss']
            print(f"   Mean:        {loss['mean']:.6f}")
            print(f"   Final:       {loss['final']:.6f}\n")
        
        if 'training_progress' in analysis:
            print("⏱️  TRAINING PROGRESS")
            print("   " + "-"*50)
            prog = analysis['training_progress']
            print(f"   Total Steps: {prog['total_steps']}")
            print(f"   Data Points: {prog['rows_logged']}\n")
        
        print("="*70)
    
    def create_dashboard_url(self, run_id: str) -> str:
        """Generate WandB dashboard URL."""
        project_path = f"{self.entity}/{self.project}" if self.entity else self.project
        return f"https://wandb.ai/{project_path}/runs/{run_id}"


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(
        description="Analyze VERL training metrics from WandB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest run
  python analyze_training.py
  
  # Analyze specific run
  python analyze_training.py --run-id abc123def456
  
  # Compare multiple runs
  python analyze_training.py --run-id run1_id run2_id run3_id
        """
    )
    
    parser.add_argument(
        "--project", 
        default="medserl-selfplay",
        help="WandB project name (default: medserl-selfplay)"
    )
    parser.add_argument(
        "--entity",
        help="WandB entity/username"
    )
    parser.add_argument(
        "--run-id",
        nargs="+",
        help="Run ID(s) to analyze (if not provided, fetches latest)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit rows to fetch (for large runs)"
    )
    parser.add_argument(
        "--export-csv",
        help="Export metrics to CSV file"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TrainingMetricsAnalyzer(project_name=args.project, entity=args.entity)
    
    try:
        # If no run ID specified, fetch latest
        if not args.run_id:
            print("📍 Fetching latest run from project...")
            project_path = f"{args.entity}/{args.project}" if args.entity else args.project
            runs = analyzer.api.runs(project_path, per_page=1)
            if not runs:
                print("❌ No runs found in project")
                return
            run_ids = [runs[0].id]
        else:
            run_ids = args.run_id
        
        # Analyze each run
        all_data = {}
        for run_id in run_ids:
            df = analyzer.fetch_run_history(run_id, limit=args.limit)
            if df is not None:
                analysis = analyzer.analyze_metrics(df)
                analyzer.print_report(run_id, analysis)
                all_data[run_id] = (df, analysis)
        
        # Export if requested
        if args.export_csv and all_data:
            run_id = list(all_data.keys())[0]
            df, _ = all_data[run_id]
            df.to_csv(args.export_csv, index=False)
            print(f"✅ Exported metrics to: {args.export_csv}")
        
        # Print dashboard URLs
        print("\n📊 WandB Dashboard:")
        for run_id in run_ids:
            url = analyzer.create_dashboard_url(run_id)
            print(f"   {url}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
