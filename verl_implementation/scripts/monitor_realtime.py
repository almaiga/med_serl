#!/usr/bin/env python3
"""
Real-time Training Monitor
Watches VERL training logs and displays key metrics in terminal
Run in a separate terminal alongside training
"""

import sys
import time
import json
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Dict, Optional

try:
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
except ImportError:
    print("❌ Missing 'rich' package. Install: pip install rich")
    sys.exit(1)


class TrainingMonitor:
    """Real-time monitor for VERL training."""
    
    def __init__(self, log_dir: Path, update_interval: float = 5.0):
        """
        Initialize monitor.
        
        Args:
            log_dir: Output directory containing checkpoints
            update_interval: Update frequency in seconds
        """
        self.log_dir = Path(log_dir)
        self.update_interval = update_interval
        self.console = Console()
        self.metrics_history = deque(maxlen=100)
        self.start_time = datetime.now()
        
    def find_latest_log(self) -> Optional[Path]:
        """Find latest JSON log file."""
        json_files = sorted(self.log_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)
        return json_files[-1] if json_files else None
    
    def read_checkpoint_info(self) -> Optional[Dict]:
        """Read info from latest checkpoint."""
        checkpoint_dirs = sorted(
            [d for d in self.log_dir.iterdir() if d.is_dir() and d.name.startswith("global_step")],
            key=lambda x: int(x.name.split("_")[-1])
        )
        
        if not checkpoint_dirs:
            return None
        
        latest_ckpt = checkpoint_dirs[-1]
        info_file = latest_ckpt / "trainer_state.json"
        
        if info_file.exists():
            try:
                with open(info_file) as f:
                    return json.load(f)
            except:
                return None
        
        return None
    
    def format_time_remaining(self, current_step: int, total_steps: int, elapsed: float) -> str:
        """Estimate time remaining."""
        if current_step == 0:
            return "N/A"
        
        time_per_step = elapsed / current_step
        remaining_steps = total_steps - current_step
        remaining_secs = time_per_step * remaining_steps
        
        hours = int(remaining_secs // 3600)
        minutes = int((remaining_secs % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    
    def create_status_table(self, checkpoint_info: Dict) -> Table:
        """Create rich table with current status."""
        table = Table(title="🚀 Training Status", show_header=False, box=None)
        table.add_column(style="cyan", width=25)
        table.add_column(style="green", width=30)
        
        if checkpoint_info:
            current_step = checkpoint_info.get('global_step', 0)
            current_epoch = checkpoint_info.get('epoch', 0)
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            elapsed_hours = elapsed / 3600
            
            table.add_row("📍 Current Step", str(current_step))
            table.add_row("📚 Current Epoch", str(current_epoch))
            table.add_row("⏱️  Elapsed Time", f"{elapsed_hours:.1f}h")
            
        return table
    
    def create_metrics_table(self) -> Table:
        """Create metrics display table."""
        table = Table(title="📊 Recent Metrics", box="simple")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        if len(self.metrics_history) > 0:
            latest = self.metrics_history[-1]
            
            # Reward score
            if 'reward_score' in latest:
                table.add_row(
                    "Reward Mean",
                    f"{latest['reward_score']:.4f}",
                    "📈 Improving" if len(self.metrics_history) > 1 and latest['reward_score'] > self.metrics_history[-2].get('reward_score', 0) else "➡️  Stable"
                )
            
            # KL divergence
            if 'kl_div' in latest:
                status = "✅ Bounded" if latest['kl_div'] < 0.01 else "⚠️  High"
                table.add_row("KL Divergence", f"{latest['kl_div']:.6f}", status)
            
            # Loss
            if 'loss' in latest:
                table.add_row("Actor Loss", f"{latest['loss']:.6f}", "➡️  Updating")
            
            # GPU stats
            if 'gpu_mem' in latest:
                table.add_row("GPU Memory", f"{latest['gpu_mem']:.1f}%", "📊 Monitoring")
        
        return table
    
    def display(self):
        """Display real-time monitoring dashboard."""
        self.console.clear()
        
        while True:
            try:
                # Fetch latest info
                checkpoint_info = self.read_checkpoint_info()
                
                # Create display
                status_table = self.create_status_table(checkpoint_info)
                metrics_table = self.create_metrics_table()
                
                # Print
                self.console.print(status_table)
                self.console.print(metrics_table)
                
                # Instructions
                info_panel = Panel(
                    "[cyan]💡 Tips:[/cyan]\n"
                    "  • [yellow]nvitop[/yellow] - GPU utilization in another terminal\n"
                    "  • [yellow]WandB[/yellow] - Real-time charts (https://wandb.ai/)\n"
                    "  • [yellow]Press Ctrl+C[/yellow] - Stop monitoring\n",
                    title="📖 Guide"
                )
                self.console.print(info_panel)
                
                time.sleep(self.update_interval)
                self.console.clear()
                
            except KeyboardInterrupt:
                self.console.print("[yellow]⏹️  Monitoring stopped.[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]❌ Error: {e}[/red]")
                time.sleep(self.update_interval)


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time VERL training monitor")
    parser.add_argument(
        "--log-dir",
        default="outputs/verl_training",
        help="Training output directory"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Update interval in seconds"
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return 1
    
    monitor = TrainingMonitor(log_dir, update_interval=args.interval)
    monitor.display()
    return 0


if __name__ == "__main__":
    sys.exit(main())
