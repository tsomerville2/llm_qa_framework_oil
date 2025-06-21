#!/usr/bin/env python3
"""
Scoring Dashboard for Seed Oil Sleuth QA Pipeline
Rich interactive dashboard for viewing evaluation results and prompt comparisons
"""

import os
import sys
from datetime import datetime
from typing import List, Dict

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.columns import Columns
    from rich.progress import BarColumn, Progress
    from rich.layout import Layout
    from rich.align import Align
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False

from scoring_database import ScoringDatabase

class ScoringDashboard:
    def __init__(self):
        self.console = Console() if INTERACTIVE_AVAILABLE else None
        self.db = ScoringDatabase()
    
    def show_banner(self):
        """Show dashboard banner"""
        if not self.console:
            return
        
        title = Text("📊 Scoring Dashboard", style="bold cyan")
        subtitle = Text("LLM Evaluation Analytics & Prompt Comparison", style="dim")
        
        banner = Panel(
            Text.assemble(title, "\n", subtitle),
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(banner)
    
    def show_database_overview(self):
        """Show database statistics"""
        if not self.console:
            return
        
        stats = self.db.get_database_stats()
        
        table = Table(title="🗄️ Database Overview", show_header=False, box=None)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green", width=15)
        
        table.add_row("Total Evaluations", str(stats["total_evaluations"]))
        table.add_row("Prompt Versions", str(stats["total_prompt_versions"]))
        table.add_row("Available Prompt Files", str(stats["available_prompt_files"]))
        table.add_row("Database Size", f"{stats['database_size_mb']} MB")
        
        self.console.print(table)
    
    def show_latest_runs(self, limit=5):
        """Show recent evaluation runs"""
        if not self.console:
            return
        
        runs = self.db.get_latest_runs(limit)
        
        if not runs:
            self.console.print("📭 No evaluation runs found")
            return
        
        table = Table(title=f"🕒 Latest {limit} Runs", show_header=True, header_style="bold magenta")
        table.add_column("Run ID", style="cyan", width=18)
        table.add_column("Student Model", style="green", width=25)
        table.add_column("Judge", style="blue", width=12)
        table.add_column("Prompt", style="yellow", width=8)
        table.add_column("Score", justify="right", style="bright_red", width=10)
        table.add_column("Date", style="dim", width=14)
        
        for run in runs:
            run_id = run.get("run_id", "unknown")[:18]
            
            # Extract student model name (remove provider prefix for cleaner display)
            student_model = run.get("student_model", "unknown")
            if ":" in student_model:
                student_model = student_model.split(":", 1)[1]
            if len(student_model) > 24:
                student_model = student_model[:21] + "..."
            
            # Extract judge model (shorter display)
            judge_model = run.get("judge_model", "unknown")
            if ":" in judge_model:
                judge_model = judge_model.split(":", 1)[1].replace("-", "")
            if len(judge_model) > 11:
                judge_model = judge_model[:8] + "..."
            
            prompt_version = run.get("prompt_version", "?")
            score = f"{run.get('overall_score', 0):.1f}/100"
            percentage = run.get('percentage_score', 0)
            
            # Color code the score based on performance
            if percentage >= 90:
                score_style = "bright_green"
            elif percentage >= 80:
                score_style = "green"  
            elif percentage >= 70:
                score_style = "yellow"
            elif percentage >= 60:
                score_style = "orange1"
            else:
                score_style = "red"
            
            date = run.get("timestamp", "")[:16].replace("T", " ")[-14:]  # Just time, shorter
            
            table.add_row(
                run_id, 
                student_model, 
                judge_model,
                prompt_version, 
                f"[{score_style}]{score}[/{score_style}]",
                date
            )
        
        self.console.print()
        self.console.print(table)
    
    def show_prompt_performance(self):
        """Show performance by prompt version"""
        if not self.console:
            return
        
        trends = self.db.get_performance_trends()
        
        if "error" in trends:
            self.console.print(f"❌ {trends['error']}")
            return
        
        version_performance = trends["version_performance"]
        
        table = Table(title="📈 Prompt Performance Summary", show_header=True, header_style="bold magenta")
        table.add_column("Version", style="cyan", width=10)
        table.add_column("Runs", justify="right", style="blue", width=6)
        table.add_column("Student Model", style="yellow", width=20)
        table.add_column("Judge Model", style="magenta", width=15)
        table.add_column("Avg Score", justify="right", style="green", width=10)
        table.add_column("Best", justify="right", style="bright_green", width=8)
        table.add_column("Worst", justify="right", style="red", width=8)
        table.add_column("Performance", width=20)
        
        for version, stats in version_performance.items():
            avg_score = stats["average_score"]
            best_score = stats["best_score"]
            worst_score = stats["worst_score"]
            run_count = stats["run_count"]
            
            # Create performance bar
            percentage = (avg_score / 100) * 100
            bar_length = int(percentage / 5)  # Scale to 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Get model info from latest run for this version
            student_model = stats.get("student_model", "Unknown")
            judge_model = stats.get("judge_model", "Unknown")
            
            # Shorten model names for display
            if ":" in student_model:
                student_short = student_model.split(":")[-1][:18]
            else:
                student_short = student_model[:18]
                
            if ":" in judge_model:
                judge_short = judge_model.split(":")[-1][:13]
            else:
                judge_short = judge_model[:13]
            
            table.add_row(
                version,
                str(run_count),
                student_short,
                judge_short,
                f"{avg_score:.1f}/100",
                f"{best_score:.1f}",
                f"{worst_score:.1f}",
                f"[{bar}] {percentage:.1f}%"
            )
        
        self.console.print()
        self.console.print(table)
    
    def show_dimension_breakdown(self, prompt_version=None):
        """Show detailed dimension breakdown"""
        if not self.console:
            return
        
        if prompt_version:
            runs = self.db.get_prompt_performance_history(prompt_version)
            title = f"📊 Dimension Breakdown - {prompt_version}"
        else:
            runs = self.db.get_latest_runs(1)
            title = "📊 Latest Run Dimension Breakdown"
        
        if not runs:
            self.console.print("❌ No data available")
            return
        
        # Use latest run for dimension breakdown
        latest_run = runs[0] if runs else None
        
        if not latest_run or not latest_run.get("dimension_scores"):
            self.console.print("❌ No dimension scores available")
            return
        
        dimensions = latest_run["dimension_scores"]
        
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Dimension", style="cyan", width=20)
        table.add_column("Score", justify="right", style="green", width=8)
        table.add_column("Visual", width=25)
        table.add_column("Grade", style="yellow", width=6)
        
        for dim_name, score in dimensions.items():
            display_name = dim_name.replace('_', ' ').title()
            
            # Create visual bar
            bar_length = int(score)
            bar = "█" * bar_length + "░" * (10 - bar_length)
            
            # Assign grade
            if score >= 9.0:
                grade = "A+"
            elif score >= 8.5:
                grade = "A"
            elif score >= 8.0:
                grade = "A-"
            elif score >= 7.5:
                grade = "B+"
            elif score >= 7.0:
                grade = "B"
            elif score >= 6.5:
                grade = "B-"
            elif score >= 6.0:
                grade = "C+"
            else:
                grade = "C"
            
            table.add_row(
                display_name,
                f"{score:.1f}/10",
                f"[{bar}] {score*10:.0f}%",
                grade
            )
        
        self.console.print()
        self.console.print(table)
    
    def compare_prompts_interactive(self):
        """Interactive prompt comparison"""
        if not INTERACTIVE_AVAILABLE:
            print("❌ Interactive mode requires 'questionary' and 'rich' packages.")
            return
        
        available_prompts = self.db.get_available_prompts()
        
        if len(available_prompts) < 2:
            self.console.print("❌ Need at least 2 prompt versions to compare")
            return
        
        # Get versions that have evaluation data
        trends = self.db.get_performance_trends()
        if "error" in trends:
            self.console.print("❌ No evaluation data available")
            return
        
        available_versions = list(trends["version_performance"].keys())
        
        if len(available_versions) < 2:
            self.console.print("❌ Need at least 2 evaluated prompt versions to compare")
            return
        
        self.console.print()
        version1 = questionary.select(
            "Select first prompt version:",
            choices=available_versions,
            instruction="(Use arrow keys)"
        ).ask()
        
        remaining_versions = [v for v in available_versions if v != version1]
        version2 = questionary.select(
            "Select second prompt version:",
            choices=remaining_versions,
            instruction="(Use arrow keys)"
        ).ask()
        
        if not version1 or not version2:
            return
        
        self.show_prompt_comparison(version1, version2)
    
    def show_prompt_comparison(self, version1, version2):
        """Show detailed comparison between two prompts"""
        if not self.console:
            return
        
        comparison = self.db.compare_prompt_versions(version1, version2)
        
        if "error" in comparison:
            self.console.print(f"❌ {comparison['error']}")
            return
        
        v1_perf = comparison["v1_performance"]
        v2_perf = comparison["v2_performance"]
        
        # Create comparison table
        table = Table(title=f"⚖️ Prompt Comparison: {version1} vs {version2}", 
                     show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column(f"{version1}", style="blue", width=15)
        table.add_column(f"{version2}", style="green", width=15)
        table.add_column("Improvement", style="yellow", width=15)
        
        # Overall scores
        table.add_row(
            "Overall Score",
            f"{v1_perf['avg_overall_score']:.1f}/100",
            f"{v2_perf['avg_overall_score']:.1f}/100",
            f"{comparison['score_improvement']:+.1f}"
        )
        
        table.add_row(
            "Percentage Score",
            f"{v1_perf['avg_percentage_score']:.1f}%",
            f"{v2_perf['avg_percentage_score']:.1f}%",
            f"{comparison['percentage_improvement']:+.1f}%"
        )
        
        table.add_row(
            "Number of Runs",
            str(v1_perf['runs_count']),
            str(v2_perf['runs_count']),
            "-"
        )
        
        # Add separator
        table.add_row("", "", "", "")
        
        # Dimension comparisons
        v1_dims = v1_perf.get("avg_dimension_scores", {})
        v2_dims = v2_perf.get("avg_dimension_scores", {})
        
        for dim in v1_dims.keys():
            if dim in v2_dims:
                improvement = v2_dims[dim] - v1_dims[dim]
                # Get max points for this dimension from weights
                max_points = dimension_weights.get(dim, 10)
                table.add_row(
                    dim.replace('_', ' ').title(),
                    f"{v1_dims[dim]:.1f}/{max_points}",
                    f"{v2_dims[dim]:.1f}/{max_points}",
                    f"{improvement:+.1f}"
                )
        
        self.console.print()
        self.console.print(table)
        
        # Show overall verdict
        if comparison["is_improvement"]:
            verdict = f"✅ {version2} performs better by {comparison['score_improvement']:.1f} points"
            style = "green"
        else:
            verdict = f"❌ {version1} performs better by {abs(comparison['score_improvement']):.1f} points"
            style = "red"
        
        verdict_panel = Panel(verdict, border_style=style)
        self.console.print()
        self.console.print(verdict_panel)
    
    def show_model_comparison_matrix(self):
        """Show comprehensive model vs model performance comparison"""
        if not self.console:
            return
        
        runs = self.db.get_latest_runs(limit=100)  # Get more data for comparison
        
        if not runs:
            self.console.print("📭 No evaluation runs found")
            return
        
        # Group by student model and prompt version
        model_performance = {}
        
        for run in runs:
            student_model = run.get("student_model", "unknown")
            prompt_version = run.get("prompt_version", "unknown")
            overall_score = run.get("overall_score", 0)
            
            # Clean up model name for display
            if ":" in student_model:
                clean_model = student_model.split(":", 1)[1]
            else:
                clean_model = student_model
            
            key = f"{clean_model}@{prompt_version}"
            
            if key not in model_performance:
                model_performance[key] = {
                    "scores": [],
                    "model": clean_model,
                    "prompt": prompt_version
                }
            
            model_performance[key]["scores"].append(overall_score)
        
        # Calculate statistics
        model_stats = {}
        for key, data in model_performance.items():
            scores = data["scores"]
            if scores:
                model_stats[key] = {
                    "model": data["model"],
                    "prompt": data["prompt"],
                    "avg_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "run_count": len(scores)
                }
        
        # Create comparison table
        table = Table(title="🏆 Model Performance Comparison Matrix", show_header=True, header_style="bold magenta")
        table.add_column("Student Model", style="green", width=30)
        table.add_column("Prompt", style="yellow", width=8)
        table.add_column("Runs", justify="center", style="dim", width=5)
        table.add_column("Avg Score", justify="right", style="cyan", width=10)
        table.add_column("Best", justify="right", style="bright_green", width=8)
        table.add_column("Worst", justify="right", style="red", width=8)
        table.add_column("Performance", style="blue", width=25)
        
        # Sort by average score descending
        sorted_models = sorted(model_stats.values(), key=lambda x: x["avg_score"], reverse=True)
        
        for stats in sorted_models:
            model = stats["model"]
            if len(model) > 29:
                model = model[:26] + "..."
            
            avg_score = stats["avg_score"]
            percentage = (avg_score / 100) * 100
            
            # Create performance bar
            bar_length = int(percentage / 5)  # Scale to 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Color code based on performance
            if percentage >= 90:
                performance_style = "bright_green"
            elif percentage >= 80:
                performance_style = "green"
            elif percentage >= 70:
                performance_style = "yellow"
            elif percentage >= 60:
                performance_style = "orange1"
            else:
                performance_style = "red"
            
            table.add_row(
                model,
                stats["prompt"],
                str(stats["run_count"]),
                f"{avg_score:.1f}/100",
                f"{stats['max_score']:.1f}",
                f"{stats['min_score']:.1f}",
                f"[{performance_style}]{bar}[/{performance_style}] {percentage:.1f}%"
            )
        
        self.console.print()
        self.console.print(table)
        
        # Show top performers
        if sorted_models:
            self.console.print()
            top_performer = sorted_models[0]
            self.console.print(f"🥇 [bold green]Top Performer:[/bold green] {top_performer['model']} @ {top_performer['prompt']} - {top_performer['avg_score']:.1f}/100")
            
            if len(sorted_models) > 1:
                improvement_opportunity = sorted_models[-1]
                self.console.print(f"🎯 [bold yellow]Most Improvement Opportunity:[/bold yellow] {improvement_opportunity['model']} @ {improvement_opportunity['prompt']} - {improvement_opportunity['avg_score']:.1f}/100")
    
    def show_detailed_dimension_analysis(self):
        """Show detailed breakdown by scoring dimensions across models"""
        if not self.console:
            return
        
        runs = self.db.get_latest_runs(limit=50)
        
        if not runs:
            self.console.print("📭 No evaluation runs found")
            return
        
        # Dimension weights for reference
        dimension_weights = {
            "json_compliance": 10,
            "seed_oil_detection": 20,
            "conversation_flow": 15,
            "mathematical_accuracy": 20,
            "report_triggering_logic": 25,
            "user_engagement": 10
        }
        
        # Group by model
        model_dimensions = {}
        
        for run in runs:
            student_model = run.get("student_model", "unknown")
            if ":" in student_model:
                clean_model = student_model.split(":", 1)[1]
            else:
                clean_model = student_model
            
            if clean_model not in model_dimensions:
                model_dimensions[clean_model] = {dim: [] for dim in dimension_weights.keys()}
            
            # Extract dimension scores
            dimension_scores = run.get("dimension_scores", {})
            for dim in dimension_weights.keys():
                score = dimension_scores.get(dim, 0)
                model_dimensions[clean_model][dim].append(score)
        
        # Create dimension analysis table - REPORT BEHAVIOR ONLY
        table = Table(title="📊 Report Behavior Analysis by Model", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="green", width=25)
        table.add_column("CorrectTiming\n(60pts)", justify="center", style="cyan", width=12)
        table.add_column("MissingReports\n(20pts)", justify="center", style="yellow", width=14)
        table.add_column("PrematureRep\n(10pts)", justify="center", style="blue", width=12)
        table.add_column("IncompleteRep\n(10pts)", justify="center", style="red", width=12)
        table.add_column("Weakest Behavior", style="dim", width=20)
        
        for model, dimensions in model_dimensions.items():
            model_display = model[:24] + "..." if len(model) > 24 else model
            
            dimension_avgs = {}
            weakest_dim = ""
            weakest_score = 100
            
            row_data = [model_display]
            
            for dim, max_points in dimension_weights.items():
                scores = dimensions[dim]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    percentage = (avg_score / max_points) * 100
                    dimension_avgs[dim] = avg_score
                    
                    # Track weakest dimension
                    if percentage < weakest_score:
                        weakest_score = percentage
                        weakest_dim = dim.replace("_", " ").title()
                    
                    # Color code based on percentage
                    if percentage >= 90:
                        style = "bright_green"
                    elif percentage >= 80:
                        style = "green"
                    elif percentage >= 70:
                        style = "yellow"
                    elif percentage >= 60:
                        style = "orange1"
                    else:
                        style = "red"
                    
                    row_data.append(f"[{style}]{avg_score:.1f}[/{style}]")
                else:
                    row_data.append("N/A")
            
            row_data.append(f"{weakest_dim} ({weakest_score:.1f}%)")
            table.add_row(*row_data)
        
        self.console.print()
        self.console.print(table)
    
    def interactive_dashboard(self):
        """Main interactive dashboard"""
        if not INTERACTIVE_AVAILABLE:
            print("❌ Interactive mode requires 'questionary' and 'rich' packages.")
            return False
        
        if not sys.stdin.isatty():
            print("❌ Interactive mode requires a terminal.")
            return False
        
        while True:
            self.show_banner()
            self.show_database_overview()
            
            choices = [
                questionary.Choice("📈 Latest Runs", "latest"),
                questionary.Choice("🏆 Model Comparison Matrix", "model_comparison"),
                questionary.Choice("📊 Detailed Dimension Analysis", "dimension_analysis"),
                questionary.Choice("🛢️ Oil Quality Analysis", "oil_quality"),
                questionary.Choice("🏆 Prompt Performance", "performance"),
                questionary.Choice("⚖️ Compare Prompts", "compare"),
                questionary.Choice("🔄 Refresh Data", "refresh"),
                questionary.Choice("💾 Export Data", "export"),
                questionary.Choice("❌ Exit", "exit")
            ]
            
            self.console.print()
            action = questionary.select(
                "Choose an action:",
                choices=choices,
                instruction="(Use arrow keys and Enter)"
            ).ask()
            
            if action == "exit" or action is None:
                self.console.print("👋 Goodbye!")
                break
            
            if action == "latest":
                limit = questionary.text(
                    "How many recent runs to show?",
                    default="10",
                    validate=lambda x: x.isdigit() and int(x) > 0
                ).ask()
                if limit:
                    self.show_latest_runs(int(limit))
            
            elif action == "model_comparison":
                self.show_model_comparison_matrix()
            
            elif action == "dimension_analysis":
                self.show_detailed_dimension_analysis()
            
            elif action == "oil_quality":
                self.show_oil_quality_analysis()
            
            elif action == "performance":
                self.show_prompt_performance()
            
            elif action == "dimensions":
                # Ask which prompt version
                trends = self.db.get_performance_trends()
                if "error" not in trends:
                    versions = list(trends["version_performance"].keys())
                    if versions:
                        prompt_choices = ["Latest Run"] + versions
                        choice = questionary.select(
                            "Select prompt version:",
                            choices=prompt_choices
                        ).ask()
                        
                        if choice and choice != "Latest Run":
                            self.show_dimension_breakdown(choice)
                        else:
                            self.show_dimension_breakdown()
            
            elif action == "compare":
                self.compare_prompts_interactive()
            
            elif action == "refresh":
                self.console.print("🔄 Data refreshed!")
            
            elif action == "export":
                export_file = self.db.export_data()
                self.console.print(f"💾 Data exported to: {export_file}")
            
            if action != "exit":
                input("\n📱 Press Enter to continue...")
                self.console.clear()
        
        return True

if __name__ == "__main__":
    dashboard = ScoringDashboard()
    dashboard.interactive_dashboard()