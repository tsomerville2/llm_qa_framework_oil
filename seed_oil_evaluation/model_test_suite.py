#!/usr/bin/env python3
"""
Model Test Suite for Seed Oil Sleuth QA Pipeline
Automated testing across multiple student models and prompt versions
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict
from student_evaluator import StudentEvaluator  
from judge_evaluator import JudgeEvaluator
from scoring_database import ScoringDatabase

try:
    import questionary
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False

class ModelTestSuite:
    def __init__(self):
        self.console = Console() if INTERACTIVE_AVAILABLE else None
        self.db = ScoringDatabase()
        self.base_dir = "seed_oil_evaluation"
        
    def get_available_models(self, model_type: str) -> List[str]:
        """Get available models from choice files"""
        choice_file = f"{self.base_dir}/llm_{model_type}_choices.txt"
        
        if not os.path.exists(choice_file):
            return []
        
        models = []
        with open(choice_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    models.append(line)
        
        return models
    
    def get_available_prompts(self) -> List[str]:
        """Get available prompt versions"""
        prompts = self.db.get_available_prompts()
        return [p["version"] for p in prompts]
    
    def run_model_evaluation(self, student_model: str, prompt_version: str, judge_model: str = None) -> Dict:
        """Run evaluation for a specific model and prompt combination"""
        
        # Temporarily update student model choice file
        original_choices = self.get_available_models("student")
        self.set_student_model(student_model)
        
        try:
            # Run student evaluation
            evaluator = StudentEvaluator(prompt_version=prompt_version, interactive=False)
            student_results = evaluator.evaluate_all_conversations()
            successful_student = len([r for r in student_results if r["evaluation_success"]])
            
            if successful_student == 0:
                return {"success": False, "error": "No successful student evaluations"}
            
            # Run judge evaluation  
            judge = JudgeEvaluator(interactive=False)
            judge_results = judge.judge_all_responses()
            successful_judge = len([r for r in judge_results if r["judgment_success"]])
            
            if successful_judge == 0:
                return {"success": False, "error": "No successful judge evaluations"}
            
            # Calculate aggregate score
            overall_scores = [r["judge_scores"]["overall_score"] for r in judge_results if r["judgment_success"]]
            avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
            
            return {
                "success": True,
                "student_model": student_model,
                "prompt_version": prompt_version,
                "judge_model": judge.judge_model_name,
                "avg_score": avg_score,
                "successful_evaluations": successful_judge,
                "total_conversations": len(student_results)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        finally:
            # Restore original choices
            self.restore_student_models(original_choices)
    
    def set_student_model(self, model: str):
        """Temporarily set a single student model"""
        choice_file = f"{self.base_dir}/llm_student_choices.txt"
        
        with open(choice_file, 'w') as f:
            f.write("# Temporary single model for testing\n")
            f.write(f"{model}\n")
    
    def restore_student_models(self, models: List[str]):
        """Restore original student model choices"""
        choice_file = f"{self.base_dir}/llm_student_choices.txt"
        
        with open(choice_file, 'w') as f:
            f.write("# Student LLM Choices\n")
            f.write("# Format: provider:model_name\n\n")
            for model in models:
                if not model.startswith('#'):
                    f.write(f"{model}\n")
    
    def run_comprehensive_test(self, student_models: List[str] = None, prompt_versions: List[str] = None) -> Dict:
        """Run comprehensive evaluation across multiple models and prompts"""
        
        if student_models is None:
            student_models = self.get_available_models("student")
        
        if prompt_versions is None:
            prompt_versions = self.get_available_prompts()
        
        if not student_models:
            return {"error": "No student models available"}
        
        if not prompt_versions:
            return {"error": "No prompt versions available"}
        
        results = {}
        total_tests = len(student_models) * len(prompt_versions)
        
        if self.console:
            self.console.print(f"🚀 Starting comprehensive test: {len(student_models)} models × {len(prompt_versions)} prompts = {total_tests} tests")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Running evaluations...", total=total_tests)
            
            for student_model in student_models:
                for prompt_version in prompt_versions:
                    test_key = f"{student_model}@{prompt_version}"
                    
                    # Update progress
                    progress.update(task, description=f"Testing {student_model.split(':')[-1][:20]}... @ {prompt_version}")
                    
                    # Run evaluation
                    result = self.run_model_evaluation(student_model, prompt_version)
                    results[test_key] = result
                    
                    # Add small delay to avoid API rate limits
                    time.sleep(1)
                    
                    progress.advance(task)
        
        return {
            "success": True,
            "total_tests": total_tests,
            "results": results,
            "completed_at": datetime.now().isoformat()
        }
    
    def show_test_results(self, test_results: Dict):
        """Display comprehensive test results"""
        if not self.console:
            return
        
        if "error" in test_results:
            self.console.print(f"❌ {test_results['error']}")
            return
        
        results = test_results["results"]
        successful_tests = [r for r in results.values() if r.get("success", False)]
        
        # Create results table
        table = Table(title="🏆 Comprehensive Model Test Results", show_header=True, header_style="bold magenta")
        table.add_column("Student Model", style="green", width=30)
        table.add_column("Prompt", style="yellow", width=8)
        table.add_column("Score", justify="right", style="cyan", width=10)
        table.add_column("Success", justify="center", style="blue", width=8)
        table.add_column("Judge", style="dim", width=12)
        
        # Sort by score descending
        sorted_results = sorted(successful_tests, key=lambda x: x.get("avg_score", 0), reverse=True)
        
        for result in sorted_results:
            model = result["student_model"]
            if ":" in model:
                model = model.split(":", 1)[1]
            if len(model) > 29:
                model = model[:26] + "..."
            
            score = result["avg_score"]
            percentage = (score / 100) * 100
            
            # Color code score
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
            
            judge = result.get("judge_model", "unknown")
            if ":" in judge:
                judge = judge.split(":", 1)[1][:11]
            
            table.add_row(
                model,
                result["prompt_version"],
                f"[{score_style}]{score:.1f}/100[/{score_style}]",
                f"✅ {result['successful_evaluations']}/{result['total_conversations']}",
                judge
            )
        
        self.console.print()
        self.console.print(table)
        
        # Show summary statistics
        if successful_tests:
            avg_scores = [r["avg_score"] for r in successful_tests]
            best_result = max(successful_tests, key=lambda x: x["avg_score"])
            worst_result = min(successful_tests, key=lambda x: x["avg_score"])
            
            summary = f"""
🎯 Test Summary:
  • Total Tests: {test_results['total_tests']}
  • Successful: {len(successful_tests)}
  • Average Score: {sum(avg_scores) / len(avg_scores):.1f}/100
  
🥇 Best Performer: {best_result['student_model'].split(':')[-1]} @ {best_result['prompt_version']} - {best_result['avg_score']:.1f}/100
🎯 Needs Improvement: {worst_result['student_model'].split(':')[-1]} @ {worst_result['prompt_version']} - {worst_result['avg_score']:.1f}/100
            """
            
            panel = Panel(summary.strip(), title="📊 Test Summary", border_style="cyan")
            self.console.print()
            self.console.print(panel)
    
    def interactive_test_suite(self):
        """Interactive test suite interface"""
        if not INTERACTIVE_AVAILABLE:
            print("❌ Interactive mode requires 'questionary' and 'rich' packages.")
            return False
        
        while True:
            self.console.print()
            title = Panel("🧪 Model Test Suite", style="bold cyan")
            self.console.print(title)
            
            choices = [
                questionary.Choice("🚀 Run Comprehensive Test (All Models × All Prompts)", "comprehensive"),
                questionary.Choice("🎯 Test Specific Model", "specific"),
                questionary.Choice("📊 View Previous Results", "results"),
                questionary.Choice("⚙️ Manage Model Choices", "manage"),
                questionary.Choice("❌ Exit", "exit")
            ]
            
            action = questionary.select(
                "Choose an action:",
                choices=choices,
                instruction="(Use arrow keys and Enter)"
            ).ask()
            
            if action == "exit" or action is None:
                break
            
            elif action == "comprehensive":
                result = self.run_comprehensive_test()
                self.show_test_results(result)
            
            elif action == "specific":
                # Let user select specific model and prompt
                student_models = self.get_available_models("student")
                prompt_versions = self.get_available_prompts()
                
                if not student_models or not prompt_versions:
                    self.console.print("❌ No models or prompts available")
                    continue
                
                model = questionary.select("Select student model:", choices=student_models).ask()
                prompt = questionary.select("Select prompt version:", choices=prompt_versions).ask()
                
                if model and prompt:
                    result = self.run_model_evaluation(model, prompt)
                    if result.get("success"):
                        self.console.print(f"✅ Test completed: {result['avg_score']:.1f}/100")
                    else:
                        self.console.print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            
            elif action == "results":
                from scoring_dashboard import ScoringDashboard
                dashboard = ScoringDashboard()
                dashboard.show_model_comparison_matrix()
            
            elif action == "manage":
                self.console.print("📁 Model choice files are located in:")
                self.console.print(f"  • {self.base_dir}/llm_student_choices.txt")
                self.console.print(f"  • {self.base_dir}/llm_judge_choices.txt") 
                self.console.print(f"  • {self.base_dir}/llm_convo_choices.txt")

if __name__ == "__main__":
    suite = ModelTestSuite()
    suite.interactive_test_suite()