#!/usr/bin/env python3
"""
Seed Oil Sleuth QA Pipeline Orchestrator
Run the complete evaluation pipeline or individual components
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Import colorama for colorful output
try:
    from colorama import init, Fore, Back, Style
    init()  # Initialize colorama
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama not available
    class MockColorama:
        def __getattr__(self, name):
            return ""
    Fore = Back = Style = MockColorama()
    COLORS_AVAILABLE = False

# Import our existing modules
from conversation_generator import ConversationGenerator
from student_evaluator import StudentEvaluator
from judge_evaluator import JudgeEvaluator
from model_selector import get_available_models, parse_model_file

# Import interactive menu libraries
try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False

class QAPipelineOrchestrator:
    def __init__(self):
        self.base_dir = "seed_oil_evaluation"
        self.console = Console() if INTERACTIVE_AVAILABLE else None
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs = [
            f"{self.base_dir}/conversations",
            f"{self.base_dir}/student_responses", 
            f"{self.base_dir}/judge_scores"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def check_api_keys(self):
        """Check that required API keys are set"""
        missing_keys = []
        if not os.getenv('OPENAI_API_KEY'):
            missing_keys.append('OPENAI_API_KEY')
        if not os.getenv('GROQ_API_KEY'):
            missing_keys.append('GROQ_API_KEY')
        
        if missing_keys:
            print(f"❌ Missing API keys: {', '.join(missing_keys)}")
            print("Please set these environment variables before running.")
            return False
        return True
    
    def run_full_pipeline(self, num_conversations=10):
        """Run the complete QA pipeline from scratch"""
        print("🚀 Starting Full QA Pipeline")
        print("=" * 50)
        
        if not self.check_api_keys():
            return False
        
        # Phase 1: Generate Conversations
        print("\n📝 Phase 1: Generating Conversations")
        generator = ConversationGenerator(interactive=False)
        conversations = generator.generate_all_conversations()
        if not conversations:
            print("❌ Failed to generate conversations")
            return False
        print(f"✅ Generated {len(conversations)} conversations")
        
        # Phase 2: Student Evaluation
        print("\n🤖 Phase 2: Student Evaluation")
        evaluator = StudentEvaluator(interactive=False)
        student_results = evaluator.evaluate_all_conversations()
        successful_student = len([r for r in student_results if r["evaluation_success"]])
        print(f"✅ Evaluated {successful_student}/{len(student_results)} conversations")
        
        if successful_student == 0:
            print("❌ No successful student evaluations")
            return False
        
        # Phase 3: Judge Evaluation
        print("\n⚖️ Phase 3: Judge Evaluation")
        judge = JudgeEvaluator(interactive=False)
        judge_results = judge.judge_all_responses()
        successful_judge = len([r for r in judge_results if r["judgment_success"]])
        print(f"✅ Judged {successful_judge}/{len(judge_results)} responses")
        
        # Show Results
        self.show_results_summary()
        
        return True
    
    def evaluate_with_pause(self, prompt_version="v1.0", student_model=None, judge_model=None):
        """Run evaluation with pause mode enabled for reviewing oil reports"""
        print("⏸️ Running Evaluation with Pause Mode")
        print("=" * 40)
        
        if not self.check_api_keys():
            return False
        
        # Check if conversations exist
        index_path = f"{self.base_dir}/conversations/index.json"
        if not os.path.exists(index_path):
            print("❌ No conversations found. Run full pipeline first.")
            return False
        
        # Student Evaluation
        print(f"\n🤖 Student Evaluation (Prompt: {prompt_version})")
        evaluator = StudentEvaluator(prompt_version=prompt_version, interactive=False, selected_model=student_model)
        student_results = evaluator.evaluate_all_conversations()
        successful_student = len([r for r in student_results if r["evaluation_success"]])
        print(f"✅ Evaluated {successful_student}/{len(student_results)} conversations")
        
        if successful_student == 0:
            print("❌ No successful student evaluations")
            return False
        
        # Judge Evaluation with Pause Mode
        print("\n⚖️ Judge Evaluation with Pause Mode")
        print("📋 After each report judgment, you'll see the oil report for review")
        print("🔄 Press Enter to continue to the next judgment")
        judge = JudgeEvaluator(interactive=False, selected_model=judge_model, pause_mode=True)
        judge_results = judge.judge_all_responses()
        successful_judge = len([r for r in judge_results if r["judgment_success"]])
        print(f"✅ Judged {successful_judge}/{len(judge_results)} responses")
        
        # Show Results
        self.show_results_summary()
        
        return True
    
    def add_conversations(self, num_new=5):
        """Add more conversations to existing dataset"""
        print(f"➕ Adding {num_new} New Conversations")
        print("=" * 40)
        
        if not self.check_api_keys():
            return False
        
        # Check if conversations already exist
        index_path = f"{self.base_dir}/conversations/index.json"
        if not os.path.exists(index_path):
            print("❌ No existing conversations found. Run full pipeline first.")
            return False
        
        # Load existing index
        with open(index_path, 'r') as f:
            existing_index = json.load(f)
        
        existing_count = len(existing_index["conversations"])
        print(f"📊 Found {existing_count} existing conversations")
        
        # Generate new conversations
        generator = ConversationGenerator()
        # Modify generator to append instead of overwrite
        new_conversations = generator.generate_additional_conversations(num_new, existing_count)
        
        if new_conversations:
            print(f"✅ Added {len(new_conversations)} new conversations")
            print("💡 Run 'python run_qa_pipeline.py --evaluate-only' to process them")
            return True
        else:
            print("❌ Failed to generate new conversations")
            return False
    
    def evaluate_only(self, prompt_version="v1.0", student_model=None, judge_model=None):
        """Run only student and judge evaluation on existing conversations"""
        print("🔄 Running Evaluation Only")
        print("=" * 30)
        
        if not self.check_api_keys():
            return False
        
        # Check if conversations exist
        index_path = f"{self.base_dir}/conversations/index.json"
        if not os.path.exists(index_path):
            print("❌ No conversations found. Run full pipeline first.")
            return False
        
        # Student Evaluation
        print(f"\n🤖 Student Evaluation (Prompt: {prompt_version})")
        evaluator = StudentEvaluator(prompt_version=prompt_version, interactive=False, selected_model=student_model)
        student_results = evaluator.evaluate_all_conversations()
        successful_student = len([r for r in student_results if r["evaluation_success"]])
        print(f"✅ Evaluated {successful_student}/{len(student_results)} conversations")
        
        if successful_student == 0:
            print("❌ No successful student evaluations")
            return False
        
        # Judge Evaluation
        print("\n⚖️ Judge Evaluation")
        judge = JudgeEvaluator(interactive=False, selected_model=judge_model)
        judge_results = judge.judge_all_responses()
        successful_judge = len([r for r in judge_results if r["judgment_success"]])
        print(f"✅ Judged {successful_judge}/{len(judge_results)} responses")
        
        # Show Results
        self.show_results_summary()
        
        return True
    
    def show_results_summary(self):
        """Display a summary of the latest results"""
        summary_path = f"{self.base_dir}/judge_scores/judgment_summary.json"
        
        if not os.path.exists(summary_path):
            print("⚠️ No judgment results found")
            return
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        stats = summary["aggregate_statistics"]
        
        print("\n" + "=" * 50)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        
        # Add colorful score display
        score = stats['average_overall_score']
        if score >= 80:
            score_color = Fore.GREEN if COLORS_AVAILABLE else ""
            score_icon = "🏆"
        elif score >= 60:
            score_color = Fore.YELLOW if COLORS_AVAILABLE else ""
            score_icon = "⭐"
        else:
            score_color = Fore.RED if COLORS_AVAILABLE else ""
            score_icon = "📉"
        
        reset_color = Style.RESET_ALL if COLORS_AVAILABLE else ""
        print(f"🎯 Overall Score: {score_color}{score_icon} {score}/100 ({stats['percentage_score']}%) {score_icon}{reset_color}")
        print(f"📈 Total Responses: {summary['successful_judgments']}/{summary['total_responses']}")
        print(f"🤖 Student Model: {summary['student_model']}")
        print(f"⚖️ Judge Model: {summary['judge_model']}")
        
        print(f"\n📈 Report Behavior Breakdown:")
        # Load weights from file
        dimension_weights = {}
        weights_file = "seed_oil_evaluation/scoring_weights.txt"
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line and not line.startswith('#'):
                        dim, weight = line.split(':', 1)
                        dimension_weights[dim.strip()] = int(weight.strip())
        else:
            # If no weights file exists, use defaults but this should not happen in normal operation
            dimension_weights = {
                "correct_report_timing": 60,
                "missing_reports": 20,
                "premature_reports": 10,
                "oil_report_quality": 10
            }
        
        for dim, score in stats['dimension_averages'].items():
            max_points = dimension_weights.get(dim, 10)
            percentage = (score / max_points) * 100 if max_points > 0 else 0
            bar_length = int(percentage / 10)
            bar = "█" * bar_length + "░" * (10 - bar_length)
            print(f"  {dim.replace('_', ' ').title():<25}: {score:4.1f}/{max_points} ({percentage:4.1f}%) [{bar}]")
        
        # Show report analysis statistics if available
        if 'report_analysis_stats' in stats:
            report_stats = stats['report_analysis_stats']
            print(f"\n🎯 Report Triggering Analysis:")
            print(f"  ✅ Correct Report Timing: {report_stats['correct_report_triggering_rate']}%")
            print(f"  ⚠️ Premature Reports: {report_stats['premature_report_rate']}%")  
            print(f"  ❌ Missing Reports: {report_stats['missing_report_rate']}%")
            print(f"  📝 Oil Report Quality Issues: {report_stats['oil_report_quality_rate']}%")
        
        print(f"\n💾 Results saved to: {self.base_dir}/")
        print(f"📅 Last updated: {summary['judged_at'][:19]}")
    
    def clean_results(self):
        """Clean all generated results (keep conversations)"""
        print("🧹 Cleaning Results")
        
        import shutil
        
        # Remove student responses
        responses_dir = f"{self.base_dir}/student_responses"
        if os.path.exists(responses_dir):
            shutil.rmtree(responses_dir)
            os.makedirs(responses_dir)
            print("✅ Cleaned student responses")
        
        # Remove judge scores  
        scores_dir = f"{self.base_dir}/judge_scores"
        if os.path.exists(scores_dir):
            shutil.rmtree(scores_dir)
            os.makedirs(scores_dir)
            print("✅ Cleaned judge scores")
        
        print("💡 Conversations preserved. Run --evaluate-only to re-process.")
    
    def reset_everything(self):
        """Reset everything - conversations, evaluations, and database with STARTOVER safeguard"""
        self.console.print("🚨 [bold red]NUCLEAR OPTION: This will permanently delete EVERYTHING![/bold red]")
        self.console.print("[red]This will delete:[/red]")
        self.console.print("  • All generated conversations")
        self.console.print("  • All student responses and evaluations")
        self.console.print("  • All judge scores and analysis")
        self.console.print("  • Complete scoring database and history")
        self.console.print("  • All analytics and tracking data")
        self.console.print("[bold red]EVERYTHING WILL BE PERMANENTLY LOST![/bold red]")
        
        confirm = questionary.confirm(
            "Are you absolutely sure you want to delete EVERYTHING?",
            default=False
        ).ask()
        
        if not confirm:
            self.console.print("❌ Reset Everything cancelled.")
            return True
        
        # Require STARTOVER confirmation
        confirmation_text = questionary.text(
            "Type 'STARTOVER' (exactly) to confirm complete reset:",
            validate=lambda x: len(x.strip()) > 0
        ).ask()
        
        if not confirmation_text or confirmation_text.strip() != "STARTOVER":
            self.console.print("❌ [bold red]Reset Everything cancelled.[/bold red]")
            self.console.print("You must type 'STARTOVER' exactly to confirm.")
            return True
        
        try:
            import shutil
            
            # Remove all directories
            dirs_to_remove = [
                f"{self.base_dir}/conversations",
                f"{self.base_dir}/student_responses", 
                f"{self.base_dir}/judge_scores"
            ]
            
            self.console.print("\n🔥 [bold red]DESTROYING EVERYTHING...[/bold red]")
            
            for dir_path in dirs_to_remove:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    self.console.print(f"✅ Deleted: {dir_path}")
            
            # Reset database if it exists
            try:
                from scoring_database import ScoringDatabase
                db = ScoringDatabase()
                if db.reset_all_data("RESETALLDATA"):
                    self.console.print("✅ Database reset successfully")
            except Exception as e:
                self.console.print(f"⚠️ Database reset failed: {e}")
            
            # Recreate empty directories
            self.ensure_directories()
            self.console.print("✅ Recreated empty directories")
            
            self.console.print("\n💥 [bold green]COMPLETE RESET SUCCESSFUL![/bold green]")
            self.console.print("Everything has been permanently deleted.")
            self.console.print("You can now start fresh with 'Full Pipeline'.")
            
            return True
            
        except Exception as e:
            self.console.print(f"\n❌ [bold red]Reset failed: {str(e)}[/bold red]")
            return False
    
    def show_welcome_banner(self):
        """Show a rich welcome banner"""
        if not self.console:
            return
        
        title = Text("🔬 Seed Oil Sleuth QA Pipeline", style="bold cyan")
        subtitle = Text("Automated LLM Evaluation System", style="dim")
        
        banner = Panel(
            Text.assemble(title, "\n", subtitle),
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(banner)
    
    def get_pipeline_status(self):
        """Get current status of the pipeline"""
        status = {
            "conversations": 0,
            "student_responses": 0,
            "judge_scores": 0,
            "latest_score": None
        }
        
        # Check conversations
        index_path = f"{self.base_dir}/conversations/index.json"
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
                status["conversations"] = index.get("total_conversations", 0)
        
        # Check student responses
        summary_path = f"{self.base_dir}/student_responses/evaluation_summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                status["student_responses"] = summary.get("successful_evaluations", 0)
        
        # Check judge scores
        judge_summary_path = f"{self.base_dir}/judge_scores/judgment_summary.json"
        if os.path.exists(judge_summary_path):
            with open(judge_summary_path, 'r') as f:
                summary = json.load(f)
                status["judge_scores"] = summary.get("successful_judgments", 0)
                if summary.get("aggregate_statistics"):
                    status["latest_score"] = f"{summary['aggregate_statistics']['average_overall_score']}/60 ({summary['aggregate_statistics']['percentage_score']}%)"
        
        return status
    
    def show_status_table(self):
        """Show current pipeline status in a rich table"""
        if not self.console:
            return
        
        status = self.get_pipeline_status()
        
        table = Table(title="📊 Pipeline Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Count", justify="right", style="green", width=10)
        table.add_column("Status", style="yellow")
        
        # Conversations row
        conv_status = "✅ Ready" if status["conversations"] > 0 else "❌ None"
        table.add_row("Conversations", str(status["conversations"]), conv_status)
        
        # Student responses row
        student_status = "✅ Evaluated" if status["student_responses"] > 0 else "⏳ Pending"
        table.add_row("Student Responses", str(status["student_responses"]), student_status)
        
        # Judge scores row
        judge_status = "✅ Scored" if status["judge_scores"] > 0 else "⏳ Pending"
        if status["latest_score"]:
            judge_status += f" ({status['latest_score']})"
        table.add_row("Judge Scores", str(status["judge_scores"]), judge_status)
        
        self.console.print()
        self.console.print(table)
    
    def interactive_menu(self):
        """Show interactive menu for pipeline operations"""
        if not INTERACTIVE_AVAILABLE:
            print("❌ Interactive mode requires 'questionary' and 'rich' packages.")
            print("Install with: pip install questionary rich")
            return False
        
        # Check if we're in a terminal that supports interactive input
        if not sys.stdin.isatty():
            print("❌ Interactive mode requires a terminal. Use command line options instead.")
            print("Usage: python run_qa_pipeline.py --help")
            return False
        
        self.show_welcome_banner()
        self.show_status_table()
        
        # Create menu options
        choices = [
            questionary.Choice("🔄 Evaluate Only (Re-run student & judge evaluation)", "evaluate"),
            questionary.Choice("⏸️ Evaluate with Pause Mode (Review each oil report)", "evaluate_pause"),
            questionary.Choice("🚀 Full Pipeline (Generate new conversations + evaluate)", "full"),  
            questionary.Choice("🌈 Rainbowblast (Test ALL models with colorful output)", "rainbowblast"),
            questionary.Choice("➕ Add Conversations (Add more test cases)", "add"),
            questionary.Choice("📊 Show Results (View latest scores)", "results"),
            questionary.Choice("📈 Scoring Dashboard (Advanced analytics)", "dashboard"),
            questionary.Choice("🧹 Clean Results (Keep conversations, clear evaluations)", "clean"),
            questionary.Choice("🗑️ Reset Database (Delete ALL scoring history)", "reset_db"),
            questionary.Choice("💥 Reset Everything (DELETE conversations, evaluations, database)", "reset_everything"),
            questionary.Choice("❌ Exit", "exit")
        ]
        
        self.console.print()
        
        try:
            action = questionary.select(
                "What would you like to do?",
                choices=choices,
                default="evaluate",
                use_shortcuts=True,
                instruction="(Use arrow keys and Enter)"
            ).ask()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n👋 Goodbye!")
            return True
        
        if action == "exit" or action is None:
            self.console.print("\n👋 Goodbye!")
            return True
        
        self.console.print()
        
        # Handle the selected action
        try:
            if action == "evaluate":
                # Ask which prompt version to use
                from scoring_database import ScoringDatabase
                db = ScoringDatabase()
                available_prompts = db.get_available_prompts()
                
                if available_prompts:
                    prompt_choices = [p["version"] for p in available_prompts]
                    prompt_version = questionary.select(
                        "Select prompt version:",
                        choices=prompt_choices,
                        default="v1.0" if "v1.0" in prompt_choices else prompt_choices[0]
                    ).ask()
                    
                    if prompt_version:
                        # Interactive model selection
                        try:
                            # Select models interactively
                            from model_selector import select_model
                            student_model = select_model('student', interactive=True)
                            judge_model = select_model('judge', interactive=True)
                            
                            if student_model and judge_model:
                                success = self.evaluate_only(prompt_version, student_model, judge_model)
                            else:
                                self.console.print("❌ Model selection cancelled")
                                success = False
                        except ValueError as e:
                            self.console.print(f"❌ Model selection failed: {e}")
                            success = False
                    else:
                        success = False
                else:
                    success = self.evaluate_only("v1.0")
            
            elif action == "evaluate_pause":
                # Evaluate with pause mode enabled
                from scoring_database import ScoringDatabase
                db = ScoringDatabase()
                available_prompts = db.get_available_prompts()
                
                if available_prompts:
                    prompt_choices = [p["version"] for p in available_prompts]
                    prompt_version = questionary.select(
                        "Select prompt version:",
                        choices=prompt_choices,
                        default="v1.0" if "v1.0" in prompt_choices else prompt_choices[0]
                    ).ask()
                    
                    if prompt_version:
                        # Interactive model selection
                        try:
                            # Select models interactively
                            from model_selector import select_model
                            student_model = select_model('student', interactive=True)
                            judge_model = select_model('judge', interactive=True)
                            
                            if student_model and judge_model:
                                success = self.evaluate_with_pause(prompt_version, student_model, judge_model)
                            else:
                                self.console.print("❌ Model selection cancelled")
                                success = False
                        except ValueError as e:
                            self.console.print(f"❌ Model selection failed: {e}")
                            success = False
                    else:
                        success = False
                else:
                    success = self.evaluate_with_pause("v1.0")
                    
            elif action == "full":
                success = self.run_full_pipeline()
            elif action == "rainbowblast":
                success = self.rainbowblast(interactive=True)
            elif action == "add":
                # Ask how many conversations to add
                num_conversations = questionary.text(
                    "How many conversations to add?",
                    default="5",
                    validate=lambda x: x.isdigit() and int(x) > 0
                ).ask()
                
                if num_conversations:
                    success = self.add_conversations(int(num_conversations))
                else:
                    success = False
            elif action == "results":
                self.show_results_summary()
                success = True
            elif action == "dashboard":
                from scoring_dashboard import ScoringDashboard
                dashboard = ScoringDashboard()
                dashboard.interactive_dashboard()
                success = True
            elif action == "clean":
                confirm = questionary.confirm(
                    "Are you sure you want to clean all evaluation results?",
                    default=False
                ).ask()
                
                if confirm:
                    self.clean_results()
                    success = True
                else:
                    self.console.print("❌ Clean operation cancelled.")
                    success = True
            elif action == "reset_db":
                self.console.print("🚨 [bold red]WARNING: This will permanently delete ALL scoring history![/bold red]")
                self.console.print("This includes all evaluation runs, prompt version tracking, and analytics data.")
                self.console.print("Conversations will be preserved, but all scoring data will be lost forever.")
                
                confirm = questionary.confirm(
                    "Are you absolutely sure you want to reset the database?",
                    default=False
                ).ask()
                
                if confirm:
                    confirmation_text = questionary.text(
                        "Type 'RESETALLDATA' (exactly) to confirm database reset:",
                        validate=lambda x: len(x.strip()) > 0
                    ).ask()
                    
                    if confirmation_text:
                        from scoring_database import ScoringDatabase
                        db = ScoringDatabase()
                        
                        if db.reset_all_data(confirmation_text.strip()):
                            self.console.print("✅ [bold green]Database reset successfully![/bold green]")
                            self.console.print("All scoring history has been permanently deleted.")
                            success = True
                        else:
                            self.console.print("❌ [bold red]Reset cancelled.[/bold red]")
                            self.console.print("You must type 'RESETALLDATA' exactly to confirm.")
                            success = True
                    else:
                        self.console.print("❌ Reset cancelled.")
                        success = True
                else:
                    self.console.print("❌ Database reset cancelled.")
                    success = True
            elif action == "reset_everything":
                success = self.reset_everything()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n👋 Operation cancelled!")
            return True
        
        if success:
            self.console.print("\n✅ Operation completed!")
            
            # Always continue to the next operation
            try:
                return self.interactive_menu()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n👋 Goodbye!")
                return True
        else:
            self.console.print("\n❌ Operation failed!")
            
            # Continue to menu even after failures
            try:
                return self.interactive_menu()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n👋 Goodbye!")
                return True

    def rainbowblast(self, judge_model=None, interactive=True):
        """Test ALL student models against ONE judge with colorful output and final scores"""
        print(f"\n{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{Back.WHITE} 🌈 RAINBOWBLAST MODE: Testing All Student Models! 🌈 {Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")
        
        try:
            # Get prompt version
            prompt_version = "v1.0"  # Default for CLI
            if interactive:
                # Ask for prompt version
                from scoring_database import ScoringDatabase
                db = ScoringDatabase()
                available_prompts = db.get_available_prompts()
                
                if available_prompts:
                    prompt_choices = [p["version"] for p in available_prompts]
                    try:
                        prompt_version = questionary.select(
                            "Select prompt version for testing:",
                            choices=prompt_choices,
                            default="v1.0" if "v1.0" in prompt_choices else prompt_choices[0]
                        ).ask()
                        
                        if not prompt_version:
                            print(f"{Fore.RED}❌ No prompt selected. Cancelled.{Style.RESET_ALL}")
                            return False
                    except (KeyboardInterrupt, EOFError):
                        print(f"\n{Fore.RED}❌ Cancelled by user{Style.RESET_ALL}")
                        return False
                else:
                    print(f"{Fore.YELLOW}⚠️ No prompts found, using default v1.0{Style.RESET_ALL}")
            
            # Get judge model
            if judge_model is None and interactive:
                # Interactive judge selection
                try:
                    from model_selector import select_model
                    judge_selection = select_model('judge', interactive=True)
                    if not judge_selection:
                        print(f"{Fore.RED}❌ No judge model selected. Cancelled.{Style.RESET_ALL}")
                        return False
                    judge_model = judge_selection
                except (KeyboardInterrupt, EOFError):
                    print(f"\n{Fore.RED}❌ Cancelled by user{Style.RESET_ALL}")
                    return False
            elif judge_model is None:
                print(f"{Fore.RED}❌ No judge model specified{Style.RESET_ALL}")
                return False
            
            # Get all student models
            student_models = get_available_models('student')
            
            judge_provider, judge_model_name = judge_model
            print(f"{Fore.CYAN}🎯 Testing {len(student_models)} student models against judge: {Fore.YELLOW}{judge_provider}:{judge_model_name}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📋 Using prompt version: {Fore.YELLOW}{prompt_version}{Style.RESET_ALL}\n")
            
            # Color list for cycling through student models
            model_colors = [Fore.RED, Fore.GREEN, Fore.BLUE, Fore.CYAN, Fore.MAGENTA, Fore.YELLOW]
            results = []
            
            total_tests = len(student_models)
            
            for student_idx, (student_provider, student_model_name) in enumerate(student_models):
                current_test = student_idx + 1
                
                # Choose color for this student model
                student_color = model_colors[student_idx % len(model_colors)]
                judge_color = Fore.WHITE  # Keep judge consistent
                
                # Colorful delimiter
                print(f"\n{student_color}{'▲'*35} TEST {current_test}/{total_tests} {'▲'*35}{Style.RESET_ALL}")
                print(f"{student_color}🎓 STUDENT: {student_provider}:{student_model_name}{Style.RESET_ALL}")
                print(f"{judge_color}⚖️  JUDGE: {judge_provider}:{judge_model_name}{Style.RESET_ALL}")
                print(f"{student_color}{'▼'*75}{Style.RESET_ALL}\n")
                
                try:
                    # Run evaluation with this student model
                    success = self.evaluate_only(
                        prompt_version=prompt_version,
                        student_model=(student_provider, student_model_name),
                        judge_model=judge_model
                    )
                    
                    if success:
                        # Get the final score - read the latest judgment summary
                        try:
                            with open('seed_oil_evaluation/judge_scores/judgment_summary.json', 'r') as f:
                                summary = json.load(f)
                                overall_score = summary.get('overall_metrics', {}).get('average_overall_score', 'N/A')
                                
                                # Colorful final score display
                                if isinstance(overall_score, (int, float)):
                                    if overall_score >= 80:
                                        score_color = Fore.GREEN
                                        score_icon = "🏆"
                                    elif overall_score >= 60:
                                        score_color = Fore.YELLOW
                                        score_icon = "⭐"
                                    else:
                                        score_color = Fore.RED
                                        score_icon = "📉"
                                    
                                    print(f"\n{score_color}{Back.BLACK} {score_icon} FINAL SCORE: {overall_score:.1f}/100 {score_icon} {Style.RESET_ALL}")
                                else:
                                    print(f"\n{Fore.WHITE}{Back.RED} ❌ SCORE: {overall_score} {Style.RESET_ALL}")
                                
                                results.append({
                                    'student_model': f"{student_provider}:{student_model_name}",
                                    'score': overall_score,
                                    'success': True
                                })
                        except Exception as e:
                            print(f"\n{Fore.WHITE}{Back.RED} ❌ Could not read final score: {str(e)} {Style.RESET_ALL}")
                            results.append({
                                'student_model': f"{student_provider}:{student_model_name}",
                                'score': 'Error',
                                'success': False
                            })
                    else:
                        print(f"\n{Fore.WHITE}{Back.RED} ❌ EVALUATION FAILED {Style.RESET_ALL}")
                        results.append({
                            'student_model': f"{student_provider}:{student_model_name}",
                            'score': 'Failed',
                            'success': False
                        })
                
                except Exception as e:
                    print(f"\n{Fore.WHITE}{Back.RED} ❌ ERROR: {str(e)} {Style.RESET_ALL}")
                    results.append({
                        'student_model': f"{student_provider}:{student_model_name}",
                        'score': 'Error',
                        'success': False
                    })
                
                # End delimiter
                print(f"\n{student_color}{'='*75}{Style.RESET_ALL}")
            
            # Final summary
            print(f"\n{Fore.MAGENTA}{'🎉'*25} RAINBOWBLAST COMPLETE {'🎉'*25}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📊 FINAL RESULTS SUMMARY ({judge_provider}:{judge_model_name} judge):{Style.RESET_ALL}\n")
            
            # Sort results by score (highest first)
            successful_results = [r for r in results if isinstance(r['score'], (int, float))]
            successful_results.sort(key=lambda x: x['score'], reverse=True)
            
            if successful_results:
                print(f"{Fore.GREEN}🏆 STUDENT MODEL RANKINGS:{Style.RESET_ALL}")
                for i, result in enumerate(successful_results):
                    score = result['score']
                    if score >= 80:
                        color = Fore.GREEN
                    elif score >= 60:
                        color = Fore.YELLOW
                    else:
                        color = Fore.RED
                    
                    print(f"{color}{i+1}. {result['student_model']}: {score:.1f}/100{Style.RESET_ALL}")
            
            failed_results = [r for r in results if not r['success']]
            if failed_results:
                print(f"\n{Fore.RED}❌ FAILED TESTS: {len(failed_results)}{Style.RESET_ALL}")
                for result in failed_results:
                    print(f"{Fore.RED}  • {result['student_model']}: {result['score']}{Style.RESET_ALL}")
            
            print(f"\n{Fore.MAGENTA}{'='*75}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"\n{Fore.WHITE}{Back.RED} ❌ RAINBOWBLAST FAILED: {str(e)} {Style.RESET_ALL}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Seed Oil Sleuth QA Pipeline')
    parser.add_argument('--full', action='store_true', 
                       help='Run complete pipeline (generate + evaluate)')
    parser.add_argument('--add-conversations', type=int, metavar='N',
                       help='Add N new conversations to existing dataset')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Run only student and judge evaluation')
    parser.add_argument('--show-results', action='store_true',
                       help='Show latest results summary')
    parser.add_argument('--clean', action='store_true',
                       help='Clean all results (keep conversations)')
    parser.add_argument('--rainbowblast', action='store_true',
                       help='Test ALL student models against one judge with colorful output')
    parser.add_argument('--rainbowblast-judge', type=str, metavar='JUDGE_MODEL',
                       help='Judge model for rainbowblast (format: provider:model)')
    
    args = parser.parse_args()
    
    orchestrator = QAPipelineOrchestrator()
    
    # If no arguments provided, show interactive menu
    if len(sys.argv) == 1:
        success = orchestrator.interactive_menu()
        return 0 if success else 1
    
    # Handle command line arguments
    if args.full:
        success = orchestrator.run_full_pipeline()
    elif args.add_conversations:
        success = orchestrator.add_conversations(args.add_conversations)
    elif args.evaluate_only:
        success = orchestrator.evaluate_only()
    elif args.show_results:
        orchestrator.show_results_summary()
        success = True
    elif args.clean:
        orchestrator.clean_results()
        success = True
    elif args.rainbowblast:
        if args.rainbowblast_judge:
            # Parse judge model from CLI argument
            if ':' not in args.rainbowblast_judge:
                print(f"❌ Invalid judge model format. Use 'provider:model' (e.g., 'openai:gpt-4o')")
                success = False
            else:
                provider, model = args.rainbowblast_judge.split(':', 1)
                judge_model = (provider.strip(), model.strip())
                success = orchestrator.rainbowblast(judge_model=judge_model, interactive=False)
        else:
            print(f"❌ --rainbowblast requires --rainbowblast-judge parameter")
            print(f"   Example: --rainbowblast --rainbowblast-judge openai:gpt-4o")
            success = False
    else:
        print("🔧 Seed Oil Sleuth QA Pipeline")
        print("=" * 35)
        print("Usage examples:")
        print("  python run_qa_pipeline.py                           # Interactive menu")
        print("  python run_qa_pipeline.py --full                    # Run complete pipeline")
        print("  python run_qa_pipeline.py --add-conversations 5     # Add 5 new conversations")
        print("  python run_qa_pipeline.py --evaluate-only           # Re-run evaluation only")
        print("  python run_qa_pipeline.py --rainbowblast --rainbowblast-judge openai:gpt-4o  # Test ALL students vs 1 judge")
        print("  python run_qa_pipeline.py --show-results            # Show latest results")
        print("  python run_qa_pipeline.py --clean                   # Clean results, keep conversations")
        success = True
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())