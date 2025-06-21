#!/usr/bin/env python3
"""
Scoring Database Manager for Seed Oil Sleuth QA Pipeline
Manages evaluation results in TinyDB for historical tracking and comparison
"""

import os
import json
from datetime import datetime
from tinydb import TinyDB, Query
from typing import Dict, List, Optional

class ScoringDatabase:
    def __init__(self, db_path="seed_oil_evaluation/scoring_history.json"):
        self.db_path = db_path
        self.db = TinyDB(db_path)
        self.evaluations = self.db.table('evaluations')
        self.prompt_versions = self.db.table('prompt_versions')
        
    def get_available_prompts(self) -> List[Dict]:
        """Get list of available prompt versions"""
        prompts_dir = "seed_oil_evaluation/prompts"
        available_prompts = []
        
        if os.path.exists(prompts_dir):
            for filename in sorted(os.listdir(prompts_dir)):
                if filename.endswith('.json'):
                    version = filename.replace('.json', '')
                    filepath = os.path.join(prompts_dir, filename)
                    
                    with open(filepath, 'r') as f:
                        prompt_data = json.load(f)
                    
                    available_prompts.append({
                        "version": version,
                        "name": prompt_data.get("name", f"Version {version}"),
                        "description": prompt_data.get("description", ""),
                        "created_at": prompt_data.get("created_at", ""),
                        "filepath": filepath
                    })
        
        return available_prompts
    
    def register_prompt_version(self, version: str, prompt_data: Dict):
        """Register a prompt version in the database"""
        Prompt = Query()
        
        # Check if version already exists
        existing = self.prompt_versions.search(Prompt.version == version)
        
        if not existing:
            self.prompt_versions.insert({
                "version": version,
                "name": prompt_data.get("name", f"Version {version}"),
                "description": prompt_data.get("description", ""),
                "created_at": prompt_data.get("created_at", datetime.now().isoformat()),
                "registered_at": datetime.now().isoformat()
            })
    
    def save_evaluation_run(self, 
                          prompt_version: str,
                          student_model: str,
                          judge_model: str,
                          results_summary: Dict,
                          individual_scores: List[Dict]):
        """Save a complete evaluation run to the database"""
        
        # Extract aggregate statistics
        stats = results_summary.get("aggregate_statistics", {})
        
        evaluation_record = {
            "run_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prompt_version}",
            "timestamp": datetime.now().isoformat(),
            "prompt_version": prompt_version,
            "student_model": student_model,
            "judge_model": judge_model,
            "total_conversations": results_summary.get("total_responses", 0),
            "successful_evaluations": results_summary.get("successful_judgments", 0),
            "overall_score": stats.get("average_overall_score", 0),
            "percentage_score": stats.get("percentage_score", 0),
            "dimension_scores": stats.get("dimension_averages", {}),
            "individual_scores": individual_scores,
            "raw_results_summary": results_summary
        }
        
        # Insert into database
        doc_id = self.evaluations.insert(evaluation_record)
        
        return doc_id, evaluation_record["run_id"]
    
    def get_prompt_performance_history(self, prompt_version: str) -> List[Dict]:
        """Get all evaluation runs for a specific prompt version"""
        Evaluation = Query()
        return self.evaluations.search(Evaluation.prompt_version == prompt_version)
    
    def get_latest_runs(self, limit: int = 10) -> List[Dict]:
        """Get the most recent evaluation runs"""
        all_runs = self.evaluations.all()
        # Sort by timestamp, most recent first
        sorted_runs = sorted(all_runs, key=lambda x: x.get("timestamp", ""), reverse=True)
        return sorted_runs[:limit]
    
    def compare_prompt_versions(self, version1: str, version2: str) -> Dict:
        """Compare performance between two prompt versions"""
        Evaluation = Query()
        
        v1_runs = self.evaluations.search(Evaluation.prompt_version == version1)
        v2_runs = self.evaluations.search(Evaluation.prompt_version == version2)
        
        if not v1_runs or not v2_runs:
            return {"error": "One or both prompt versions have no evaluation data"}
        
        # Calculate averages for each version
        def calculate_avg_performance(runs):
            if not runs:
                return {}
            
            total_score = sum(run.get("overall_score", 0) for run in runs)
            total_percentage = sum(run.get("percentage_score", 0) for run in runs)
            
            # Calculate dimension averages
            dimension_totals = {}
            for run in runs:
                for dim, score in run.get("dimension_scores", {}).items():
                    dimension_totals[dim] = dimension_totals.get(dim, 0) + score
            
            dimension_avgs = {dim: total / len(runs) for dim, total in dimension_totals.items()}
            
            return {
                "runs_count": len(runs),
                "avg_overall_score": total_score / len(runs),
                "avg_percentage_score": total_percentage / len(runs),
                "avg_dimension_scores": dimension_avgs,
                "latest_run": max(runs, key=lambda x: x.get("timestamp", ""))["timestamp"]
            }
        
        v1_performance = calculate_avg_performance(v1_runs)
        v2_performance = calculate_avg_performance(v2_runs)
        
        # Calculate improvements
        score_diff = v2_performance.get("avg_overall_score", 0) - v1_performance.get("avg_overall_score", 0)
        percentage_diff = v2_performance.get("avg_percentage_score", 0) - v1_performance.get("avg_percentage_score", 0)
        
        return {
            "version1": version1,
            "version2": version2,
            "v1_performance": v1_performance,
            "v2_performance": v2_performance,
            "score_improvement": score_diff,
            "percentage_improvement": percentage_diff,
            "is_improvement": score_diff > 0
        }
    
    def get_performance_trends(self) -> Dict:
        """Get overall performance trends across all runs"""
        all_runs = self.evaluations.all()
        
        if not all_runs:
            return {"error": "No evaluation data available"}
        
        # Sort by timestamp
        sorted_runs = sorted(all_runs, key=lambda x: x.get("timestamp", ""))
        
        # Calculate trends
        scores_over_time = [(run["timestamp"], run.get("overall_score", 0)) for run in sorted_runs]
        
        # Group by prompt version
        version_performance = {}
        version_runs = {}
        for run in sorted_runs:
            version = run.get("prompt_version", "unknown")
            if version not in version_performance:
                version_performance[version] = []
                version_runs[version] = []
            version_performance[version].append(run.get("overall_score", 0))
            version_runs[version].append(run)
        
        # Calculate version averages with model info
        version_averages = {}
        for version, scores in version_performance.items():
            runs = version_runs[version]
            latest_run = max(runs, key=lambda x: x.get("timestamp", ""))
            
            version_averages[version] = {
                "average_score": sum(scores) / len(scores),
                "run_count": len(scores),
                "best_score": max(scores),
                "worst_score": min(scores),
                "student_model": latest_run.get("student_model", "Unknown"),
                "judge_model": latest_run.get("judge_model", "Unknown")
            }
        
        return {
            "total_runs": len(all_runs),
            "scores_over_time": scores_over_time,
            "version_performance": version_averages,
            "best_overall_run": max(sorted_runs, key=lambda x: x.get("overall_score", 0)),
            "latest_run": sorted_runs[-1]
        }
    
    def export_data(self, output_file: str = None) -> str:
        """Export all data to JSON file"""
        if not output_file:
            output_file = f"scoring_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "evaluations": self.evaluations.all(),
            "prompt_versions": self.prompt_versions.all(),
            "available_prompts": self.get_available_prompts()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_file
    
    def reset_all_data(self, confirmation_text: str) -> bool:
        """Reset all database data - requires typing 'RESETALLDATA' to confirm"""
        if confirmation_text != "RESETALLDATA":
            return False
        
        # Clear all tables
        self.evaluations.truncate()
        self.prompt_versions.truncate()
        
        # Close and recreate database file
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        # Reinitialize database
        self.db = TinyDB(self.db_path)
        self.evaluations = self.db.table('evaluations')
        self.prompt_versions = self.db.table('prompt_versions')
        
        return True
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_evaluations": len(self.evaluations),
            "total_prompt_versions": len(self.prompt_versions),
            "available_prompt_files": len(self.get_available_prompts()),
            "database_path": self.db_path,
            "database_size_mb": round(os.path.getsize(self.db_path) / 1024 / 1024, 2) if os.path.exists(self.db_path) else 0
        }

if __name__ == "__main__":
    # Example usage
    db = ScoringDatabase()
    
    print("📊 Database Stats:")
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\\n📋 Available Prompts:")
    prompts = db.get_available_prompts()
    for prompt in prompts:
        print(f"  {prompt['version']}: {prompt['name']}")