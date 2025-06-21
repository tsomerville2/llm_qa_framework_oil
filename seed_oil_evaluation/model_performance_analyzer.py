#!/usr/bin/env python3
"""
Model Performance Analyzer for Seed Oil Sleuth QA Pipeline
Analyzes LLM performance across different scoring categories with detailed breakdowns
"""

import json
import os
from typing import Dict, List, Optional
from scoring_database import ScoringDatabase

class ModelPerformanceAnalyzer:
    def __init__(self):
        self.db = ScoringDatabase()
        self.scoring_weights = self.load_scoring_weights()
        
    def load_scoring_weights(self) -> Dict[str, int]:
        """Load scoring weights from configuration file"""
        weights_file = "seed_oil_evaluation/scoring_weights.txt"
        weights = {}
        
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if ':' in line:
                            name, value = line.split(':', 1)
                            weights[name.strip()] = int(value.strip())
        
        return weights
    
    def analyze_model_performance(self, model_name: str, prompt_version: str = None) -> Dict:
        """Analyze performance for a specific model and prompt version"""
        
        # Get all runs for this model
        all_runs = self.db.get_latest_runs(limit=100)
        
        # Filter by model and prompt version
        filtered_runs = []
        for run in all_runs:
            if model_name in run.get("student_model", ""):
                if prompt_version is None or run.get("prompt_version") == prompt_version:
                    filtered_runs.append(run)
        
        if not filtered_runs:
            return {"error": f"No runs found for model {model_name}"}
        
        # Calculate detailed performance metrics
        analysis = {
            "model_name": model_name,
            "prompt_version": prompt_version or "all_versions",
            "total_runs": len(filtered_runs),
            "overall_performance": self.calculate_overall_performance(filtered_runs),
            "dimension_analysis": self.analyze_dimensions(filtered_runs),
            "report_triggering_analysis": self.analyze_report_triggering(filtered_runs),
            "conversation_difficulty_breakdown": self.analyze_by_difficulty(filtered_runs),
            "improvement_trends": self.analyze_trends(filtered_runs)
        }
        
        return analysis
    
    def calculate_overall_performance(self, runs: List[Dict]) -> Dict:
        """Calculate overall performance statistics"""
        if not runs:
            return {}
        
        overall_scores = [run.get("overall_score", 0) for run in runs]
        percentage_scores = [run.get("percentage_score", 0) for run in runs]
        
        return {
            "average_score": round(sum(overall_scores) / len(overall_scores), 2),
            "max_score": max(overall_scores),
            "min_score": min(overall_scores),
            "average_percentage": round(sum(percentage_scores) / len(percentage_scores), 1),
            "score_distribution": self.get_score_distribution(overall_scores),
            "grade_distribution": self.get_grade_distribution(runs)
        }
    
    def analyze_dimensions(self, runs: List[Dict]) -> Dict:
        """Analyze performance across all scoring dimensions"""
        dimension_analysis = {}
        
        for dimension, max_points in self.scoring_weights.items():
            scores = []
            for run in runs:
                dim_scores = run.get("dimension_scores", {})
                scores.append(dim_scores.get(dimension, 0))
            
            if scores:
                avg_score = sum(scores) / len(scores)
                dimension_analysis[dimension] = {
                    "average_score": round(avg_score, 2),
                    "max_possible": max_points,
                    "percentage": round((avg_score / max_points) * 100, 1),
                    "max_achieved": max(scores),
                    "min_achieved": min(scores),
                    "consistency": self.calculate_consistency(scores),
                    "performance_level": self.get_performance_level(avg_score, max_points)
                }
        
        return dimension_analysis
    
    def analyze_report_triggering(self, runs: List[Dict]) -> Dict:
        """Analyze report triggering performance in detail"""
        report_stats = {
            "total_conversations": 0,
            "correct_triggering": 0,
            "premature_reports": 0,
            "missing_reports": 0,
            "oil_report_qualitys": 0,
            "conversation_completeness_breakdown": {"complete": 0, "partial": 0},
            "report_accuracy_by_completeness": {}
        }
        
        for run in runs:
            # Get individual results from the run
            individual_results = run.get("individual_scores", [])
            
            for result in individual_results:
                report_analysis = result.get("report_analysis", {})
                report_stats["total_conversations"] += 1
                
                # Track conversation completeness
                was_complete = report_analysis.get("conversation_was_complete", False)
                if was_complete:
                    report_stats["conversation_completeness_breakdown"]["complete"] += 1
                else:
                    report_stats["conversation_completeness_breakdown"]["partial"] += 1
                
                # Track report triggering accuracy
                if report_analysis.get("report_triggering_correct", False):
                    report_stats["correct_triggering"] += 1
                
                # Track penalties
                penalties = report_analysis.get("major_penalties_applied", [])
                if "premature_report" in penalties:
                    report_stats["premature_reports"] += 1
                if "missing_report" in penalties:
                    report_stats["missing_reports"] += 1
                if "oil_report_quality" in penalties:
                    report_stats["oil_report_qualitys"] += 1
        
        # Calculate rates
        if report_stats["total_conversations"] > 0:
            total = report_stats["total_conversations"]
            report_stats["accuracy_rates"] = {
                "correct_triggering_rate": round((report_stats["correct_triggering"] / total) * 100, 1),
                "premature_report_rate": round((report_stats["premature_reports"] / total) * 100, 1),
                "missing_report_rate": round((report_stats["missing_reports"] / total) * 100, 1),
                "oil_report_quality_rate": round((report_stats["oil_report_qualitys"] / total) * 100, 1)
            }
        
        return report_stats
    
    def analyze_by_difficulty(self, runs: List[Dict]) -> Dict:
        """Analyze performance breakdown by conversation difficulty"""
        difficulty_breakdown = {}
        
        for run in runs:
            individual_results = run.get("individual_scores", [])
            
            for result in individual_results:
                # This would need to be extracted from the original conversation metadata
                # For now, we'll skip this analysis until we have access to that data
                pass
        
        return {"note": "Difficulty analysis requires individual conversation metadata"}
    
    def analyze_trends(self, runs: List[Dict]) -> Dict:
        """Analyze performance trends over time"""
        if len(runs) < 2:
            return {"note": "Need at least 2 runs to analyze trends"}
        
        # Sort runs by timestamp
        sorted_runs = sorted(runs, key=lambda x: x.get("timestamp", ""))
        
        # Compare first and last runs
        first_run = sorted_runs[0]
        last_run = sorted_runs[-1]
        
        score_change = last_run.get("overall_score", 0) - first_run.get("overall_score", 0)
        
        return {
            "total_runs": len(runs),
            "time_span": f"{first_run.get('timestamp', '')[:10]} to {last_run.get('timestamp', '')[:10]}",
            "score_improvement": round(score_change, 2),
            "percentage_improvement": round(score_change / 100 * 100, 1),
            "trend": "improving" if score_change > 0 else "declining" if score_change < 0 else "stable"
        }
    
    def get_score_distribution(self, scores: List[float]) -> Dict:
        """Get distribution of scores across ranges"""
        ranges = {
            "90-100": 0,
            "80-89": 0,
            "70-79": 0,
            "60-69": 0,
            "50-59": 0,
            "below-50": 0
        }
        
        for score in scores:
            if score >= 90:
                ranges["90-100"] += 1
            elif score >= 80:
                ranges["80-89"] += 1
            elif score >= 70:
                ranges["70-79"] += 1
            elif score >= 60:
                ranges["60-69"] += 1
            elif score >= 50:
                ranges["50-59"] += 1
            else:
                ranges["below-50"] += 1
        
        return ranges
    
    def get_grade_distribution(self, runs: List[Dict]) -> Dict:
        """Get distribution of letter grades"""
        grades = {}
        
        for run in runs:
            individual_results = run.get("individual_scores", [])
            for result in individual_results:
                grade = result.get("grade", "Unknown")
                grades[grade] = grades.get(grade, 0) + 1
        
        return grades
    
    def calculate_consistency(self, scores: List[float]) -> str:
        """Calculate consistency rating based on score variance"""
        if not scores or len(scores) < 2:
            return "insufficient_data"
        
        avg = sum(scores) / len(scores)
        variance = sum((score - avg) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Consistency based on standard deviation relative to average
        consistency_ratio = std_dev / avg if avg > 0 else float('inf')
        
        if consistency_ratio < 0.1:
            return "very_consistent"
        elif consistency_ratio < 0.2:
            return "consistent"
        elif consistency_ratio < 0.3:
            return "moderately_consistent"
        else:
            return "inconsistent"
    
    def get_performance_level(self, score: float, max_score: float) -> str:
        """Get performance level classification"""
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        if percentage >= 90:
            return "excellent"
        elif percentage >= 80:
            return "very_good"
        elif percentage >= 70:
            return "good"
        elif percentage >= 60:
            return "satisfactory"
        elif percentage >= 50:
            return "needs_improvement"
        else:
            return "poor"
    
    def generate_performance_report(self, model_name: str, prompt_version: str = None) -> str:
        """Generate a comprehensive performance report"""
        analysis = self.analyze_model_performance(model_name, prompt_version)
        
        if "error" in analysis:
            return f"❌ {analysis['error']}"
        
        report = []
        report.append(f"📊 PERFORMANCE ANALYSIS: {analysis['model_name']}")
        report.append(f"📋 Prompt Version: {analysis['prompt_version']}")
        report.append(f"🔢 Total Runs: {analysis['total_runs']}")
        report.append("=" * 60)
        
        # Overall Performance
        overall = analysis['overall_performance']
        report.append(f"\n🎯 OVERALL PERFORMANCE:")
        report.append(f"  Average Score: {overall['average_score']}/100 ({overall['average_percentage']}%)")
        report.append(f"  Score Range: {overall['min_score']} - {overall['max_score']}")
        
        # Dimension Analysis
        report.append(f"\n📈 DIMENSION BREAKDOWN:")
        dim_analysis = analysis['dimension_analysis']
        for dimension, stats in dim_analysis.items():
            level = stats['performance_level'].replace('_', ' ').title()
            consistency = stats['consistency'].replace('_', ' ').title()
            report.append(f"  {dimension.replace('_', ' ').title():<25}: {stats['average_score']}/{stats['max_possible']} ({stats['percentage']}%) - {level} ({consistency})")
        
        # Report Triggering Analysis
        report.append(f"\n🎯 REPORT TRIGGERING ANALYSIS:")
        report_analysis = analysis['report_triggering_analysis']
        if 'accuracy_rates' in report_analysis:
            rates = report_analysis['accuracy_rates']
            report.append(f"  ✅ Correct Triggering: {rates['correct_triggering_rate']}%")
            report.append(f"  ⚠️ Premature Reports: {rates['premature_report_rate']}%")
            report.append(f"  ❌ Missing Reports: {rates['missing_report_rate']}%")
            report.append(f"  📝 Incomplete Reports: {rates['oil_report_quality_rate']}%")
        
        # Trends
        trends = analysis['improvement_trends']
        if 'trend' in trends:
            report.append(f"\n📈 PERFORMANCE TRENDS:")
            report.append(f"  Trend: {trends['trend'].title()}")
            report.append(f"  Score Change: {trends['score_improvement']} points")
        
        return "\n".join(report)

if __name__ == "__main__":
    analyzer = ModelPerformanceAnalyzer()
    
    # Example usage
    print("🔍 Model Performance Analyzer")
    print("=" * 40)
    
    # Analyze current model performance
    report = analyzer.generate_performance_report("groq:meta-llama/llama-4-maverick-17b-128e-instruct", "v1.0")
    print(report)