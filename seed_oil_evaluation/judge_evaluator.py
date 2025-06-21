#!/usr/bin/env python3
"""
Judge Evaluator for Seed Oil Sleuth Evaluation
Uses OpenAI o3-pro to score student responses on multiple dimensions
"""

import os
import json
import uuid
from datetime import datetime
from model_selector import select_model, get_model_client
from judge_oil_contents import OilContentsJudge

class JudgeEvaluator:
    def __init__(self, interactive=False, selected_model=None, pause_mode=False):
        self.responses_dir = "seed_oil_evaluation/student_responses"
        self.scores_dir = "seed_oil_evaluation/judge_scores"
        self.interactive = interactive
        self.pause_mode = pause_mode  # Toggle for pause functionality
        
        # Load scoring weights
        self.weights = self.load_scoring_weights()
        
        # Use pre-selected model or select new one
        if selected_model:
            provider, model = selected_model
            model_selection = selected_model
        else:
            model_selection = select_model('judge', interactive=interactive)
            if not model_selection:
                raise ValueError("No judge model selected")
            provider, model = model_selection
        
        self.client, self.model = get_model_client(provider, model)
        self.provider = provider
        self.judge_model_name = f"{provider}:{model}"
        
        # Initialize specialized Oil Contents Judge
        self.oil_contents_judge = OilContentsJudge(self.client, self.model, self.provider)
        
        print(f"🤖 Using judge model: {self.judge_model_name}")
        print(f"🛢️ Oil Contents Judge initialized")
        
        # Show initial menu if interactive
        if self.interactive:
            self.show_menu()
    
    def show_menu(self):
        """Show menu options for pause mode toggle"""
        print("\n" + "="*60)
        print("JUDGE EVALUATOR MENU")
        print("="*60)
        print("Commands:")
        print("  p - Toggle pause mode (currently: {})".format("ON" if self.pause_mode else "OFF"))
        print("  s - Start evaluation")
        print("  q - Quit")
        print("="*60)
        
        while True:
            choice = input("Enter command: ").lower().strip()
            if choice == 'p':
                self.pause_mode = not self.pause_mode
                status = "ON" if self.pause_mode else "OFF"
                print(f"📱 Pause mode: {status}")
                if self.pause_mode:
                    print("   → After each report judgment, you'll see the oil report for review")
            elif choice == 's':
                print("🚀 Starting evaluation...")
                break
            elif choice == 'q':
                print("👋 Goodbye!")
                exit(0)
            else:
                print("❌ Invalid choice. Use p/s/q")
    
    def display_oil_report_panel(self, student_response, conversation_id):
        """Display the oil report in a formatted panel for review"""
        try:
            import json
            response_data = json.loads(student_response)
            report = response_data.get("report", {})
            
            if not report:
                print("\n" + "="*80)
                print("📋 NO REPORT GENERATED")
                print("="*80)
                print("The student did not generate a report for this conversation.")
                print("="*80)
                return
            
            # Extract report components
            scores = report.get("scores", {})
            summary = report.get("summary", "No summary provided")
            tips = report.get("practicalTips", [])
            conclusion = report.get("conclusion", "No conclusion provided")
            
            # Display formatted panel
            print("\n" + "="*80)
            print(f"📋 OIL REPORT REVIEW - {conversation_id[:8]}")
            print("="*80)
            
            # Scores section
            print("📊 SCORES:")
            print(f"   LA Estimate: {scores.get('estimatedTotalLa', 'N/A')}")
            print(f"   Daily Calories: {scores.get('dailyCalories', 'N/A')}")
            print(f"   LA % of Calories: {scores.get('laPercentageOfCalories', 'N/A')}")
            print(f"   Score: {scores.get('score', 'N/A')}")
            print(f"   Target: {scores.get('idealTarget', 'N/A')}")
            
            print("\n📝 SUMMARY:")
            # Wrap long text
            summary_wrapped = self.wrap_text(summary, 70)
            for line in summary_wrapped:
                print(f"   {line}")
            
            print("\n💡 PRACTICAL TIPS:")
            for i, tip in enumerate(tips, 1):
                tip_wrapped = self.wrap_text(tip, 65)
                print(f"   {i}. {tip_wrapped[0]}")
                for line in tip_wrapped[1:]:
                    print(f"      {line}")
            
            print("\n🎯 CONCLUSION:")
            conclusion_wrapped = self.wrap_text(conclusion, 70)
            for line in conclusion_wrapped:
                print(f"   {line}")
            
            print("="*80)
            
        except json.JSONDecodeError:
            print("\n" + "="*80)
            print("❌ INVALID REPORT FORMAT")
            print("="*80)
            print("Could not parse the student response as valid JSON.")
            print("="*80)
        except Exception as e:
            print(f"\n❌ Error displaying report: {e}")
    
    def wrap_text(self, text, width):
        """Simple text wrapping for display"""
        if len(text) <= width:
            return [text]
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
    
    def wait_for_continue(self):
        """Wait for user to press a key to continue"""
        print("\n🔄 Press Enter to continue to next judgment...")
        input()
        print("\n" + "🚀 " + "="*78 + " 🚀")
        print("🚀 MOVING TO NEXT CONVERSATION...")
        print("🚀 " + "="*78 + " 🚀")
        print()  # Extra spacing
    
    def display_conversation_banner(self, conv_num, conversation_id, metadata):
        """Display a banner showing conversation info when in pause mode"""
        conv_type = metadata.get("conversation_type", "unknown")
        expected_oils = metadata.get("expected_seed_oil_sources", [])
        estimated_count = metadata.get("estimated_hidden_seed_oils", 0)
        difficulty = metadata.get("difficulty_level", "unknown")
        
        print("🔍 " + "="*78 + " 🔍")
        conv_display = f"#{conv_num}" if conv_num is not None else "N/A"
        print(f"📋 CONVERSATION {conv_display} - {conversation_id[:8]}")
        print("="*80)
        print(f"📊 Type: {conv_type.replace('_', ' ').title()}")
        print(f"⚡ Difficulty: {difficulty.title()}")
        print(f"🎯 Expected Seed Oil Sources ({len(expected_oils)}): {', '.join(expected_oils) if expected_oils else 'None expected'}")
        print(f"🔢 Estimated Hidden Oils: {estimated_count}")
        print("="*80)

    def calculate_standard_grade(self, score):
        """Convert numerical score to standard school grade"""
        # Standard grading scale (can be adjusted)
        if score >= 97:
            return "A+"
        elif score >= 93:
            return "A"
        elif score >= 90:
            return "A-"
        elif score >= 87:
            return "B+"
        elif score >= 83:
            return "B"
        elif score >= 80:
            return "B-"
        elif score >= 77:
            return "C+"
        elif score >= 73:
            return "C"
        elif score >= 70:
            return "C-"
        elif score >= 67:
            return "D+"
        elif score >= 65:
            return "D"
        else:
            return "F"
    
    def load_scoring_weights(self):
        """Load scoring weights from configuration file"""
        weights_file = "seed_oil_evaluation/scoring_weights.txt"
        default_weights = {
            "correct_report_timing": 60,
            "missing_reports": 20,
            "premature_reports": 10,
            "oil_report_quality": 10
        }
        
        if not os.path.exists(weights_file):
            print(f"⚠️ Scoring weights file not found at {weights_file}, using defaults")
            return default_weights
        
        try:
            weights = {}
            with open(weights_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line and not line.startswith('#'):
                        dim, weight = line.split(':', 1)
                        weights[dim.strip()] = int(weight.strip())
            
            # Validate that all required dimensions are present
            for dim in default_weights:
                if dim not in weights:
                    print(f"⚠️ Missing dimension '{dim}' in weights file, using default")
                    weights[dim] = default_weights[dim]
            
            print(f"✅ Loaded scoring weights: {weights}")
            return weights
            
        except Exception as e:
            print(f"⚠️ Error loading scoring weights: {e}, using defaults")
            return default_weights
        
    def create_judging_prompt(self, conversation_data, student_response, metadata):
        """Create the prompt for the judge to evaluate the student response"""
        
        return f"""
You are an expert evaluator for "The Seed Oil Sleuth" AI assistant. Your task is to score how well a student LLM followed the complex system instructions.

## ORIGINAL CONVERSATION:
{json.dumps(conversation_data['conversation']['messages'], indent=2)}

## CONVERSATION METADATA:
- Difficulty Level: {metadata['difficulty_level']}
- User Cooperation: {metadata['user_cooperation_level']}
- Expected Completeness: {metadata['conversation_completeness']}
- Expected Seed Oil Sources: {metadata['expected_seed_oil_sources']}
- Should Trigger Report: {metadata['should_trigger_report']}

## STUDENT MODEL RESPONSE:
{student_response}

## CRITICAL EVALUATION RULES:

### REPORT TRIGGERING LOGIC (MOST IMPORTANT):
**ONLY trigger reports when:**
- Conversation completeness is "complete" AND user has finished discussing all meals (breakfast→lunch→dinner+snacks)
- OR user explicitly requests LA report or says "that's all the food I'll eat today"

**MAJOR PENALTIES:**
- PREMATURE REPORT: -25 points if report given when conversation incomplete (should_trigger_report=false)
- MISSING REPORT: -25 points if no report given when conversation complete (should_trigger_report=true)
- INCOMPLETE REPORT: -10 points if report fields are empty when report is triggered

## EVALUATION CRITERIA - REPORT TRIGGERING BEHAVIOR ONLY:
Score ONLY on these 4 report behaviors (total 100 points):

**CRITICAL NEG TEST LOGIC: For NEG tests (should_trigger_report=false), NOT giving a report is PERFECT behavior and should receive FULL points in ALL dimensions. Do NOT penalize for empty reports on NEG tests.**

1. **Correct Report Timing** ({self.weights['correct_report_timing']} points):
   - Full points: Correctly triggered report when needed (POS tests) OR correctly did NOT trigger when not needed (NEG tests)
   - Zero points: Wrong timing decisions (gave report on NEG test OR missed report on POS test)

2. **Missing Reports** ({self.weights['missing_reports']} points):
   - Full points: Gave report when needed (POS tests) OR correctly didn't give report when not needed (NEG tests)
   - Zero points: Failed to give report when conversation was complete AND should_trigger_report=true

3. **Premature Reports** ({self.weights['premature_reports']} points):
   - Full points: Did NOT give premature report (always good) OR gave no report on NEG test (perfect)
   - Zero points: ONLY when gave report too early on incomplete conversation

4. **Oil Report Quality** ({self.weights['oil_report_quality']} points):
   - Full points: Gave complete report when needed (POS tests) OR gave no report when not needed (NEG tests)
   - Partial points: Gave incomplete report when complete report was needed (POS tests only)
   - Zero points: ONLY when gave incomplete report on POS test that required complete report

IGNORE: seed oil detection, JSON compliance, conversation flow, mathematical accuracy, user engagement - WE DON'T CARE ABOUT THOSE

## CONVERSATION ANALYSIS:
- Conversation Completeness: {metadata['conversation_completeness']}
- Should Trigger Report: {metadata['should_trigger_report']}
- Expected Seed Oils: {metadata['expected_seed_oil_sources']}

**CRITICAL QUESTION: Did the student correctly handle report triggering based on conversation completeness?**

Please provide your evaluation in the following JSON format:

{{
  "overall_score": 0-100,
  "dimension_scores": {{
    "correct_report_timing": 0-{self.weights['correct_report_timing']},
    "missing_reports": 0-{self.weights['missing_reports']},
    "premature_reports": 0-{self.weights['premature_reports']},
    "oil_report_quality": 0-{self.weights['oil_report_quality']}
  }},
  "report_analysis": {{
    "conversation_was_complete": true|false,
    "should_have_triggered_report": true|false,
    "student_triggered_report": true|false,
    "report_triggering_correct": true|false,
    "report_content_complete": true|false,
    "major_penalties_applied": ["premature_report"|"missing_report"|"incomplete_report"]
  }},
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
  "detailed_feedback": {{
    "correct_report_timing": "Detailed feedback on correct report triggering decisions",
    "missing_reports": "Detailed feedback on when reports were missing",
    "premature_reports": "Detailed feedback on reports given too early",
    "oil_report_quality": "Detailed feedback on the quality and completeness of report content"
  }},
  "improvement_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
  "grade": "A+|A|A-|B+|B|B-|C+|C|C-|D+|D|F"
}}

**GRADING SCALE REFERENCE:**
- A+ (97-100): Exceptional performance, perfect execution
- A (93-96): Excellent performance, minimal issues
- A- (90-92): Very good performance, minor issues
- B+ (87-89): Good performance, some issues
- B (83-86): Satisfactory performance, moderate issues
- B- (80-82): Below satisfactory, several issues
- C+ (77-79): Poor performance, many issues
- C (73-76): Very poor performance, major issues
- C- (70-72): Failing with minimal competency
- D+ (67-69): Clearly failing, severe issues
- D (65-66): Major failure
- F (0-64): Complete failure
"""

    def load_student_responses(self):
        """Load all student responses from the responses directory"""
        
        with open(f"{self.responses_dir}/evaluation_summary.json", 'r') as f:
            summary = json.load(f)
        
        # Include ALL responses - both successful and failed
        # Failed responses will get automatic 0 scores
        student_responses = summary["results"]
        
        return student_responses
    
    def load_conversation_data(self, conversation_id):
        """Load the original conversation data"""
        
        conversations_dir = "seed_oil_evaluation/conversations"
        with open(f"{conversations_dir}/index.json", 'r') as f:
            index = json.load(f)
        
        for conv_info in index["conversations"]:
            filepath = f"{conversations_dir}/{conv_info['file']}"
            with open(filepath, 'r') as f:
                conversation_data = json.load(f)
                if conversation_data["metadata"]["conversation_id"] == conversation_id:
                    return conversation_data
        
        return None
    
    def judge_response(self, student_data, conv_num=None):
        """Judge a single student response"""
        
        conversation_id = student_data["conversation_id"]
        conversation_data = self.load_conversation_data(conversation_id)
        
        if not conversation_data:
            print(f"Could not find conversation data for {conversation_id}")
            return None
        
        # Handle failed student evaluations - give them 0 scores
        if not student_data.get("evaluation_success", False):
            return {
                "conversation_id": conversation_id,
                "judged_at": datetime.now().isoformat(),
                "judge_model": self.judge_model_name,
                "student_model": student_data.get("student_model", "unknown"),
                "original_metadata": student_data["original_metadata"],
                "student_response": student_data.get("student_response"),
                "judge_scores": {
                    "overall_score": 0,
                    "dimension_scores": {
                        "correct_report_timing": 0,
                        "missing_reports": 0,
                        "premature_reports": 0,
                        "oil_report_quality": 0
                    },
                    "report_analysis": {
                        "conversation_was_complete": student_data["original_metadata"].get("conversation_completeness") == "complete",
                        "should_have_triggered_report": student_data["original_metadata"].get("should_trigger_report", False),
                        "student_triggered_report": False,
                        "report_triggering_correct": False,
                        "report_content_complete": False,
                        "major_penalties_applied": ["failed_to_respond"]
                    },
                    "strengths": [],
                    "weaknesses": ["Failed to generate valid response"],
                    "detailed_feedback": {
                        "correct_report_timing": "Student failed to generate any response",
                        "missing_reports": "No response generated",
                        "premature_reports": "No response generated", 
                        "oil_report_quality": "No response generated"
                    },
                    "improvement_suggestions": ["Fix technical issues preventing response generation"],
                    "grade": "F",
                    "penalty_details": {
                        "original_score": 0,
                        "total_penalty": 0,
                        "penalties_applied": ["failed_response"],
                        "adjusted_score": 0
                    }
                },
                "judgment_success": True  # Mark as successful judgment even though student failed
            }
        
        judging_prompt = self.create_judging_prompt(
            conversation_data,
            student_data["student_response"],
            student_data["original_metadata"]
        )
        
        try:
            # Handle different providers and models
            if self.provider == 'openai':
                if 'o1' in self.model or 'o3' in self.model:
                    # o1/o3 models require max_completion_tokens and support structured outputs
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": judging_prompt}],
                        max_completion_tokens=25000,
                        response_format={"type": "json_object"}
                    )
                elif 'gpt-4.1' in self.model:
                    # GPT-4.1 models don't support json_object mode yet, use prompt-based JSON
                    judging_prompt += "\n\nPlease respond with valid JSON only, no other text."
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": judging_prompt}],
                        temperature=0.7
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": judging_prompt}],
                        temperature=0.7
                    )
            else:
                # Groq or other providers
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": judging_prompt}],
                    temperature=0.7
                )
            
            judge_response = response.choices[0].message.content.strip()
            
            # Clean up response if it has markdown code blocks
            if judge_response.startswith("```json"):
                judge_response = judge_response[7:]
            if judge_response.endswith("```"):
                judge_response = judge_response[:-3]
            
            judge_data = json.loads(judge_response)
            
            # Get specialized Oil Contents evaluation
            oil_contents_result = self.oil_contents_judge.evaluate_report_contents(
                student_data["student_response"], 
                student_data["original_metadata"]
            )
            
            # Replace the oil_report_quality dimension score with the specialized evaluation
            if oil_contents_result.get("evaluation_success", False):
                # Store original overall score
                original_overall = judge_data.get("overall_score", 0)
                
                # Scale the 0-100 oil contents score to the weighted dimension max
                oil_score_scaled = (oil_contents_result["overall_score"] / 100) * self.weights["oil_report_quality"]
                judge_data["dimension_scores"]["oil_report_quality"] = round(oil_score_scaled, 1)
                
                # Recalculate overall score based on updated dimensions
                dimension_scores = judge_data["dimension_scores"]
                recalculated_overall = sum(dimension_scores.values())
                judge_data["overall_score"] = round(recalculated_overall, 1)
                
                # Apply standard grading scale
                standard_grade = self.calculate_standard_grade(recalculated_overall)
                judge_data["grade"] = standard_grade
                
                # Add detailed oil contents analysis to the judge data
                judge_data["oil_contents_analysis"] = oil_contents_result
                
                # Display oil quality feedback if available
                quality_feedback = oil_contents_result.get("quality_feedback", "No feedback available")
                print(f"🛢️ Oil Quality: {oil_contents_result['overall_score']}/100 - {quality_feedback}")
                print("-"*80)
                
                # Store pause info for later display (after judge analysis)
                if self.pause_mode:
                    judge_data["_pause_info"] = {
                        "conv_num": conv_num,
                        "conversation_id": conversation_id,
                        "metadata": student_data["original_metadata"],
                        "student_response": student_data["student_response"]
                    }
            else:
                print(f"⚠️ Oil Contents evaluation failed: {oil_contents_result.get('error', 'Unknown error')}")
                
                # Store pause info for failed evaluations too
                if self.pause_mode:
                    judge_data["_pause_info"] = {
                        "conv_num": conv_num,
                        "conversation_id": conversation_id,
                        "metadata": student_data["original_metadata"],
                        "student_response": student_data["student_response"],
                        "failed_evaluation": True
                    }
            
            # Apply major penalties to overall score
            judge_data = self.apply_penalty_adjustments(judge_data)
            
            # Ensure standard grading is applied even if Oil Contents Judge wasn't used
            if "oil_contents_analysis" not in judge_data:
                current_score = judge_data.get("overall_score", 0)
                standard_grade = self.calculate_standard_grade(current_score)
                judge_data["grade"] = standard_grade
            
            return judge_data
            
        except Exception as e:
            print(f"Error judging response {conversation_id}: {e}")
            return None
    
    def apply_penalty_adjustments(self, judge_data):
        """Apply major penalties to the overall score based on report analysis"""
        
        if "report_analysis" not in judge_data:
            return judge_data
        
        original_score = judge_data.get("overall_score", 0)
        penalties_applied = []
        total_penalty = 0
        
        report_analysis = judge_data["report_analysis"]
        major_penalties = report_analysis.get("major_penalties_applied", [])
        
        # Apply specific penalties
        if "premature_report" in major_penalties:
            total_penalty += 25
            penalties_applied.append("premature_report: -25 points")
        
        if "missing_report" in major_penalties:
            total_penalty += 25
            penalties_applied.append("missing_report: -25 points")
        
        # Skip incomplete_report penalty if we have Oil Contents analysis
        # The Oil Contents Judge now handles report quality evaluation
        if "incomplete_report" in major_penalties:
            if not judge_data.get("oil_contents_analysis"):
                # Only apply penalty if Oil Contents Judge wasn't used
                total_penalty += 10
                penalties_applied.append("incomplete_report: -10 points")
            else:
                # Oil Contents Judge score already reflects report quality
                penalties_applied.append("incomplete_report: handled by Oil Report Quality Judge")
        
        # Apply penalties to overall score
        adjusted_score = max(0, original_score - total_penalty)
        
        # Update judge data
        judge_data["overall_score"] = adjusted_score
        judge_data["penalty_details"] = {
            "original_score": original_score,
            "total_penalty": total_penalty,
            "penalties_applied": penalties_applied,
            "adjusted_score": adjusted_score
        }
        
        return judge_data
    
    def judge_all_responses(self):
        """Judge all student responses and save results"""
        
        student_responses = self.load_student_responses()
        judgment_results = []
        
        print("⚖️ JUDGE EVALUATIONS")
        print(f"{'#':<2} {'ID':<8} {'Type':<8} {'CorrectTiming':<12} {'MissedReports':<12} {'PrematureRep':<11} {'OilQuality':<10} {'Score':<5} {'Grade':<5} {'Issues':<15}")
        
        for i, student_data in enumerate(student_responses, 1):
            conv_id = student_data["conversation_id"]
            
            judge_score = self.judge_response(student_data, conv_num=i)
            
            if judge_score:
                judgment_result = {
                    "conversation_id": conv_id,
                    "judged_at": datetime.now().isoformat(),
                    "judge_model": self.judge_model_name,
                    "student_model": student_data["student_model"],
                    "original_metadata": student_data["original_metadata"],
                    "student_response": student_data["student_response"],
                    "judge_scores": judge_score,
                    "judgment_success": True
                }
                
                # Save individual judgment
                filename = f"judge_score_{i:03d}_{conv_id[:8]}.json"
                filepath = f"{self.scores_dir}/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(judgment_result, f, indent=2)
                
                judgment_results.append(judgment_result)
                # Analyze and display judge response details
                judge_analysis = self.analyze_judge_response(judge_score)
                conv_type = student_data["original_metadata"].get("conversation_type", "unknown")
                self.print_judge_analysis(i, conv_id, conv_type, judge_analysis, filename, judge_score.get('overall_score', 0), judge_score)
                
                # Show pause display after all analysis is complete
                if self.pause_mode and "_pause_info" in judge_score:
                    pause_info = judge_score["_pause_info"]
                    print("\n")  # Add spacing after judge analysis
                    self.display_conversation_banner(pause_info["conv_num"], pause_info["conversation_id"], pause_info["metadata"])
                    
                    if pause_info.get("failed_evaluation"):
                        print("\n📝 NO REPORT OR FAILED EVALUATION")
                        print("="*80)
                        print("The oil contents evaluation failed or no report was generated.")
                        print("="*80)
                    else:
                        self.display_oil_report_panel(pause_info["student_response"], pause_info["conversation_id"])
                    
                    self.wait_for_continue()
                    # Clean up the pause info from the data
                    del judge_score["_pause_info"]
                
            else:
                judgment_result = {
                    "conversation_id": conv_id,
                    "judged_at": datetime.now().isoformat(),
                    "judge_model": self.judge_model_name,
                    "student_model": student_data["student_model"],
                    "original_metadata": student_data["original_metadata"],
                    "student_response": student_data["student_response"],
                    "judge_scores": None,
                    "judgment_success": False,
                    "error": "Failed to generate judgment"
                }
                judgment_results.append(judgment_result)
                print(f"Failed to judge response {i}")
        
        # Calculate aggregate statistics
        successful_judgments = [r for r in judgment_results if r["judgment_success"]]
        if successful_judgments:
            overall_scores = [r["judge_scores"]["overall_score"] for r in successful_judgments]
            avg_score = sum(overall_scores) / len(overall_scores)
            
            # Use instance weights and calculate average dimension scores
            dimension_avgs = {}
            dimension_weights = self.weights
            
            for dim, max_points in dimension_weights.items():
                scores = [r["judge_scores"]["dimension_scores"].get(dim, 0) for r in successful_judgments]
                dimension_avgs[dim] = sum(scores) / len(scores)
            
            # Calculate report analysis statistics
            report_stats = {
                "correct_report_triggering": 0,
                "premature_reports": 0,
                "missing_reports": 0,
                "oil_report_quality": 0
            }
            
            for result in successful_judgments:
                report_analysis = result["judge_scores"].get("report_analysis", {})
                if report_analysis.get("report_triggering_correct", False):
                    report_stats["correct_report_triggering"] += 1
                
                penalties = report_analysis.get("major_penalties_applied", [])
                if "premature_report" in penalties:
                    report_stats["premature_reports"] += 1
                if "missing_report" in penalties:
                    report_stats["missing_reports"] += 1
                if "incomplete_report" in penalties:
                    report_stats["oil_report_quality"] += 1
        else:
            avg_score = 0
            dimension_avgs = {}
            report_stats = {
                "correct_report_triggering": 0,
                "premature_reports": 0,
                "missing_reports": 0,
                "oil_report_quality": 0
            }
        
        # Save master results  
        prompt_version = student_responses[0].get("prompt_version", "unknown") if student_responses else "unknown"
        
        results_summary = {
            "judged_at": datetime.now().isoformat(),
            "total_responses": len(student_responses),
            "successful_judgments": len(successful_judgments),
            "judge_model": self.judge_model_name,
            "student_model": student_responses[0]["student_model"] if student_responses else "unknown",
            "prompt_version": prompt_version,
            "aggregate_statistics": {
                "average_overall_score": round(avg_score, 2),
                "max_possible_score": 100,
                "percentage_score": round((avg_score / 100) * 100, 1),
                "dimension_averages": {k: round(v, 2) for k, v in dimension_avgs.items()},
                "report_analysis_stats": {
                    "correct_report_triggering_rate": round((report_stats["correct_report_triggering"] / len(successful_judgments)) * 100, 1) if successful_judgments else 0,
                    "premature_report_rate": round((report_stats["premature_reports"] / len(successful_judgments)) * 100, 1) if successful_judgments else 0,
                    "missing_report_rate": round((report_stats["missing_reports"] / len(successful_judgments)) * 100, 1) if successful_judgments else 0,
                    "oil_report_quality_rate": round((report_stats["oil_report_quality"] / len(successful_judgments)) * 100, 1) if successful_judgments else 0
                }
            },
            "results": judgment_results
        }
        
        with open(f"{self.scores_dir}/judgment_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save to TinyDB for historical tracking
        from scoring_database import ScoringDatabase
        
        try:
            db = ScoringDatabase()
            
            # Register prompt version if needed
            if prompt_version != "unknown":
                prompt_file = f"seed_oil_evaluation/prompts/{prompt_version}.json"
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r') as f:
                        prompt_data = json.load(f)
                    db.register_prompt_version(prompt_version, prompt_data)
            
            # Save evaluation run
            individual_scores = [r["judge_scores"] for r in successful_judgments if r["judgment_success"]]
            
            doc_id, run_id = db.save_evaluation_run(
                prompt_version=prompt_version,
                student_model=results_summary["student_model"],
                judge_model=results_summary["judge_model"],
                results_summary=results_summary,
                individual_scores=individual_scores
            )
            
            print(f"💾 Saved to database as run: {run_id}")
            
        except Exception as e:
            print(f"⚠️ Failed to save to database: {e}")
        
        return judgment_results
    
    def analyze_judge_response(self, judge_scores: dict) -> dict:
        """Analyze judge response for component presence and scoring"""
        
        dimension_scores = judge_scores.get("dimension_scores", {})
        report_analysis = judge_scores.get("report_analysis", {})
        penalty_details = judge_scores.get("penalty_details", {})
        
        analysis = {
            "overall_score": judge_scores.get("overall_score", 0),
            "grade": judge_scores.get("grade", "F"),
            "has_penalty": bool(penalty_details),
            "penalty_amount": penalty_details.get("total_penalty", 0),
            "major_penalties": report_analysis.get("major_penalties_applied", []),
            
            # Dimension scores - REPORT BEHAVIOR ONLY
            "correct_report_timing": dimension_scores.get("correct_report_timing", 0),
            "missing_reports": dimension_scores.get("missing_reports", 0),
            "premature_reports": dimension_scores.get("premature_reports", 0),
            "oil_report_quality": dimension_scores.get("oil_report_quality", 0),
            
            # Report analysis flags
            "conversation_complete": report_analysis.get("conversation_was_complete", False),
            "should_trigger": report_analysis.get("should_have_triggered_report", False),
            "student_triggered": report_analysis.get("student_triggered_report", False),
            "triggering_correct": report_analysis.get("report_triggering_correct", False),
            "content_complete": report_analysis.get("report_content_complete", False),
            
            # Has feedback components
            "has_strengths": bool(judge_scores.get("strengths")),
            "has_weaknesses": bool(judge_scores.get("weaknesses")),
            "has_suggestions": bool(judge_scores.get("improvement_suggestions")),
            "has_detailed_feedback": bool(judge_scores.get("detailed_feedback"))
        }
        
        return analysis
    
    def print_judge_analysis(self, conv_num: int, conv_id: str, conv_type: str, analysis: dict, filename: str, overall_score: float, judge_scores: dict = None):
        """Print clean analysis of judge response"""
        
        # Create issues summary - prioritize oil quality feedback
        issues = []
        if analysis["major_penalties"]:
            issues.extend(analysis["major_penalties"])
        if not analysis["triggering_correct"]:
            issues.append("wrong_trigger")
        
        # Add oil quality feedback if available and score is not perfect
        oil_quality_score = analysis.get("oil_report_quality", 35)
        if oil_quality_score < 35 and judge_scores:  # Less than perfect oil quality
            # Try to get quality feedback from oil_contents_analysis
            oil_analysis = judge_scores.get("oil_contents_analysis", {})
            quality_feedback = oil_analysis.get("quality_feedback", "")
            if quality_feedback and "Perfect NEG test" not in quality_feedback:
                # Truncate long feedback for display
                feedback_short = quality_feedback[:25] + "..." if len(quality_feedback) > 25 else quality_feedback
                issues.append(f"oil:{feedback_short}")
        
        issues_str = ",".join(issues) if issues else "-"
        
        # Shorten conversation type for display
        type_short = conv_type.replace("positive_complete", "POS").replace("negative_incomplete", "NEG").replace("user_requested", "USER")
        
        # Format single clean row with manual alignment - REPORT BEHAVIOR ONLY
        print(f"{conv_num:2d} {conv_id[:8]} {type_short:<8s} {analysis['correct_report_timing']:12.1f} {analysis['missing_reports']:12.1f} {analysis['premature_reports']:11.1f} {analysis['oil_report_quality']:10.1f} {overall_score:5.1f} {analysis['grade']:<5s} {issues_str:<15s}")

if __name__ == "__main__":
    judge = JudgeEvaluator()
    results = judge.judge_all_responses()
    successful = len([r for r in results if r["judgment_success"]])
    print(f"\\nJudged {successful}/{len(results)} responses successfully!")
    
    if successful > 0:
        with open("seed_oil_evaluation/judge_scores/judgment_summary.json", 'r') as f:
            summary = json.load(f)
        
        stats = summary["aggregate_statistics"]
        print(f"\\n📊 EVALUATION RESULTS:")
        print(f"Average Overall Score: {stats['average_overall_score']}/60 ({stats['percentage_score']}%)")
        print(f"\\n📈 Dimension Averages:")
        for dim, score in stats['dimension_averages'].items():
            print(f"  {dim.replace('_', ' ').title()}: {score}/10")