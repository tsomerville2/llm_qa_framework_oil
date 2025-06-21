#!/usr/bin/env python3
"""
Oil Contents Judge for Seed Oil Sleuth Evaluation

Dedicated judge for evaluating the quality and completeness of student-generated reports.
This judge focuses specifically on the content quality of reports when they are generated,
scoring on a 0-100 scale based on multiple detailed criteria.

Usage:
    from judge_oil_contents import OilContentsJudge
    judge = OilContentsJudge(client, model)
    score = judge.evaluate_report_contents(student_response, conversation_context)
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

class OilContentsJudge:
    def __init__(self, client, model, provider="openai"):
        """
        Initialize the Oil Contents Judge
        
        Args:
            client: LLM client (OpenAI, Groq, etc.)
            model: Model name string
            provider: Provider name (openai, groq, etc.)
        """
        self.client = client
        self.model = model
        self.provider = provider
        
    def create_contents_judging_prompt(self, student_response: str, conversation_context: Dict) -> str:
        """Create the specialized prompt for judging oil report contents"""
        
        return f"""
You are an expert evaluator specializing in assessing the quality and completeness of seed oil analysis reports.

## STUDENT'S REPORT TO EVALUATE:
{student_response}

## CONVERSATION CONTEXT:
- Expected seed oil sources: {conversation_context.get('expected_seed_oil_sources', [])}
- Estimated hidden oils: {conversation_context.get('estimated_hidden_seed_oils', 0)}
- Difficulty level: {conversation_context.get('difficulty_level', 'unknown')}

## EVALUATION CRITERIA (Total: 100 points)

### 1. NUMERICAL ACCURACY & CALCULATIONS (25 points)
**Score 0-25 points based on:**
- Are numeric values realistic and properly formatted? (not "X.X g" or "XXXX kcal")
- Do calculations appear mathematically sound?
- Is the LA percentage calculation reasonable? (LA grams * 9 kcal/g / total calories * 100)
- Are daily calorie estimates reasonable? (1500-3000 kcal typical range)
- Is the estimated total LA in grams realistic? (not obviously wrong)

**Scoring guide:**
- 25 pts: All numbers are realistic, properly calculated, well-formatted
- 20 pts: Most numbers good, minor calculation issues or formatting problems  
- 15 pts: Some realistic numbers, but major calculation errors or unrealistic values
- 5 pts: Mix of realistic and placeholder/unrealistic numbers (MAJOR PENALTY for placeholders)
- 2 pts: Mostly placeholder numbers but some attempt at real values (SEVERE PENALTY)
- 0 pts: All placeholder numbers ("X.X g", "XXXX kcal", etc.) (MAXIMUM PENALTY)

### 2. SUMMARY QUALITY & SPECIFICITY (25 points)
**Score 0-25 points based on:**
- Is the summary specific to the user's actual food intake from the conversation?
- Does it accurately reflect the seed oil sources identified?
- Is it personalized rather than generic template text?
- Does it provide meaningful insights about the user's LA intake level?

**Scoring guide:**
- 25 pts: Highly specific, personalized summary referencing actual foods discussed
- 20 pts: Good specificity with some personalization
- 15 pts: Somewhat specific but missing key details from conversation
- 10 pts: Generic but relevant summary with minimal personalization
- 5 pts: Very generic summary with placeholder language
- 0 pts: Completely generic template ("Brief summary of findings")

### 3. PRACTICAL TIPS RELEVANCE & QUALITY (25 points)
**Score 0-25 points based on:**
- Are tips specific to the actual foods/oils mentioned in the conversation?
- Do they provide actionable, practical advice?
- Are they relevant to the user's specific situation and intake level?
- Do they demonstrate understanding of seed oil sources identified?

**Scoring guide:**
- 25 pts: Highly specific tips directly addressing foods from conversation
- 20 pts: Good tips with clear relevance to user's situation
- 15 pts: Relevant tips but somewhat generic
- 10 pts: Basic tips with minimal connection to specific conversation
- 5 pts: Very generic tips with little relevance
- 0 pts: Placeholder tips ("tip 1", "tip 2", "tip 3")

### 4. CONCLUSION PERSONALIZATION & MOTIVATION (25 points)
**Score 0-25 points based on:**
- Is the conclusion personalized to the user's specific results?
- Does it provide appropriate encouragement or motivation?
- Is it specific rather than generic template language?
- Does it acknowledge the user's current LA intake level appropriately?

**Scoring guide:**
- 25 pts: Highly personalized, motivating conclusion reflecting user's specific situation
- 20 pts: Good personalization with appropriate tone
- 15 pts: Somewhat personalized but missing specificity
- 10 pts: Basic conclusion with minimal personalization
- 5 pts: Generic conclusion with template language
- 0 pts: Completely generic ("Encouraging conclusion with next steps")

## ADDITIONAL QUALITY INDICATORS

### BONUS CONSIDERATIONS (can add up to 5 bonus points):
- Exceptional insight into specific food sources
- Creative or particularly helpful practical suggestions
- Demonstrates deep understanding of seed oil content in foods
- Appropriate tone matching the severity of LA intake level

### PENALTY CONSIDERATIONS (subtract points):
- Factually incorrect information about seed oils (-5 to -10 points)
- Inappropriate recommendations for the user's intake level (-3 to -7 points)
- Confusing or contradictory information (-2 to -5 points)

## SCORING GUIDELINES BY OVERALL QUALITY:

**90-100 points: EXCELLENT**
- All sections are highly specific, accurate, and personalized
- Numbers are realistic and well-calculated
- Tips are directly relevant to conversation foods
- Conclusion is motivating and appropriate

**75-89 points: GOOD** 
- Most sections are specific and relevant
- Numbers are mostly realistic with minor issues
- Tips are relevant but may lack some specificity
- Conclusion is appropriate but may be somewhat generic

**60-74 points: ADEQUATE**
- Mix of specific and generic content
- Some realistic numbers mixed with placeholders
- Tips are somewhat relevant but generic
- Conclusion is basic but acceptable

**40-59 points: POOR**
- Mostly generic content with minimal specificity
- Many placeholder numbers or unrealistic calculations
- Tips are generic and not conversation-specific
- Conclusion is template-like

**20-39 points: VERY POOR**
- Almost entirely generic template content
- Mostly placeholder numbers
- Generic tips with no conversation relevance
- Generic conclusion

**0-19 points: FAILING**
- Completely generic template
- All placeholder numbers
- No personalization or specificity
- Pure template responses

Please provide your evaluation in the following JSON format:

{{
  "overall_score": 0-100,
  "dimension_scores": {{
    "numerical_accuracy": 0-25,
    "summary_quality": 0-25, 
    "practical_tips": 0-25,
    "conclusion_quality": 0-25
  }},
  "bonus_points": 0-5,
  "penalty_points": 0-10,
  "detailed_analysis": {{
    "numerical_accuracy_feedback": "Specific feedback on numbers and calculations",
    "summary_quality_feedback": "Specific feedback on summary content",
    "practical_tips_feedback": "Specific feedback on tip relevance and quality",
    "conclusion_quality_feedback": "Specific feedback on conclusion personalization",
    "overall_strengths": ["strength 1", "strength 2"],
    "overall_weaknesses": ["weakness 1", "weakness 2"],
    "improvement_suggestions": ["suggestion 1", "suggestion 2"]
  }},
  "quality_level": "EXCELLENT|GOOD|ADEQUATE|POOR|VERY_POOR|FAILING"
}}
"""

    def evaluate_report_contents(self, student_response: str, conversation_context: Dict) -> Dict[str, Any]:
        """
        Evaluate the quality and completeness of a student's report contents
        
        Args:
            student_response: JSON string of student's response
            conversation_context: Dictionary with conversation metadata
            
        Returns:
            Dictionary containing detailed evaluation results with 0-100 score
        """
        
        # Parse student response to check if it even contains a report
        try:
            parsed_response = json.loads(student_response)
            report_data = parsed_response.get("report")
            
            if not report_data:
                # Check if this is a NEG test - no report should get full score
                should_trigger = conversation_context.get("should_trigger_report", True)
                
                if not should_trigger:
                    # NEG test: no report is perfect behavior - give full score
                    return {
                        "overall_score": 100,
                        "dimension_scores": {
                            "numerical_accuracy": 25,
                            "summary_quality": 25,
                            "practical_tips": 25,
                            "conclusion_quality": 25
                        },
                        "bonus_points": 0,
                        "penalty_points": 0,
                        "detailed_analysis": {
                            "numerical_accuracy_feedback": "Correctly provided no report on NEG test",
                            "summary_quality_feedback": "Correctly provided no report on NEG test", 
                            "practical_tips_feedback": "Correctly provided no report on NEG test",
                            "conclusion_quality_feedback": "Correctly provided no report on NEG test",
                            "overall_strengths": ["Perfect NEG test behavior - correctly provided no report"],
                            "overall_weaknesses": [],
                            "improvement_suggestions": ["Continue this excellent NEG test behavior"]
                        },
                        "quality_level": "EXCELLENT",
                        "quality_feedback": "Perfect NEG test behavior - correctly provided no report",
                        "evaluation_success": True
                    }
                else:
                    # POS test: no report when one was expected - failure
                    return {
                        "overall_score": 0,
                        "dimension_scores": {
                            "numerical_accuracy": 0,
                            "summary_quality": 0,
                            "practical_tips": 0,
                            "conclusion_quality": 0
                        },
                        "bonus_points": 0,
                        "penalty_points": 0,
                        "detailed_analysis": {
                            "numerical_accuracy_feedback": "No report present when one was expected",
                            "summary_quality_feedback": "No report present when one was expected", 
                            "practical_tips_feedback": "No report present when one was expected",
                            "conclusion_quality_feedback": "No report present when one was expected",
                            "overall_strengths": [],
                            "overall_weaknesses": ["No report generated when one was expected"],
                            "improvement_suggestions": ["Generate a complete report with all required fields"]
                        },
                        "quality_level": "FAILING",
                        "quality_feedback": "No report generated when one was expected",
                        "evaluation_success": True
                    }
                
        except json.JSONDecodeError:
            return {
                "overall_score": 0,
                "evaluation_success": False,
                "error": "Invalid JSON in student response"
            }
        
        # Create the judging prompt
        judging_prompt = self.create_contents_judging_prompt(student_response, conversation_context)
        
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
                        temperature=0.3
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": judging_prompt}],
                        temperature=0.3,
                        response_format={"type": "json_object"}
                    )
            else:
                # Groq or other providers
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": judging_prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
            
            judge_response = response.choices[0].message.content.strip()
            
            # Clean up response if it has markdown code blocks
            if judge_response.startswith("```json"):
                judge_response = judge_response[7:]
            if judge_response.endswith("```"):
                judge_response = judge_response[:-3]
            
            # Parse and validate the judge response
            judge_data = json.loads(judge_response)
            
            # Ensure score is within valid range
            overall_score = max(0, min(100, judge_data.get("overall_score", 0)))
            judge_data["overall_score"] = overall_score
            judge_data["evaluation_success"] = True
            judge_data["evaluated_at"] = datetime.now().isoformat()
            
            # Generate concise quality feedback for display and database
            judge_data["quality_feedback"] = self.generate_quality_feedback(judge_data)
            
            return judge_data
            
        except Exception as e:
            return {
                "overall_score": 0,
                "evaluation_success": False,
                "error": f"Failed to evaluate oil contents: {str(e)}",
                "evaluated_at": datetime.now().isoformat()
            }
    
    def generate_quality_feedback(self, evaluation_result: Dict) -> str:
        """
        Generate concise quality feedback focused on math accuracy and realism
        
        Args:
            evaluation_result: Result from evaluate_report_contents()
            
        Returns:
            String with concise feedback explaining mathematical/realistic issues
        """
        if not evaluation_result.get("evaluation_success", False):
            return f"Evaluation failed: {evaluation_result.get('error', 'Unknown error')}"
        
        score = evaluation_result["overall_score"]
        analysis = evaluation_result.get("detailed_analysis", {})
        
        # Check for mathematical issues first
        numerical_feedback = analysis.get("numerical_accuracy_feedback", "")
        
        # Look for specific mathematical problems
        math_issues = []
        if "0.0 g" in numerical_feedback and "unrealistic" in numerical_feedback.lower():
            math_issues.append("0g LA unrealistic with identified oils")
        elif "xxx" in numerical_feedback.lower() or "placeholder" in numerical_feedback.lower():
            math_issues.append("Used placeholder numbers instead of realistic estimates")
        elif "calculation" in numerical_feedback.lower() and ("missing" in numerical_feedback.lower() or "incorrect" in numerical_feedback.lower()):
            math_issues.append("Mathematical calculations incorrect or missing")
        
        # Check for oil identification issues
        summary_feedback = analysis.get("summary_quality_feedback", "")
        tips_feedback = analysis.get("practical_tips_feedback", "")
        
        oil_issues = []
        if "not mention" in summary_feedback.lower() or "missing" in tips_feedback.lower():
            oil_issues.append("Missing specific oils from conversation")
        elif "generic" in tips_feedback.lower():
            oil_issues.append("Generic tips not specific to identified oils")
        
        # For perfect scores
        if score >= 95:
            return "Excellent: Realistic math, specific oils, targeted alternatives"
        
        # For high scores - focus on main math/oil issues
        if score >= 85:
            if math_issues:
                return f"Good quality, math issue: {math_issues[0]}"
            elif oil_issues:
                return f"Good quality, oil issue: {oil_issues[0]}"
            else:
                return "Good quality with minor issues"
        
        # For medium scores - highlight math problems first
        if score >= 60:
            issues = math_issues + oil_issues
            if issues:
                return f"Adequate quality, needs work: {issues[0]}"
            else:
                return "Adequate quality but needs improvement"
        
        # For low scores - critical math problems
        if score >= 40:
            if math_issues:
                return f"Poor quality: {math_issues[0]}"
            else:
                return "Poor quality: Major mathematical or oil identification issues"
        
        # For very low scores
        return "Very poor: Unrealistic numbers and/or missing oil analysis"

    def get_quality_insights(self, evaluation_result: Dict) -> str:
        """
        Generate a human-readable summary of the evaluation results
        
        Args:
            evaluation_result: Result from evaluate_report_contents()
            
        Returns:
            String with readable insights about the report quality
        """
        if not evaluation_result.get("evaluation_success", False):
            return f"❌ Evaluation failed: {evaluation_result.get('error', 'Unknown error')}"
        
        score = evaluation_result["overall_score"]
        quality = evaluation_result.get("quality_level", "UNKNOWN")
        
        # Get dimension breakdown
        dims = evaluation_result.get("dimension_scores", {})
        numerical = dims.get("numerical_accuracy", 0)
        summary = dims.get("summary_quality", 0) 
        tips = dims.get("practical_tips", 0)
        conclusion = dims.get("conclusion_quality", 0)
        
        insights = f"📊 Oil Contents Quality: {score}/100 ({quality})\n"
        insights += f"   • Numerical Accuracy: {numerical}/25\n"
        insights += f"   • Summary Quality: {summary}/25\n" 
        insights += f"   • Practical Tips: {tips}/25\n"
        insights += f"   • Conclusion Quality: {conclusion}/25\n"
        
        # Add strengths and weaknesses if available
        analysis = evaluation_result.get("detailed_analysis", {})
        strengths = analysis.get("overall_strengths", [])
        weaknesses = analysis.get("overall_weaknesses", [])
        
        if strengths:
            insights += f"   ✅ Strengths: {', '.join(strengths[:2])}\n"
        if weaknesses:
            insights += f"   ❌ Weaknesses: {', '.join(weaknesses[:2])}\n"
            
        return insights

if __name__ == "__main__":
    # Test the OilContentsJudge with sample data
    print("🧪 Testing Oil Contents Judge...")
    
    # Mock client and model for testing
    class MockClient:
        class ChatCompletions:
            class Completions:
                def create(self, **kwargs):
                    class MockChoice:
                        class MockMessage:
                            content = '{"overall_score": 85, "dimension_scores": {"numerical_accuracy": 20, "summary_quality": 22, "practical_tips": 21, "conclusion_quality": 22}, "bonus_points": 0, "penalty_points": 0, "quality_level": "GOOD"}'
                        message = MockMessage()
                    
                    class MockResponse:
                        choices = [MockChoice()]
                    
                    return MockResponse()
            completions = Completions()
        
        chat = ChatCompletions()
    
    # Test with sample data
    judge = OilContentsJudge(MockClient(), "test-model")
    
    sample_response = """
    {
        "message": "Your report will be ready below!",
        "emotion": "Supportive",
        "possibleSeedOilSources": ["granola", "mayonnaise"],
        "report": {
            "scores": {
                "estimatedTotalLa": "12.5 g",
                "dailyCalories": "2000 kcal",
                "laPercentageOfCalories": "5.6%",
                "score": "Yellow",
                "idealTarget": "1.5-3.0% of calories"
            },
            "summary": "Based on your breakfast and lunch, you have moderate seed oil intake primarily from granola and mayonnaise.",
            "practicalTips": ["Switch to homemade granola with olive oil", "Try avocado-based mayo instead"],
            "conclusion": "Small changes to these two items could significantly reduce your daily LA intake."
        },
        "reportIsComplete": true
    }
    """
    
    context = {
        "expected_seed_oil_sources": ["mayonnaise", "granola"],
        "estimated_hidden_seed_oils": 2,
        "difficulty_level": "moderate"
    }
    
    result = judge.evaluate_report_contents(sample_response, context)
    print(judge.get_quality_insights(result))
    print("✅ Oil Contents Judge test completed!")