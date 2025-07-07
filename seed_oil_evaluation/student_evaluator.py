#!/usr/bin/env python3
"""
Student Evaluator for Seed Oil Sleuth Evaluation
Uses Groq Llama-4 Maverick to evaluate conversations with the Seed Oil Sleuth system prompt
"""

import os
import json
import uuid
import time
from datetime import datetime
from model_selector import select_model, get_model_client

class StudentEvaluator:
    def __init__(self, prompt_version="v1.0", interactive=False, selected_model=None):
        self.conversations_dir = "seed_oil_evaluation/conversations"
        self.responses_dir = "seed_oil_evaluation/student_responses"
        self.prompt_version = prompt_version
        self.interactive = interactive
        self.system_prompt = self.load_prompt(prompt_version)
        
        # Use pre-selected model or select new one
        if selected_model:
            provider, model = selected_model
            model_selection = selected_model
        else:
            model_selection = select_model('student', interactive=interactive)
            if not model_selection:
                raise ValueError("No student model selected")
            provider, model = model_selection
        
        self.client, self.model = get_model_client(provider, model)
        self.provider = provider
        self.student_model_name = f"{provider}:{model}"
        
        print(f"🤖 Using student model: {self.student_model_name}")
    
    def load_prompt(self, version):
        """Load system prompt from versioned file"""
        prompt_file = f"seed_oil_evaluation/prompts/{version}.json"
        
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt version {version} not found at {prompt_file}")
        
        with open(prompt_file, 'r') as f:
            prompt_data = json.load(f)
        
        return json.dumps(prompt_data["prompt"])
    
    def load_conversations(self):
        """Load all conversations from the conversations directory"""
        
        # Load index to get conversation list
        with open(f"{self.conversations_dir}/index.json", 'r') as f:
            index = json.load(f)
        
        conversations = []
        for conv_info in index["conversations"]:
            filepath = f"{self.conversations_dir}/{conv_info['file']}"
            with open(filepath, 'r') as f:
                conversation_data = json.load(f)
                conversations.append(conversation_data)
        
        return conversations
    
    def evaluate_conversation(self, conversation_data):
        """Evaluate a single conversation with the student model"""
        
        # Prepare messages for the student model
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add the conversation messages
        for msg in conversation_data["conversation"]["messages"]:
            messages.append(msg)
        
        # Some models (especially groq compound models) require last message to be from user
        # If conversation ends with assistant message, add a user prompt
        if messages[-1]["role"] == "assistant":
            messages.append({
                "role": "user", 
                "content": "Please provide your response now based on our conversation."
            })
        
        try:
            # Start timing
            start_time = time.time()
            
            # Handle different providers
            if self.provider == 'openai':
                if 'o1' in self.model or 'o3' in self.model:
                    # o1/o3 models require max_completion_tokens and support structured outputs
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=25000,
                        response_format={"type": "json_object"}
                    )
                elif 'gpt-4.1' in self.model:
                    # GPT-4.1 models don't support json_object mode yet, use prompt-based JSON
                    # Add explicit JSON instruction to the last message
                    messages[-1]["content"] += "\n\nPlease respond with valid JSON only, no other text."
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=1,
                        max_tokens=2181
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=1,
                        max_tokens=2181,
                        response_format={"type": "json_object"}
                    )
            else:
                # Groq or other providers
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1,
                    max_tokens=2181,
                    top_p=1,
                    stream=False,
                    response_format={"type": "json_object"},
                    seed=42,
                )
            
            # End timing
            end_time = time.time()
            response_time = end_time - start_time
            
            student_response = response.choices[0].message.content
            return student_response, response_time
            
        except Exception as e:
            print(f"Error evaluating conversation {conversation_data['metadata']['conversation_id']}: {e}")
            return None, None
    
    def evaluate_all_conversations(self):
        """Evaluate all conversations and save results"""
        
        conversations = self.load_conversations()
        evaluation_results = []
        
        print("🤖 STUDENT RESPONSES")
        print(f"{'#':<2} {'ID':<8} {'Type':<8} {'Message':<7} {'Emotion':<7} {'Sources':<7} {'Report':<6} {'Complete':<8} {'Scores':<6} {'Summary':<7} {'Tips':<4} {'Conclusion':<10} {'Time(s)':<8}")
        
        for i, conversation_data in enumerate(conversations, 1):
            conv_id = conversation_data["metadata"]["conversation_id"]
            
            result = self.evaluate_conversation(conversation_data)
            
            if result and result[0]:
                student_response, response_time = result
                # Parse student response for readability
                try:
                    parsed_response = json.loads(student_response)
                    student_response_readable = json.dumps(parsed_response, indent=2)
                except json.JSONDecodeError:
                    student_response_readable = student_response
                
                evaluation_result = {
                    "conversation_id": conv_id,
                    "evaluated_at": datetime.now().isoformat(),
                    "student_model": self.student_model_name,
                    "prompt_version": self.prompt_version,
                    "original_metadata": conversation_data["metadata"],
                    "student_response": student_response,
                    "student_response_readable": student_response_readable,
                    "response_time_seconds": round(response_time, 3),
                    "evaluation_success": True
                }
                
                # Save individual result
                filename = f"student_resp_{i:03d}_{conv_id[:8]}.json"
                filepath = f"{self.responses_dir}/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(evaluation_result, f, indent=2)
                
                # Analyze and display student response details
                response_analysis = self.analyze_student_response(student_response)
                conv_type = conversation_data["metadata"].get("conversation_type", "unknown")
                self.print_student_analysis(i, conv_id, conv_type, response_analysis, filename, response_time)
                
                evaluation_results.append(evaluation_result)
                
            else:
                evaluation_result = {
                    "conversation_id": conv_id,
                    "evaluated_at": datetime.now().isoformat(),
                    "student_model": self.student_model_name,
                    "prompt_version": self.prompt_version,
                    "original_metadata": conversation_data["metadata"],
                    "student_response": None,
                    "response_time_seconds": None,
                    "evaluation_success": False,
                    "error": "Failed to generate response"
                }
                evaluation_results.append(evaluation_result)
                print(f"Failed to evaluate conversation {i}")
        
        # Save master results
        results_summary = {
            "evaluated_at": datetime.now().isoformat(),
            "total_conversations": len(conversations),
            "successful_evaluations": len([r for r in evaluation_results if r["evaluation_success"]]),
            "student_model": self.student_model_name,
            "prompt_version": self.prompt_version,
            "results": evaluation_results
        }
        
        with open(f"{self.responses_dir}/evaluation_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        return evaluation_results
    
    def analyze_student_response(self, response_json: str) -> dict:
        """Analyze student response JSON for component presence"""
        try:
            response = json.loads(response_json)
        except json.JSONDecodeError:
            return {"valid_json": False, "error": "Invalid JSON"}
        
        analysis = {
            "valid_json": True,
            "message": bool(response.get("message", "").strip()),
            "emotion": bool(response.get("emotion", "").strip()),
            "possible_seed_oil_sources": bool(response.get("possibleSeedOilSources")),
            "seed_oil_count": len(response.get("possibleSeedOilSources", [])),
            "report_present": bool(response.get("report")),
            "report_is_complete": response.get("reportIsComplete", False),
            "report_has_scores": bool((response.get("report") or {}).get("scores")),
            "report_has_summary": bool((response.get("report") or {}).get("summary", "") and str((response.get("report") or {}).get("summary", "")).strip()),
            "report_has_tips": bool((response.get("report") or {}).get("practicalTips")),
            "report_has_conclusion": bool((response.get("report") or {}).get("conclusion", "") and str((response.get("report") or {}).get("conclusion", "")).strip()),
        }
        
        # Count filled report fields
        report_fields_filled = sum([
            analysis["report_has_scores"],
            analysis["report_has_summary"], 
            analysis["report_has_tips"],
            analysis["report_has_conclusion"]
        ])
        analysis["report_fields_filled"] = report_fields_filled
        
        return analysis
    
    def print_student_analysis(self, conv_num: int, conv_id: str, conv_type: str, analysis: dict, filename: str, response_time: float = None):
        """Print clean analysis of student response"""
        
        if not analysis["valid_json"]:
            time_str = f"{response_time:.1f}s" if response_time else "N/A"
            print(f"{conv_num:2d} {conv_id[:8]} {conv_type:<8s} ❌ INVALID JSON {time_str:>8s}")
            return
        
        # Format single clean row with manual alignment
        msg = "X" if analysis["message"] else " "
        emo = "X" if analysis["emotion"] else " "
        src = str(analysis["seed_oil_count"])
        prs = "X" if analysis["report_present"] else " "
        cmp = "X" if analysis["report_is_complete"] else " "
        scr = "X" if analysis["report_has_scores"] else " "
        sum_field = "X" if analysis["report_has_summary"] else " "
        tip = "X" if analysis["report_has_tips"] else " "
        con = "X" if analysis["report_has_conclusion"] else " "
        
        # Shorten conversation type for display
        type_short = conv_type.replace("positive_complete", "POS").replace("negative_incomplete", "NEG").replace("user_requested", "USER")
        
        # Format timing
        time_str = f"{response_time:.1f}s" if response_time else "N/A"
        
        print(f"{conv_num:2d} {conv_id[:8]} {type_short:<8s} {msg:<7s} {emo:<7s} {src:<7s} {prs:<6s} {cmp:<8s} {scr:<6s} {sum_field:<7s} {tip:<4s} {con:<10s} {time_str:>8s}")

if __name__ == "__main__":
    evaluator = StudentEvaluator()
    results = evaluator.evaluate_all_conversations()
    successful = len([r for r in results if r["evaluation_success"]])
    print(f"\\nEvaluated {successful}/{len(results)} conversations successfully!")