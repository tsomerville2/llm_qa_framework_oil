#!/usr/bin/env python3
"""
Conversation Generator for Seed Oil Sleuth Evaluation
Uses OpenAI o3 to generate realistic diet conversations with metadata
"""

import os
import json
import uuid
from datetime import datetime
from model_selector import select_model, get_model_client

class ConversationGenerator:
    def __init__(self, interactive=False):
        self.conversations_dir = "seed_oil_evaluation/conversations"
        self.interactive = interactive
        
        # Select conversation generation model
        model_selection = select_model('conversation', interactive=interactive)
        if not model_selection:
            raise ValueError("No conversation generation model selected")
        
        provider, model = model_selection
        self.client, self.model = get_model_client(provider, model)
        self.provider = provider
        
        print(f"🤖 Using conversation model: {provider}:{model}")
        
    def generate_conversation(self, scenario_params):
        """Generate a single conversation with the given scenario parameters"""
        
        generation_prompt = f"""
Generate a realistic conversation between a user and "The Seed Oil Sleuth" assistant about their daily diet. 

Scenario Parameters:
- Difficulty: {scenario_params['difficulty']}
- User cooperation: {scenario_params['user_cooperation']}
- Conversation completeness: {scenario_params['completeness']}
- Expected seed oil sources: {scenario_params['expected_sources']}

Requirements:
1. Create a natural back-and-forth conversation
2. User should reveal their meals gradually (breakfast, lunch, dinner, snacks)
3. IMPORTANT: The user must mention foods that contain ALL of these specific seed oil sources: {scenario_params['expected_sources']}
4. The conversation should match the specified completeness level
5. User cooperation level affects how detailed/evasive their responses are
6. If completeness is "complete", the user should give clear signals they're done sharing (e.g., "that's all I ate", "that's everything")

Return ONLY a JSON object with this structure:
{{
  "messages": [
    {{"role": "assistant", "content": "Hi there! I'm The Seed Oil Sleuth..."}},
    {{"role": "user", "content": "User response"}},
    {{"role": "assistant", "content": "Assistant response"}},
    ...
  ]
}}

Make it realistic - include specific foods, brands, cooking methods, and portions where appropriate.
"""

        try:
            # Handle different providers
            if self.provider == 'openai':
                # Check if it's an o1 model (no temperature support)
                if 'o1' in self.model or 'o3' in self.model:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": generation_prompt}]
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": generation_prompt}],
                        temperature=0.8
                    )
            else:
                # Groq or other providers
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": generation_prompt}],
                    temperature=0.8
                )
            
            
            response_content = response.choices[0].message.content.strip()
            print(f"Raw response: {response_content[:200]}...")  # Debug output
            
            # Clean up response if it has markdown code blocks
            if response_content.startswith("```json"):
                response_content = response_content[7:]  # Remove ```json
            if response_content.endswith("```"):
                response_content = response_content[:-3]  # Remove ```
            
            conversation_data = json.loads(response_content)
            return conversation_data
            
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return None
    
    def create_metadata(self, scenario_params, conversation):
        """Create metadata for the conversation to help with scoring"""
        
        # Analyze conversation to extract actual seed oil sources mentioned
        conversation_text = " ".join([msg["content"] for msg in conversation["messages"]])
        actual_seed_oil_sources = self.extract_actual_seed_oil_sources(conversation["messages"])
        
        # Determine if user explicitly completed conversation
        user_completion_signals = self.detect_user_completion_signals(conversation["messages"])
        
        # Check if conversation covered full timeline
        timeline_coverage = self.analyze_timeline_coverage(conversation["messages"])
        
        return {
            "conversation_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "difficulty_level": scenario_params["difficulty"],
            "user_cooperation_level": scenario_params["user_cooperation"],
            "conversation_completeness": scenario_params["completeness"],
            "expected_seed_oil_sources": scenario_params["expected_sources"],
            "actual_seed_oil_sources": actual_seed_oil_sources,
            "estimated_hidden_seed_oils": len(actual_seed_oil_sources),
            "should_trigger_report": scenario_params["completeness"] == "complete",
            "conversation_type": "positive_complete" if scenario_params["completeness"] == "complete" else "negative_incomplete",
            "conversation_flow_complexity": scenario_params["difficulty"],
            "total_messages": len(conversation["messages"]),
            "user_completion_signals": user_completion_signals,
            "timeline_coverage": timeline_coverage,
            "user_explicitly_finished": user_completion_signals["completion_detected"],
            "evaluation_notes": "Generated conversation - metadata includes completion analysis"
        }
    
    def detect_user_completion_signals(self, messages):
        """Detect signals that user has finished sharing their food information"""
        completion_phrases = [
            "that's all",
            "that's everything", 
            "that's everything i ate",
            "that's everything i had",
            "nothing else",
            "i'm done",
            "that's it",
            "all the food",
            "give me the report",
            "la report",
            "seed oil report",
            "that's all i ate",
            "that's all i had",
            "no more food",
            "finished eating",
            "everything i ate today",
            "everything i ate and drank today",
            "all i ate today",
            "all i ate and drank today",
            "that's my full day",
            "that's my whole day"
        ]
        
        explicit_completion = False
        implicit_completion = False
        last_user_message = ""
        
        # Look at user messages for completion signals
        user_messages = [msg["content"].lower() for msg in messages if msg["role"] == "user"]
        
        if user_messages:
            last_user_message = user_messages[-1]
            
            # Check for explicit completion phrases
            for phrase in completion_phrases:
                if phrase in last_user_message:
                    explicit_completion = True
                    break
            
            # Check for implicit signals (short responses at end)
            if len(user_messages) >= 3:
                last_few = user_messages[-2:]
                if all(len(msg.split()) <= 5 for msg in last_few):
                    implicit_completion = True
        
        return {
            "explicit_completion": explicit_completion,
            "implicit_completion": implicit_completion,
            "last_user_message": last_user_message,
            "completion_detected": explicit_completion or implicit_completion
        }
    
    def analyze_timeline_coverage(self, messages):
        """Analyze if conversation covered full daily timeline"""
        conversation_text = " ".join([msg["content"].lower() for msg in messages])
        
        timeline_markers = {
            "breakfast": ["breakfast", "morning", "cereal", "eggs", "toast", "coffee"],
            "lunch": ["lunch", "sandwich", "salad", "noon", "midday"],
            "dinner": ["dinner", "evening", "supper", "stir fry", "pasta", "chicken"],
            "snacks": ["snack", "chips", "nuts", "fruit", "cookie", "bar"]
        }
        
        coverage = {}
        for meal_type, keywords in timeline_markers.items():
            coverage[meal_type] = any(keyword in conversation_text for keyword in keywords)
        
        covered_meals = sum(coverage.values())
        covered_full_day = covered_meals >= 3  # At least breakfast, lunch, dinner
        
        return {
            "meal_coverage": coverage,
            "covered_meals_count": covered_meals,
            "covered_full_day": covered_full_day,
            "missing_meals": [meal for meal, covered in coverage.items() if not covered]
        }
    
    def extract_actual_seed_oil_sources(self, messages):
        """Extract actual seed oil sources mentioned in the conversation"""
        conversation_text = " ".join([msg["content"].lower() for msg in messages])
        
        # Common seed oil source foods and their indicators
        seed_oil_indicators = {
            "fries": ["fries", "french fries", "fried potatoes", "canola oil", "vegetable oil"],
            "mayonnaise": ["mayo", "mayonnaise", "hellmann", "miracle whip"],
            "chips": ["chips", "lay's", "lays", "potato chips", "corn chips", "cheetos", "doritos"],
            "crackers": ["crackers", "goldfish", "cheez-its", "ritz"],
            "fried chicken": ["fried chicken", "kfc", "popeyes", "fried"],
            "salad dressing": ["salad dressing", "vinaigrette", "ranch", "italian dressing"],
            "packaged snacks": ["granola bar", "energy bar", "protein bar", "packaged"],
            "condiments": ["ketchup", "mustard", "bbq sauce", "hot sauce"],
            "takeout": ["takeout", "delivery", "fast food", "restaurant"],
            "frozen meals": ["frozen dinner", "lean cuisine", "hot pocket"],
            "baked goods": ["muffin", "cookie", "cake", "pastry", "croissant"],
            "margarine": ["margarine", "butter substitute", "spread"],
            "nuts": ["mixed nuts", "trail mix", "roasted nuts"],
            "soup": ["canned soup", "campbell's", "progresso"],
            "coffee creamer": ["creamer", "coffee mate", "half and half"],
            "granola": ["granola", "cereal", "breakfast bar"],
            "pasta sauce": ["pasta sauce", "marinara", "alfredo"],
            "protein bars": ["protein bar", "quest bar", "cliff bar"],
            "meal replacement": ["protein shake", "meal replacement", "ensure"]
        }
        
        found_sources = []
        for source_type, indicators in seed_oil_indicators.items():
            for indicator in indicators:
                if indicator in conversation_text:
                    if source_type not in found_sources:
                        found_sources.append(source_type)
                    break
        
        return found_sources
    
    def generate_scenario_set(self):
        """Generate 10 diverse conversation scenarios"""
        
        scenarios = [
            {
                "difficulty": "simple",
                "user_cooperation": "helpful",
                "completeness": "complete",
                "expected_sources": ["fries", "mayonnaise"]
            },
            {
                "difficulty": "moderate", 
                "user_cooperation": "detailed",
                "completeness": "complete",
                "expected_sources": ["salad dressing", "crackers", "fried chicken"]
            },
            {
                "difficulty": "complex",
                "user_cooperation": "evasive", 
                "completeness": "partial",
                "expected_sources": ["hidden restaurant oils", "packaged snacks", "condiments"]
            },
            {
                "difficulty": "simple",
                "user_cooperation": "helpful",
                "completeness": "partial",
                "expected_sources": ["chips", "margarine"]
            },
            {
                "difficulty": "moderate",
                "user_cooperation": "evasive",
                "completeness": "complete", 
                "expected_sources": ["frozen meals", "takeout", "baked goods"]
            },
            {
                "difficulty": "complex",
                "user_cooperation": "detailed",
                "completeness": "complete",
                "expected_sources": ["multiple restaurant meals", "processed meats", "energy bars"]
            },
            {
                "difficulty": "simple",
                "user_cooperation": "detailed",
                "completeness": "partial",
                "expected_sources": ["sandwich spread", "cereal"]
            },
            {
                "difficulty": "moderate",
                "user_cooperation": "helpful",
                "completeness": "complete",
                "expected_sources": ["coffee creamer", "granola", "soup"]
            },
            {
                "difficulty": "complex",
                "user_cooperation": "evasive",
                "completeness": "partial",
                "expected_sources": ["fast food", "vending machine snacks", "meal replacement"]
            },
            {
                "difficulty": "moderate",
                "user_cooperation": "detailed", 
                "completeness": "complete",
                "expected_sources": ["pasta sauce", "nuts", "protein bars"]
            }
        ]
        
        return scenarios
    
    def generate_all_conversations(self):
        """Generate all 10 conversations and save them"""
        
        scenarios = self.generate_scenario_set()
        generated_conversations = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"Generating conversation {i}/10...")
            
            conversation = self.generate_conversation(scenario)
            if conversation:
                metadata = self.create_metadata(scenario, conversation)
                
                conversation_data = {
                    "metadata": metadata,
                    "conversation": conversation
                }
                
                # Save individual conversation file
                filename = f"conv_{i:03d}_{metadata['conversation_id'][:8]}.json"
                filepath = os.path.join(self.conversations_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(conversation_data, f, indent=2)
                
                generated_conversations.append(conversation_data)
                print(f"Saved: {filename}")
            else:
                print(f"Failed to generate conversation {i}")
        
        # Save master index
        index_data = {
            "generated_at": datetime.now().isoformat(),
            "total_conversations": len(generated_conversations),
            "conversations": [
                {
                    "file": f"conv_{i:03d}_{conv['metadata']['conversation_id'][:8]}.json",
                    "id": conv["metadata"]["conversation_id"],
                    "difficulty": conv["metadata"]["difficulty_level"],
                    "completeness": conv["metadata"]["conversation_completeness"]
                }
                for i, conv in enumerate(generated_conversations, 1)
            ]
        }
        
        with open(f"{self.conversations_dir}/index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
        
        return generated_conversations
    
    def generate_additional_conversations(self, num_new, existing_count):
        """Generate additional conversations and append to existing dataset"""
        
        scenarios = self.generate_scenario_set()
        # Cycle through scenarios if we need more than 10
        extended_scenarios = (scenarios * ((num_new // len(scenarios)) + 1))[:num_new]
        
        generated_conversations = []
        
        for i, scenario in enumerate(extended_scenarios, 1):
            conv_num = existing_count + i
            print(f"Generating conversation {conv_num} ({i}/{num_new})...")
            
            conversation = self.generate_conversation(scenario)
            if conversation:
                metadata = self.create_metadata(scenario, conversation)
                
                conversation_data = {
                    "metadata": metadata,
                    "conversation": conversation
                }
                
                # Save individual conversation file
                filename = f"conv_{conv_num:03d}_{metadata['conversation_id'][:8]}.json"
                filepath = os.path.join(self.conversations_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(conversation_data, f, indent=2)
                
                generated_conversations.append(conversation_data)
                print(f"Saved: {filename}")
            else:
                print(f"Failed to generate conversation {conv_num}")
        
        # Update master index
        index_path = f"{self.conversations_dir}/index.json"
        with open(index_path, 'r') as f:
            existing_index = json.load(f)
        
        # Add new conversations to index
        for i, conv in enumerate(generated_conversations):
            conv_num = existing_count + i + 1
            existing_index["conversations"].append({
                "file": f"conv_{conv_num:03d}_{conv['metadata']['conversation_id'][:8]}.json",
                "id": conv["metadata"]["conversation_id"],
                "difficulty": conv["metadata"]["difficulty_level"],
                "completeness": conv["metadata"]["conversation_completeness"]
            })
        
        existing_index["total_conversations"] = len(existing_index["conversations"])
        existing_index["last_updated"] = datetime.now().isoformat()
        
        with open(index_path, 'w') as f:
            json.dump(existing_index, f, indent=2)
        
        return generated_conversations

if __name__ == "__main__":
    generator = ConversationGenerator()
    conversations = generator.generate_all_conversations()
    print(f"\nGenerated {len(conversations)} conversations successfully!")