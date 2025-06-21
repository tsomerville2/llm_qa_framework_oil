#!/usr/bin/env python3
"""
Conversation Generator V2 for Seed Oil Sleuth Evaluation
Generates 3 clear categories of conversations with unambiguous metadata
"""

import os
import json
import uuid
import random
from datetime import datetime
from model_selector import select_model, get_model_client

class ConversationGeneratorV2:
    def __init__(self, interactive=False):
        self.conversations_dir = "seed_oil_evaluation/conversations"
        self.interactive = interactive
        
        # Load conversation weights
        self.weights = self.load_conversation_weights()
        
        # Select conversation generation model
        model_selection = select_model('conversation', interactive=interactive)
        if not model_selection:
            raise ValueError("No conversation generation model selected")
        
        provider, model = model_selection
        self.client, self.model = get_model_client(provider, model)
        self.provider = provider
        
        print(f"🤖 Using conversation model: {provider}:{model}")
        
    def load_conversation_weights(self):
        """Load conversation type weights from file"""
        weights_file = "seed_oil_evaluation/weights-positive-negative.txt"
        weights = {
            "positive_complete_conversations": 75,
            "negative_incomplete_conversations": 15, 
            "user_requested_reports": 10
        }
        
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line and not line.startswith('#'):
                        category, weight = line.split(':', 1)
                        weights[category.strip()] = int(weight.strip())
        
        return weights
    
    def determine_conversation_type(self):
        """Randomly select conversation type based on weights"""
        total_weight = sum(self.weights.values())
        rand_num = random.randint(1, total_weight)
        
        cumulative = 0
        for category, weight in self.weights.items():
            cumulative += weight
            if rand_num <= cumulative:
                return category
        
        return "positive_complete_conversations"  # fallback
    
    def generate_positive_conversation(self):
        """Generate POSITIVE example: Complete conversation covering full day"""
        
        prompt = """
Generate a COMPLETE conversation between a user and "The Seed Oil Sleuth" assistant. 

REQUIREMENTS FOR POSITIVE EXAMPLES:
1. Cover ALL meal periods: breakfast → lunch → dinner → snacks
2. Natural conversation flow that reaches a clear conclusion
3. Assistant naturally wraps up after getting full day's information
4. Include realistic seed oil sources in the foods mentioned
5. Conversation should feel naturally complete

The assistant should end with something like:
- "Thanks for sharing your full day with me..."
- "Now that we've covered your meals..."
- "Let me analyze your daily intake..."

Return ONLY a JSON object:
{
  "messages": [
    {"role": "assistant", "content": "Hi there! I'm The Seed Oil Sleuth..."},
    {"role": "user", "content": "..."},
    ...
  ]
}
"""
        
        return self.call_model(prompt)
    
    def generate_negative_conversation(self):
        """Generate NEGATIVE example: Truly incomplete conversation"""
        
        prompt = """
Generate an INCOMPLETE conversation between a user and "The Seed Oil Sleuth" assistant.

REQUIREMENTS FOR NEGATIVE EXAMPLES:
1. Cover ONLY 1-2 meal periods (like just breakfast + lunch, or just breakfast)
2. Stop abruptly in the middle of gathering information
3. User's final message should NOT indicate they're done
4. User might say something like "I need to run" or "Hold on, someone's at the door"
5. Clearly incomplete - assistant still needs more information

Examples of how to end:
- User: "Oh wait, I have to take this call"
- User: "Actually, I'm running late for work"  
- User: "Let me think about lunch and get back to you"

Return ONLY a JSON object:
{
  "messages": [
    {"role": "assistant", "content": "Hi there! I'm The Seed Oil Sleuth..."},
    {"role": "user", "content": "..."},
    ...
  ]
}
"""
        
        return self.call_model(prompt)
    
    def generate_user_requested_conversation(self):
        """Generate USER-REQUESTED example: User explicitly asks for report"""
        
        prompt = """
Generate a conversation where the USER explicitly asks for their seed oil report/analysis.

REQUIREMENTS FOR USER-REQUESTED EXAMPLES:
1. Can cover partial meals (breakfast + lunch) or full day
2. User specifically requests their report/analysis
3. User says something like:
   - "Can you give me my seed oil report now?"
   - "I'd like to see my LA analysis"
   - "Can you analyze what I've told you so far?"
   - "Show me my food report"
4. Assistant acknowledges the request

Examples of user request endings:
- User: "That's all I can remember. Can you give me my seed oil analysis?"
- User: "I need to run, but can you show me my report with what we've covered?"
- User: "Actually, can you analyze my breakfast and lunch for seed oils?"

Return ONLY a JSON object:
{
  "messages": [
    {"role": "assistant", "content": "Hi there! I'm The Seed Oil Sleuth..."},
    {"role": "user", "content": "..."},
    ...
  ]
}
"""
        
        return self.call_model(prompt)
    
    def call_model(self, prompt):
        """Call the model with the given prompt"""
        try:
            if self.provider == 'openai':
                if 'o1' in self.model or 'o3' in self.model:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8
                    )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8
                )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content)
            
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return None
    
    def create_metadata(self, conversation_type, messages):
        """Create unambiguous metadata based on conversation type"""
        
        # Extract food sources from conversation
        conversation_text = " ".join([msg["content"] for msg in messages])
        food_sources = self.extract_seed_oil_sources(conversation_text)
        
        if conversation_type == "positive_complete_conversations":
            return {
                "conversation_completeness": "complete",
                "should_trigger_report": True,
                "conversation_type": "positive_complete",
                "difficulty_level": random.choice(["simple", "moderate"]),
                "user_cooperation_level": random.choice(["helpful", "detailed"]),
                "expected_seed_oil_sources": food_sources[:3],  # limit to 3
                "estimated_hidden_seed_oils": len(food_sources[:3]),
                "conversation_flow_complexity": random.choice(["simple", "moderate"]),
                "total_messages": len(messages)
            }
        
        elif conversation_type == "negative_incomplete_conversations":
            return {
                "conversation_completeness": "partial", 
                "should_trigger_report": False,
                "conversation_type": "negative_incomplete",
                "difficulty_level": random.choice(["simple", "moderate", "complex"]),
                "user_cooperation_level": random.choice(["helpful", "evasive"]),
                "expected_seed_oil_sources": food_sources[:2],  # fewer sources
                "estimated_hidden_seed_oils": len(food_sources[:2]),
                "conversation_flow_complexity": random.choice(["simple", "moderate"]),
                "total_messages": len(messages)
            }
        
        else:  # user_requested_reports
            return {
                "conversation_completeness": "complete",
                "should_trigger_report": True, 
                "conversation_type": "user_requested",
                "difficulty_level": random.choice(["simple", "moderate"]),
                "user_cooperation_level": random.choice(["helpful", "detailed"]),
                "expected_seed_oil_sources": food_sources[:3],
                "estimated_hidden_seed_oils": len(food_sources[:3]),
                "conversation_flow_complexity": random.choice(["simple", "moderate"]),
                "total_messages": len(messages)
            }
    
    def extract_seed_oil_sources(self, text):
        """Extract likely seed oil sources from conversation text"""
        text = text.lower()
        possible_sources = []
        
        # Common seed oil source keywords
        source_keywords = {
            "mayonnaise": ["mayo", "mayonnaise"],
            "fries": ["fries", "french fries"],
            "chips": ["chips", "potato chips"],
            "margarine": ["margarine", "spread"],
            "cooking_oil": ["vegetable oil", "canola oil", "cooking oil"],
            "salad_dressing": ["dressing", "vinaigrette"],
            "packaged_snacks": ["crackers", "granola bars", "cookies"],
            "takeout": ["takeout", "restaurant", "fast food"],
            "frozen_meals": ["frozen", "microwave meal"],
            "coffee_creamer": ["creamer", "coffee cream"],
            "baked_goods": ["cake", "pastry", "muffin"]
        }
        
        for source, keywords in source_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    possible_sources.append(source)
                    break
        
        return list(set(possible_sources))  # remove duplicates
    
    def generate_conversations(self, num_conversations=10):
        """Generate specified number of conversations with proper distribution"""
        
        print(f"🎯 Generating {num_conversations} conversations with distribution:")
        print(f"  ✅ Positive Complete: {self.weights['positive_complete_conversations']}%")
        print(f"  ❌ Negative Incomplete: {self.weights['negative_incomplete_conversations']}%") 
        print(f"  🙋 User Requested: {self.weights['user_requested_reports']}%")
        
        # Calculate exact counts based on weights
        pos_count = round(num_conversations * self.weights['positive_complete_conversations'] / 100)
        neg_count = round(num_conversations * self.weights['negative_incomplete_conversations'] / 100)
        user_count = round(num_conversations * self.weights['user_requested_reports'] / 100)
        
        # Adjust if rounding doesn't add up to total
        total_assigned = pos_count + neg_count + user_count
        if total_assigned != num_conversations:
            pos_count += (num_conversations - total_assigned)
        
        print(f"📊 Exact counts: Positive={pos_count}, Negative={neg_count}, User={user_count}")
        
        # Create conversation type list
        conversation_types = (
            ["positive_complete_conversations"] * pos_count +
            ["negative_incomplete_conversations"] * neg_count +
            ["user_requested_reports"] * user_count
        )
        
        # Shuffle to randomize order
        random.shuffle(conversation_types)
        
        conversations = []
        
        for i in range(num_conversations):
            conversation_type = conversation_types[i]
            
            print(f"\n📝 Generating conversation {i+1}/{num_conversations} - Type: {conversation_type}")
            
            # Generate conversation based on type
            if conversation_type == "positive_complete_conversations":
                conversation = self.generate_positive_conversation()
            elif conversation_type == "negative_incomplete_conversations":
                conversation = self.generate_negative_conversation()
            else:
                conversation = self.generate_user_requested_conversation()
            
            if conversation and "messages" in conversation:
                # Create unambiguous metadata
                metadata = self.create_metadata(conversation_type, conversation["messages"])
                metadata["conversation_id"] = str(uuid.uuid4())
                metadata["generated_at"] = datetime.now().isoformat()
                metadata["evaluation_notes"] = f"Generated {conversation_type} - clear ground truth"
                
                conversation_data = {
                    "metadata": metadata,
                    "conversation": conversation
                }
                
                conversations.append(conversation_data)
                
                # Save individual conversation file
                filename = f"conv_{i+1:03d}_{metadata['conversation_id'][:8]}.json"
                filepath = os.path.join(self.conversations_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(conversation_data, f, indent=2)
                
                print(f"✅ Saved: {filename} ({conversation_type})")
            else:
                print(f"❌ Failed to generate conversation {i+1}")
        
        # Create index file
        self.create_index_file(conversations)
        
        return conversations
    
    def create_index_file(self, conversations):
        """Create index file with conversation metadata"""
        
        index_data = {
            "generated_at": datetime.now().isoformat(),
            "total_conversations": len(conversations),
            "generation_model": f"{self.provider}:{self.model}",
            "conversation_weights": self.weights,
            "conversations": []
        }
        
        for i, conv in enumerate(conversations, 1):
            index_data["conversations"].append({
                "file": f"conv_{i:03d}_{conv['metadata']['conversation_id'][:8]}.json",
                "conversation_id": conv["metadata"]["conversation_id"],
                "conversation_type": conv["metadata"]["conversation_type"],
                "completeness": conv["metadata"]["conversation_completeness"],
                "should_trigger_report": conv["metadata"]["should_trigger_report"],
                "total_messages": conv["metadata"]["total_messages"]
            })
        
        index_path = os.path.join(self.conversations_dir, "index.json")
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"\n📊 Generated {len(conversations)} conversations:")
        type_counts = {}
        for conv in conversations:
            conv_type = conv["metadata"]["conversation_type"]
            type_counts[conv_type] = type_counts.get(conv_type, 0) + 1
        
        for conv_type, count in type_counts.items():
            percentage = (count / len(conversations)) * 100
            print(f"  {conv_type}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    generator = ConversationGeneratorV2()
    conversations = generator.generate_conversations(10)
    print(f"\n✅ Successfully generated {len(conversations)} conversations with clear ground truth!")