#!/usr/bin/env python3
"""
Fetch Available Models from API Providers
Creates model choice files for student and judge LLMs
"""

import os
from openai import OpenAI
from groq import Groq

def fetch_openai_models():
    """Fetch available OpenAI models"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        models = client.models.list()
        
        # Filter for relevant models (skip embeddings, fine-tuned, etc.)
        relevant_models = []
        for model in models.data:
            model_id = model.id
            # Include GPT models, o1 models, but exclude embeddings, TTS, etc.
            if any(prefix in model_id for prefix in ['gpt-', 'o1-', 'o3-']):
                if not any(exclude in model_id for exclude in ['embed', 'tts', 'whisper', 'dall-e']):
                    relevant_models.append(f"openai:{model_id}")
        
        return sorted(relevant_models)
    
    except Exception as e:
        print(f"Error fetching OpenAI models: {e}")
        # Return known models as fallback
        return [
            "openai:gpt-4o",
            "openai:gpt-4o-mini", 
            "openai:gpt-4-turbo",
            "openai:gpt-3.5-turbo",
            "openai:o1-preview",
            "openai:o1-mini",
            "openai:o1-2024-12-17"
        ]

def fetch_groq_models():
    """Fetch available Groq models"""
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        models = client.models.list()
        
        groq_models = []
        for model in models.data:
            model_id = model.id
            groq_models.append(f"groq:{model_id}")
        
        return sorted(groq_models)
    
    except Exception as e:
        print(f"Error fetching Groq models: {e}")
        # Return known models as fallback
        return [
            "groq:llama-3.3-70b-versatile",
            "groq:llama-3.1-70b-versatile",
            "groq:llama-3.1-8b-instant",
            "groq:mixtral-8x7b-32768",
            "groq:gemma2-9b-it"
        ]

def categorize_models_for_student():
    """Categorize models suitable for student evaluation"""
    openai_models = fetch_openai_models()
    groq_models = fetch_groq_models()
    
    # Student models - typically smaller, faster models
    student_models = []
    
    # Add Groq models (good for student evaluation - fast and capable)
    student_models.extend(groq_models)
    
    # Add some OpenAI models suitable for students
    for model in openai_models:
        if any(pattern in model for pattern in ['gpt-3.5', 'gpt-4o-mini']):
            student_models.append(model)
    
    return student_models

def categorize_models_for_judge():
    """Categorize models suitable for judge evaluation"""
    openai_models = fetch_openai_models()
    
    # Judge models - typically stronger, more reasoning-capable models  
    judge_models = []
    
    # Add OpenAI models suitable for judging (o1, gpt-4o variants)
    for model in openai_models:
        if any(pattern in model for pattern in ['o1-', 'o3-', 'gpt-4o', 'gpt-4-turbo']):
            judge_models.append(model)
    
    return judge_models

def write_model_choices():
    """Write model choices to text files"""
    
    print("🔍 Fetching available models...")
    
    # Get student models
    student_models = categorize_models_for_student()
    with open('seed_oil_evaluation/llm_student_choices.txt', 'w') as f:
        f.write("# Student LLM Choices\n")
        f.write("# Format: provider:model_name\n")
        f.write("# Edit this file to add/remove models or change names\n\n")
        for model in student_models:
            f.write(f"{model}\n")
    
    print(f"✅ Wrote {len(student_models)} student models to llm_student_choices.txt")
    
    # Get judge models  
    judge_models = categorize_models_for_judge()
    with open('seed_oil_evaluation/llm_judge_choices.txt', 'w') as f:
        f.write("# Judge LLM Choices\n")
        f.write("# Format: provider:model_name\n") 
        f.write("# Edit this file to add/remove models or change names\n\n")
        for model in judge_models:
            f.write(f"{model}\n")
    
    print(f"✅ Wrote {len(judge_models)} judge models to llm_judge_choices.txt")
    
    # Also create a conversation generator choices file
    conversation_models = []
    openai_models = fetch_openai_models()
    for model in openai_models:
        if any(pattern in model for pattern in ['gpt-4o', 'gpt-4-turbo', 'o1-']):
            conversation_models.append(model)
    
    with open('seed_oil_evaluation/llm_conversation_generator_choices.txt', 'w') as f:
        f.write("# Conversation Generator LLM Choices\n")
        f.write("# Format: provider:model_name\n")
        f.write("# Edit this file to add/remove models or change names\n\n")
        for model in conversation_models:
            f.write(f"{model}\n")
    
    print(f"✅ Wrote {len(conversation_models)} conversation generator models to llm_conversation_generator_choices.txt")
    
    print(f"\n📝 Model files created in seed_oil_evaluation/")
    print("You can now edit these files to customize your model choices.")

if __name__ == "__main__":
    write_model_choices()