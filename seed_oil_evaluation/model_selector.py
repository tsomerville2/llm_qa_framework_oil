#!/usr/bin/env python3
"""
Model Selector for Seed Oil Sleuth QA Pipeline
Handles model selection from configuration files
"""

import os
from typing import List, Tuple, Optional

try:
    import questionary
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False

def parse_model_file(filepath: str) -> List[Tuple[str, str]]:
    """Parse model choices file and return list of (provider, model) tuples"""
    
    if not os.path.exists(filepath):
        return []
    
    models = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse provider:model format
            if ':' in line:
                provider, model = line.split(':', 1)
                models.append((provider.strip(), model.strip()))
    
    return models

def get_available_models(model_type: str) -> List[Tuple[str, str]]:
    """Get available models for a specific type"""
    
    file_mapping = {
        'conversation': 'seed_oil_evaluation/llm_convo_choices.txt',
        'student': 'seed_oil_evaluation/llm_student_choices.txt', 
        'judge': 'seed_oil_evaluation/llm_judge_choices.txt'
    }
    
    filepath = file_mapping.get(model_type)
    if not filepath:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return parse_model_file(filepath)

def select_model(model_type: str, interactive: bool = False) -> Optional[Tuple[str, str]]:
    """Select a model for the given type"""
    
    available_models = get_available_models(model_type)
    
    if not available_models:
        print(f"❌ No models available for {model_type}. Check llm_{model_type}_choices.txt")
        return None
    
    # If only one model, use it directly
    if len(available_models) == 1:
        provider, model = available_models[0]
        print(f"🤖 Using {model_type} model: {provider}:{model}")
        return (provider, model)
    
    # Multiple models available
    if not interactive or not INTERACTIVE_AVAILABLE:
        # Non-interactive: use first model
        provider, model = available_models[0]
        print(f"🤖 Using default {model_type} model: {provider}:{model}")
        return (provider, model)
    
    # Interactive: ask user to choose
    choices = [f"{provider}:{model}" for provider, model in available_models]
    
    try:
        selected = questionary.select(
            f"Select {model_type} model:",
            choices=choices,
            instruction="(Use arrow keys)"
        ).ask()
        
        if selected:
            provider, model = selected.split(':', 1)
            return (provider.strip(), model.strip())
        else:
            return None
            
    except (KeyboardInterrupt, EOFError):
        print(f"\n❌ {model_type.title()} model selection cancelled")
        return None

def get_model_client(provider: str, model: str):
    """Get appropriate API client for the model"""

    if provider == 'openai':
        """
        Return an OpenAI client *wrapper* that transparently supports both chat
        and completions-only models.

        Rationale
        ---------
        OpenAI `o1-*` and `o3-*` models (e.g. `o3-pro`) are classic completion
        models – they are **not** compatible with the `/v1/chat/completions`
        endpoint.  The rest of our codebase, however, is written against the
        Chat Completions API (`client.chat.completions.create(...)`).  To avoid
        touching every call-site we wrap the regular `OpenAI` client in a thin
        adapter that:

        • Intercepts `client.chat.completions.create` calls.
        • Detects whether the requested model is chat-capable.
        • For classic completion models it reroutes the request to
          `client.completions.create`, converts the messages array to a prompt,
          and finally returns a response object that exposes the *same* shape
          (i.e. `response.choices[0].message.content`).

        Chat-capable models are forwarded unchanged.
        """

        from openai import OpenAI

        class _ResponseShim:  # pylint: disable=too-few-public-methods
            """Wrap a completion response so that it mimics a chat response."""

            def __init__(self, text: str):
                message_obj = type("_Msg", (), {"content": text})()
                choice_obj = type("_Choice", (), {"message": message_obj})()
                self.choices = [choice_obj]

        class _CompletionsProxy:  # pylint: disable=too-few-public-methods
            def __init__(self, base_client, default_model):
                self._base = base_client
                self._default_model = default_model

            @staticmethod
            def _is_chat_model(model_id: str) -> bool:
                """Rudimentary check whether a model ID supports chat API."""
                chat_prefixes = (
                    "gpt-",  # all modern GPT chat models
                    "ft:",    # fine-tuned chat models keep this prefix
                )
                return model_id.startswith(chat_prefixes)

            @staticmethod
            def _messages_to_prompt(messages):
                """Flatten chat messages -> single prompt string."""
                lines = []
                role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
                for msg in messages or []:
                    role = role_map.get(msg.get("role", "user"), msg.get("role"))
                    lines.append(f"{role}: {msg.get('content', '')}")
                # The assistant is expected to continue from the conversation.
                return "\n".join(lines) + "\nAssistant:"

            def create(self, *args, **kwargs):  # pylint: disable=missing-docstring
                # Preserve a copy for potential chat fallback.
                orig_kwargs = kwargs.copy()

                # The callers always pass the model explicitly but we still keep
                # a default fallback.
                model_id = kwargs.get("model", self._default_model)

                # Fast-path: if we *know* it's a chat model, bypass extra work.
                if self._is_chat_model(model_id):
                    return self._base.chat.completions.create(*args, **kwargs)

                # ------- Attempt completion call -------

                completion_kwargs = kwargs.copy()

                # Convert messages -> prompt if needed (mutates completion_kwargs only)
                if "messages" in completion_kwargs and "prompt" not in completion_kwargs:
                    completion_kwargs["prompt"] = self._messages_to_prompt(
                        completion_kwargs.pop("messages")
                    )

                # Translate `max_completion_tokens` -> `max_tokens`
                if (
                    "max_completion_tokens" in completion_kwargs
                    and "max_tokens" not in completion_kwargs
                ):
                    completion_kwargs["max_tokens"] = completion_kwargs.pop(
                        "max_completion_tokens"
                    )

                # Remove chat-specific parameters that completions endpoint rejects.
                completion_kwargs.pop("response_format", None)
                completion_kwargs.pop("stream", None)

                try:
                    response = self._base.completions.create(*args, **completion_kwargs)
                    return _ResponseShim(response.choices[0].text)
                except Exception as exc:  # Broad catch – we inspect the content.
                    msg = str(exc).lower()
                    is_chat_error = (
                        "chat model" in msg and "v1/completions" in msg
                    ) or (
                        "did you mean to use v1/chat/completions" in msg
                    )

                    if not is_chat_error:
                        raise  # Some other failure – bubble up.

                    # Fallback: treat as chat model and retry via chat endpoint with
                    # *original* kwargs (contains messages etc.).
                    return self._base.chat.completions.create(*args, **orig_kwargs)

        class _ChatProxy:  # pylint: disable=too-few-public-methods
            def __init__(self, base_client, default_model):
                self.completions = _CompletionsProxy(base_client, default_model)

        class _OpenAIWrapper:  # pylint: disable=too-few-public-methods
            """Expose the real client whilst swapping the chat handler."""

            def __init__(self, base_client, default_model):
                self._base = base_client
                self.chat = _ChatProxy(base_client, default_model)

            def __getattr__(self, item):
                return getattr(self._base, item)

        raw_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        client_wrapper = _OpenAIWrapper(raw_client, model)
        return client_wrapper, model
    
    elif provider == 'groq':
        from groq import Groq  
        return Groq(api_key=os.getenv('GROQ_API_KEY')), model
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def validate_model_files():
    """Validate that model choice files exist and have content"""
    
    required_files = [
        'seed_oil_evaluation/llm_convo_choices.txt',
        'seed_oil_evaluation/llm_student_choices.txt',
        'seed_oil_evaluation/llm_judge_choices.txt'
    ]
    
    missing_files = []
    empty_files = []
    
    for filepath in required_files:
        if not os.path.exists(filepath):
            missing_files.append(filepath)
        else:
            models = parse_model_file(filepath)
            if not models:
                empty_files.append(filepath)
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        return False
    
    if empty_files:
        print(f"⚠️ Empty model files: {', '.join(empty_files)}")
        return False
    
    return True

if __name__ == "__main__":
    # Test the model selector
    print("🧪 Testing Model Selector")
    print("=" * 30)
    
    if not validate_model_files():
        print("❌ Model file validation failed")
        exit(1)
    
    for model_type in ['conversation', 'student', 'judge']:
        print(f"\n📋 Available {model_type} models:")
        models = get_available_models(model_type)
        for i, (provider, model) in enumerate(models, 1):
            print(f"  {i}. {provider}:{model}")
        
        # Test non-interactive selection
        selected = select_model(model_type, interactive=False)
        if selected:
            provider, model = selected
            print(f"✅ Selected: {provider}:{model}")
        else:
            print(f"❌ No model selected for {model_type}")
    
    print(f"\n✅ Model selector test complete")