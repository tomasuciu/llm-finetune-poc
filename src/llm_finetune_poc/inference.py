import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to fine-tuned model checkpoint
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def generate_function_call(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_message: str,
    functions: Optional[List[Dict[str, Any]]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate function call from user message.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        system_prompt: System prompt with function definitions
        user_message: User's request
        functions: List of available functions (optional, for display)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    
    Returns:
        Generated response (function call or text)
    """
    # Construct conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback to simple concatenation
        input_text = f"{system_prompt}\n\n{user_message}\n\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def parse_function_call(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse function call from model response.
    
    Expects format like:
    {
        "name": "function_name",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2"
        }
    }
    
    Args:
        response: Model's response text
    
    Returns:
        Parsed function call dict or None if not a function call
    """
    try:
        # Look for JSON in response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            return None
        
        json_str = response[start_idx:end_idx]
        function_call = json.loads(json_str)
        
        # Validate structure
        if "name" in function_call and "arguments" in function_call:
            return function_call
        
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def execute_function(function_call: Dict[str, Any]) -> Any:
    """
    Simulate function execution (replace with actual function calls).
    
    Args:
        function_call: Parsed function call
    
    Returns:
        Function result
    """
    function_name = function_call["name"]
    arguments = function_call["arguments"]
    
    # Example: Simulate function execution
    print(f"\nExecuting function: {function_name}")
    print(f"Arguments: {json.dumps(arguments, indent=2)}")
    
    # Replace this with actual function calls
    if function_name == "get_weather":
        return {
            "location": arguments.get("location", "Unknown"),
            "temperature": 72,
            "conditions": "Sunny",
            "humidity": 45
        }
    elif function_name == "search_database":
        return {
            "results": [
                {"id": 1, "name": "Result 1"},
                {"id": 2, "name": "Result 2"}
            ],
            "count": 2
        }
    else:
        return {"error": f"Unknown function: {function_name}"}


def main():
    parser = argparse.ArgumentParser(
        description="Inference with fine-tuned function calling model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    # Example system prompt with function definitions
    system_prompt = """You are a helpful assistant with access to the following functions:

    1. get_weather(location: str) -> dict
       Get current weather information for a location.

    2. search_database(query: str, limit: int = 10) -> dict
       Search the database for relevant information.

    When you need to call a function, respond with a JSON object in this format:
    {
        "name": "function_name",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2"
        }
    }
    """
    
    if args.interactive:
        # Interactive mode
        print("\n" + "="*60)
        print("Function Calling Inference - Interactive Mode")
        print("="*60)
        print("Type 'quit' or 'exit' to exit\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                print("\nAssistant: Thinking...")
                response = generate_function_call(
                    model, tokenizer, system_prompt, user_input
                )
                
                function_call = parse_function_call(response)
                
                if function_call:
                    result = execute_function(function_call)
                    print(f"Result: {json.dumps(result, indent=2)}")
                else:
                    print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    else:
        # Example queries
        examples = [
            "What's the weather in Los Angeles?",
            "Search the postgres database on Nebius",
        ]
        
        print("\n" + "="*60)
        print("Function Calling Inference - Example Queries")
        print("="*60)
        
        for query in examples:
            print(f"\n{'─'*60}")
            print(f"🗣️  Query: {query}")
            print(f"{'─'*60}")
            
            response = generate_function_call(
                model, tokenizer, system_prompt, query
            )
            
            function_call = parse_function_call(response)
            
            if function_call:
                result = execute_function(function_call)
                print(f"Result: {json.dumps(result, indent=2)}")
            else:
                print(f"Response: {response}")


if __name__ == "__main__":
    main()
