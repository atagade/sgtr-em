import argparse
from utils.models_utils import add_temp_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add temporary model for SGTR-EM experiments")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--value", type=str, required=True, help="Value of the model")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID")
    parser.add_argument("--is_lora", action='store_true', help="Whether the model is a LoRA model")
    
    args = parser.parse_args()
    
    add_temp_model(
        enum_name=args.name,
        enum_value=args.value,
        model_id=args.model_id,
        is_lora=args.is_lora
    )

    print(f"Temporary model '{args.name}' with ID '{args.model_id}' added successfully.")