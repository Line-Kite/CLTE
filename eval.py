import os
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.file_utils import save_json
from tasks.task1_evaluator import run_evaluation_task1
from tasks.task2_evaluator import run_evaluation_task2
from tasks.task3_evaluator import run_evaluation_task3


# Configure logging
logger = logging.getLogger(__name__)


class Qwen3Model:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            device_map="auto"
        )
    
    def chat(self, prompt, system_prompt = None, max_new_tokens=32768, enable_thinking=False):
        messages = []
        sys_msg = system_prompt 
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return content
    
    def release(self):
        """Release model resources including GPU memory"""
        if hasattr(self, 'model'):
            # Delete the model and clean up GPU memory
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU cache to free memory
        
        if hasattr(self, 'tokenizer'):
            del self.tokenizer  # Delete tokenizer to free CPU memory



def main():
    """Main function to set up and run evaluation."""
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path of the model to evaluate")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model to evaluate")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum token length for generation")
    parser.add_argument("--test_time", type=int, default=1, help="Number of test iterations")
    
    # Logging configuration
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path).lower()
    
    # Initialize model
    logger.info(f"Initializing model: {args.model_name}")
    model = Qwen3Model(args.model_path)
    
    results_dir = os.path.join(args.save_dir, f"{args.model_name}_{args.test_time}_{args.max_length}")
    os.makedirs(results_dir, exist_ok=True)

    # Run evaluation
    results_task1 = run_evaluation_task1(model, results_dir, args)
    results_task2 = run_evaluation_task2(model, results_dir, args)
    results_task3 = run_evaluation_task3(model, results_dir, args)
    logger.info("Evaluation completed successfully")

    # Save accuracy results
    scores_filepath = os.path.join(results_dir, "score.json")
    
    save_json({"task1": results_task1, "task2": results_task2, "task3": results_task3}, scores_filepath)
    logger.info(f"Accuracy scores saved to {scores_filepath}")


if __name__ == "__main__":
    main()