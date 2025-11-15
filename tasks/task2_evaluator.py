from tqdm import tqdm
import os
import logging
from utils.file_utils import save_json, load_jsonl, save_jsonl, append_to_jsonl
from utils.prompts import build_task2_prompt
from utils.metrics import extract_last_boxed_answer, calculate_accuracy

# Configure logging
logger = logging.getLogger(__name__)


def calculate_overall_accuracy(results):
    """
    Calculate overall accuracy score from evaluation results.
    
    Args:
        results: List of dictionaries containing 'answer' and 'prediction' keys
    
    Returns:
        Dictionary containing overall accuracy score
    """
    # Initialize data containers
    overall_data = {"answers": [], "predictions": []}
    
    # Collect all answers and predictions
    for item in results:
        overall_data["answers"].append(item["answer"])
        overall_data["predictions"].append(item["prediction"])
    
    # Calculate overall accuracy
    overall_accuracy = calculate_accuracy(overall_data["predictions"], overall_data["answers"])
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    
    accuracy_scores = {"overall": overall_accuracy}
    return accuracy_scores


def run_evaluation_task2(model, results_dir, args):
    """
    Run evaluation on the dataset using the specified model and parameters.
    
    Args:
        model: The model instance to use for evaluation
        args: Command line arguments containing evaluation parameters
    """
    # Generate output file path
    task_name = f"task2.jsonl"
    results_filepath = os.path.join(results_dir, task_name)
    
    # Load existing results or initialize empty list
    if os.path.exists(results_filepath):
        evaluation_results = load_jsonl(results_filepath)
        logger.info(f"Loaded {len(evaluation_results)} existing results from {results_filepath}")
    else:
        evaluation_results = []
        logger.info("Starting new evaluation run")
    
    # Load and preprocess evaluation dataset
    raw_data = load_jsonl(os.path.join(args.data_dir, "datasets", task_name))
    evaluation_data = []
    
    for item in raw_data:
        for qas in item["qas"]:
            evaluation_data.append({
                "question": qas["question"], 
                "answer": qas["answer"], 
                "material": item["material"]
            })
    
    total_samples = len(evaluation_data)
    logger.info(f"Loaded {total_samples} evaluation samples")
    
    # Calculate how many samples need to be processed
    total_iterations = args.test_time
    samples_to_process = total_iterations * total_samples - len(evaluation_results)
    
    if samples_to_process <= 0:
        logger.info("All samples have already been processed. Skipping evaluation.")
    else:
        logger.info(f"Processing {samples_to_process} samples across {total_iterations} iterations")
    
    # Run evaluation for specified number of test iterations
    for iteration in range(args.test_time):
        logger.info(f"Starting evaluation iteration {iteration + 1}/{args.test_time}")
        
        for sample_idx, sample in enumerate(tqdm(evaluation_data, desc=f"Iteration {iteration + 1}")):
            # Skip if this sample has already been processed in current iteration
            result_index = sample_idx + iteration * total_samples
            if result_index < len(evaluation_results):
                continue
            
            # Generate prompt and get model response
            prompt = build_task2_prompt(sample["question"], sample["material"])
            model_response = model.chat(prompt, max_new_tokens=args.max_length)
            
            # Extract prediction and store results
            sample["response"] = model_response
            sample["prediction"] = extract_last_boxed_answer(model_response)
            
            append_to_jsonl(sample, results_filepath)
            evaluation_results.append(sample)
        
        logger.info(f"Completed iteration {iteration + 1}/{args.test_time}")
    
    # Calculate and save accuracy scores
    logger.info("Calculating accuracy scores...")
    accuracy_scores = calculate_overall_accuracy(evaluation_results)
    
    return accuracy_scores


