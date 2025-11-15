
from tqdm import tqdm
import os
import logging
from utils.file_utils import save_json, load_jsonl, save_jsonl, append_to_jsonl
from utils.prompts import build_task1_prompt
from utils.metrics import extract_last_boxed_answer, calculate_accuracy

# Configure logging
logger = logging.getLogger(__name__)


def calculate_field_accuracy(results):
    """
    Calculate accuracy scores for each field and overall average.
    
    Args:
        results: List of dictionaries containing 'answer', 'prediction', and 'field' keys
    
    Returns:
        Dictionary containing accuracy scores per field and overall average
    """
    # Initialize data containers
    overall_data = {"answers": [], "predictions": []}
    field_data = {}
    
    # Organize data by field
    for item in results:
        field = item["field"]
        sub_field = item["sub_field"]
        
        # Initialize field entry if not exists
        if field not in field_data:
            field_data[field] = {"answers": [], "predictions": [], "sub_field": {}}
        if sub_field not in field_data[field]["sub_field"]:
            field_data[field][sub_field] = {"answers": [], "predictions": []}
        
        # Add data to field-specific and overall containers
        field_data[field][sub_field]["answers"].append(item["answer"])
        field_data[field][sub_field]["predictions"].append(item["prediction"])
        field_data[field]["answers"].append(item["answer"])
        field_data[field]["predictions"].append(item["prediction"])
        overall_data["answers"].append(item["answer"])
        overall_data["predictions"].append(item["prediction"])

    # Calculate accuracy for each field and overall
    accuracy_scores = {}
    
    for field_name, field_results in field_data.items():
        field_accuracy = calculate_accuracy(field_results["predictions"], field_results["answers"])
        logger.info(f"{field_name}: {field_accuracy:.4f}")
        accuracy_scores[field_name] = {"overall": field_accuracy}
        for sub_field_name, sub_field_results in field_results["sub_field"].items():
            sub_field_accuracy = calculate_accuracy(sub_field_results["predictions"], sub_field_results["answers"])
            accuracy_scores[field_name][sub_field_name] = sub_field_accuracy
    
    # Calculate overall accuracy
    overall_accuracy = calculate_accuracy(overall_data["predictions"], overall_data["answers"])
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    accuracy_scores["overall"] = overall_accuracy
    
    return accuracy_scores


def run_evaluation_task1(model, results_dir, args):
    """
    Run evaluation on the datasets using the specified model and parameters.
    
    Args:
        model: The model instance to use for evaluation
        args: Command line arguments containing evaluation parameters
    """
    # Generate output file path
    task_name = f"task1.jsonl"
    results_filepath = os.path.join(results_dir, task_name)
    
    # Load existing results or initialize empty list
    if os.path.exists(results_filepath):
        evaluation_results = load_jsonl(results_filepath)
        logger.info(f"Loaded {len(evaluation_results)} existing results from {results_filepath}")
    else:
        evaluation_results = []
        logger.info("Starting new evaluation run")
    
    # Load evaluation datasets
    evaluation_data = load_jsonl(os.path.join(args.data_dir, "datasets", task_name))
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
            prompt = build_task1_prompt(sample["question"], sample["field"])
            model_response = model.chat(prompt, max_new_tokens=args.max_length)
            
            # Extract prediction and store results
            sample["response"] = model_response
            sample["prediction"] = extract_last_boxed_answer(model_response)
            
            append_to_jsonl(sample, results_filepath)
            evaluation_results.append(sample)
        
        logger.info(f"Completed iteration {iteration + 1}/{args.test_time}")
    
    # Calculate and save accuracy scores
    logger.info("Calculating accuracy scores...")
    accuracy_scores = calculate_field_accuracy(evaluation_results)
    
    return accuracy_scores



