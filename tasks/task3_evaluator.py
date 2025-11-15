from tqdm import tqdm
import os
import logging
import copy
import json
from utils.file_utils import save_json, load_jsonl, save_jsonl, append_to_jsonl
from utils.prompts import build_teacher_prompt, build_guided_student_prompt
from utils.metrics import extract_last_boxed_answer, extract_last_knowledge_object, calculate_accuracy
from models.student import StudentModel

# Configure logging
logger = logging.getLogger(__name__)


def generate_knowledge_for_task3(model, results_dir, args):
    # Generate output file path
    knowledge_filename = "task3_knowledge.jsonl"
    knowledge_filepath = os.path.join(results_dir, knowledge_filename)
    
    # Load existing results or initialize empty list
    if os.path.exists(knowledge_filepath):
        knowledge_results = load_jsonl(knowledge_filepath)
        logger.info(f"Loaded {len(knowledge_results)} existing knowledge from {knowledge_filepath}")
    else:
        knowledge_results = []
        logger.info("Starting knowledge generation for task 3")
    
    # Load evaluation dataset
    evaluation_data = load_jsonl(os.path.join(args.data_dir, "datasets", "task3.jsonl"))
    total_samples = len(evaluation_data)
    logger.info(f"Loaded {total_samples} evaluation samples")
    
    # Calculate how many samples need to be processed
    samples_to_process = total_samples - len(knowledge_results)
    
    if samples_to_process <= 0:
        logger.info("All samples have already been processed. Skipping evaluation.")
        return knowledge_results
    
    logger.info(f"Processing {samples_to_process} new samples")
    
    # Process each sample
    for sample_idx, sample in enumerate(tqdm(evaluation_data, desc="Processing samples")):
        # Skip if this sample has already been processed
        if sample_idx < len(knowledge_results):
            continue
        
        try:
            # Generate prompt and get model response
            system_prompt, user_prompt = build_teacher_prompt(sample["guideline"], sample["material"])
            model_response = model.chat(
                user_prompt, 
                system_prompt=system_prompt, 
                max_new_tokens=args.max_length
            )
            
            # Extract knowledge and store results
            sample["response"] = model_response
            sample["knowledge"] = extract_last_knowledge_object(model_response)
            
            # Save results
            append_to_jsonl(sample, knowledge_filepath)
            knowledge_results.append(sample)
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            # Continue with next sample even if current one fails
            continue

    logger.info(f"Completed knowledge generation for task 3. Processed {len(knowledge_results)} samples")
    
    return knowledge_results


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

def run_evaluation_student(model, evaluation_data, results_dir, args):
    """
    Run evaluation on the dataset using the specified model and parameters.
    
    Args:
        model: The model instance to use for evaluation
        args: Command line arguments containing evaluation parameters
    
    Returns:
        Dictionary containing accuracy scores
    """
    # Generate output file path
    results_filename = f"task3_student_{model.model_name}.jsonl"
    results_filepath = os.path.join(results_dir, results_filename)
    
    # Load existing results or initialize empty list
    if os.path.exists(results_filepath):
        evaluation_results = load_jsonl(results_filepath)
        logger.info(f"Loaded {len(evaluation_results)} existing results from {results_filepath}")
    else:
        evaluation_results = []
        logger.info("Starting new evaluation run")
    
    total_samples = len(evaluation_data)
    
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
            system_prompt, question_prompt = build_guided_student_prompt(
                sample["question"], 
                sample["material"], 
                sample["knowledge"] if isinstance(sample["knowledge"], str) 
                else json.dumps(sample["knowledge"], indent=4, ensure_ascii=False)
            )
            
            model_response = model.chat(question_prompt, system_prompt)
            
            # Extract prediction and store results
            sample["response"] = model_response
            sample["prediction"] = extract_last_boxed_answer(model_response)
            
            append_to_jsonl(sample, results_filepath)
            evaluation_results.append(sample)
        
        logger.info(f"Completed iteration {iteration + 1}/{args.test_time}")
    
    # Calculate and save accuracy scores
    logger.info("Calculating accuracy scores...")
    accuracy_scores = calculate_overall_accuracy(evaluation_results)
    accuracy_scores["model_name"] = model.model_name
    
    return accuracy_scores


def run_evaluation_task3(model, results_dir, args):
    # Load and preprocess evaluation dataset
    evaluation_data = generate_knowledge_for_task3(model, results_dir, args)
    total_samples = len(evaluation_data)
    logger.info(f"Loaded {total_samples} evaluation samples")

    model.release()

    accuracy_scores = {}
    for model_name in ["qwen-1_8b", "qwen-7b", "qwen-14b", "yi-6b", "internlm2-7b"]:
        logger.info(f"Test {model_name} student")
        student_model = StudentModel(model_name, os.path.join(args.data_dir, "models", model_name))
        accuracy_score = run_evaluation_student(student_model, copy.deepcopy(evaluation_data), results_dir, args)
        accuracy_scores[model_name] = accuracy_score["overall"]
        student_model.release()
    values = list(accuracy_scores.values())
    mean = sum(values) / len(values)
    logger.info(f"Overall accuracy: {mean:.4f}")
    return {"overall": mean, "student_scores": accuracy_scores}
        
    