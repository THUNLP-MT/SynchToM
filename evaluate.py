"""
Evaluation Script for Theory-of-Mind (ToM) Inference

This script evaluates AI model inferences using LLM-as-Judge with rubrics.
Rubrics are read directly from the inference file (instance['rubrics']).

Usage:
    python evaluate.py --input <inference_file.json> [options]

Environment Variables:
    OPENAI_API_KEY: API key for the judge model (default: 'your-api-key')
    OPENAI_BASE_URL: Base URL for the OpenAI API (default: 'http://localhost:8060/v1')
    JUDGE_MODEL: Model to use as judge (default: 'Qwen/Qwen3-32B')

Example:
    python evaluate.py --input inference_results.json --workers 4 --judge-model "gpt-4"
"""

from openai import OpenAI
import json
import datetime
import os
import logging
import time
import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logger (relative path to current directory)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment variables with defaults
api_key = os.environ.get('OPENAI_API_KEY', 'your-api-key')
base_url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:8060/v1')

# Judge model - can be overridden via command line
JUDGE_MODEL = os.environ.get('JUDGE_MODEL', "Qwen/Qwen3-32B")

# Number of instances to evaluate (set to None to evaluate all)
NUM_INSTANCES_TO_EVALUATE = None

# Timeout and retry configuration
MAX_RETRIES = 3  # Maximum number of retries on timeout

# Create evaluation directory (relative path)
eval_dir = Path("evaluation_results")
eval_dir.mkdir(exist_ok=True)


def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text, handling cases where it's wrapped in markdown or other text.
    
    Enhanced version with multiple fallback strategies for robust JSON extraction.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    json_pattern = r'```(?:json)?\s*(\{.*\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    raise ValueError(f"Could not extract valid JSON from response. Response preview: {text[:500]}...")


def evaluate_with_rubric(instance_id, ground_truth, inference_result, rubrics, client):
    """Use LLM-as-Judge to evaluate the inference based on rubrics."""
    
    # Count criteria for each dimension
    criteria_counts = {}
    for dimension, criteria in rubrics.items():
        criteria_counts[dimension] = len(criteria)
    
    # Calculate total max score
    total_max_score = sum(criteria_counts.values())
    
    # Format rubrics as JSON
    rubrics_text = json.dumps(rubrics, indent=2, ensure_ascii=False)
    
    judge_prompt = f"""# Task: OBJECTIVE AUDIT of Theory-of-Mind (ToM) Inference 

You are a **STRICT** expert evaluator. Your task is to verify if the AI's inference logically aligns with specific Rubric Criteria.

## EVALUATION RULES

1.  **Strict Scoring**: Go througth each criterion one by one. For each criterion, only score 1 if the inference is entirely correct, fully matches the rubric, and the inference contains no contradictions with the criterion. If the inference is only partially correct, incorrect in any part, contradicts the facts, score 0.

2.  **No Benefit of the Doubt**: Do not assume the model "meant" the right thing. Grade only what is explicitly written. You need to find the flaw until you prove it right

3.  **Chain of Thought Required**: You must explain your reasoning for EACH criterion before assigning scores.

4.  **Binary Scoring (0 or 1)**:
    - **Score 1 (Pass)**: The inference clearly covers the core concept of the criterion.
    - **Score 0 (Fail)**: The inference misses the concept, contradicts it, or is too vague to be useful.
---

## AI System's Inference (The Candidate)

**Inferred Latent Belief Explanation:**
{inference_result['latent_belief_explanation']}

**Inferred User Profile:**
{inference_result['user_profile_modeling']}

**Inferred Correct Resolution:**
{inference_result['correct_resolution']}

---

## Evaluation Rubrics

{rubrics_text}

---

## INSTRUCTIONS FOR RESPONSE

**Step 1: CRITICAL ANALYSIS (Text Format)**
For each dimension, strictly compare the Inference against the Rubric Criteria.
- List which specific keywords are missing.
- Point out where the logic deviates from the Ground Truth.
- Explicitly state "Criterion X is NOT met because..."

**Step 2: SCORING (JSON Format)**
After your analysis, output the JSON object.
Ensure the scores (0 or 1) strictly align with your analysis in Step 1.

```json
{{
  "latent_belief_explanation": {{
    "criterion_scores": {{
      "criterion_1": 0 or 1,
      "criterion_2": 0 or 1,
      ... (include all criteria for this dimension)
    }},
    "total_score": <sum of criterion scores>,
    "max_score": {criteria_counts.get('latent_belief_explanation', 5)},
    "feedback": "Brief explanation of which criteria were met and which were missed"
  }},
  "user_profile_modeling": {{
    "criterion_scores": {{
      "criterion_1": 0 or 1,
      "criterion_2": 0 or 1,
      ... (include all criteria for this dimension)
    }},
    "total_score": <sum of criterion scores>,
    "max_score": {criteria_counts.get('user_profile_modeling', 5)},
    "feedback": "Brief explanation of which criteria were met and which were missed"
  }},
  "correct_resolution": {{
    "criterion_scores": {{
      "criterion_1": 0 or 1,
      "criterion_2": 0 or 1,
      ... (include all criteria for this dimension)
    }},
    "total_score": <sum of criterion scores>,
    "max_score": {criteria_counts.get('correct_resolution', 5)},
    "feedback": "Brief explanation of which criteria were met and which were missed"
  }},
  "overall_summary": {{
    "total_score": <sum of all dimension scores>,
    "max_score": {total_max_score},
    "percentage": <(total_score / max_score) * 100>,
    "strengths": "What the AI did well",
    "weaknesses": "What the AI missed or got wrong",
    "overall_assessment": "Brief overall evaluation"
  }}
}}
```

**IMPORTANT:** Return the JSON object strictly.
"""
    
    logger.info(f"   🤖 Calling judge model ({JUDGE_MODEL})...")
    
    # Retry loop with timeout and JSON extraction error handling
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"   🔄 Attempt {attempt}/{MAX_RETRIES}")
            
            judge_response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0
            )
            
            response_text = judge_response.choices[0].message.content

            try:
                result = extract_json_from_text(response_text)
                
                # Verify and normalize scores
                for dimension in ['latent_belief_explanation', 'user_profile_modeling', 'correct_resolution']:
                    if dimension in result:
                        # Ensure total_score matches sum of criterion_scores
                        if 'criterion_scores' in result[dimension]:
                            calculated_total = sum(result[dimension]['criterion_scores'].values())
                            result[dimension]['total_score'] = calculated_total
                        
                        # Ensure max_score is set correctly
                        if dimension in criteria_counts:
                            result[dimension]['max_score'] = criteria_counts[dimension]
                
                # Recalculate overall summary
                if 'overall_summary' in result:
                    total = sum(result[dim]['total_score'] for dim in ['latent_belief_explanation', 'user_profile_modeling', 'correct_resolution'] if dim in result)
                    result['overall_summary']['total_score'] = total
                    result['overall_summary']['max_score'] = total_max_score
                    result['overall_summary']['percentage'] = (total / total_max_score * 100) if total_max_score > 0 else 0.0
                
                # Add raw response to result
                result['raw_response'] = response_text
                
                logger.info("   ✓ Evaluation completed")
                return result
            except ValueError as json_error:
                logger.warning(f"   ⚠️  JSON extraction failed on attempt {attempt}/{MAX_RETRIES}: {str(json_error)[:100]}")
                if attempt < MAX_RETRIES:
                    logger.debug(f"   Problematic response: {response_text[:300]}")
                    wait_time = 2 ** attempt
                    logger.info(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"   ✗ Max retries reached - all JSON extraction attempts failed")
                    raise
            
        except ValueError:
            if attempt >= MAX_RETRIES:
                raise
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                logger.warning(f"   ⏱️  Timeout on attempt {attempt}/{MAX_RETRIES}: {error_msg}")
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt
                    logger.info(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"   ✗ Max retries reached for evaluation")
                    raise
            else:
                logger.error(f"   ✗ Non-timeout error: {error_msg}")
                raise


def evaluate_instance(instance, client):
    """Evaluate a single instance (with pre-computed inference).
    
    Rubrics are read directly from the instance data.
    """
    instance_id = instance['instance_id']
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating {instance_id}")
    logger.info(f"{'='*70}")
    
    try:
        # Extract data from instance
        ground_truth = instance['ground_truth']
        inference_result = instance['inference']
        
        # Get rubrics directly from instance
        rubrics = instance.get('rubrics')
        
        if not rubrics:
            logger.error(f"   ✗ No rubrics found in instance {instance_id}")
            raise ValueError(f"No rubrics found in instance {instance_id}. Please ensure rubrics are included in the inference file.")
        
        logger.info(f"   ✓ Using rubrics from instance")
        
        # Run evaluation
        logger.info("📊 Running evaluation with judge...")
        judge_result = evaluate_with_rubric(
            instance_id=instance_id,
            ground_truth=ground_truth,
            inference_result=inference_result,
            rubrics=rubrics,
            client=client
        )
        
        # Combine results
        result = {
            "instance_id": instance_id,
            "domain": instance.get('domain', 'Unknown'),
            "evaluated_at": datetime.datetime.now().isoformat(),
            "ground_truth": ground_truth,
            "inference": inference_result,
            "rubrics": rubrics,
            "evaluation": judge_result
        }
        
        logger.info(f"✓ {instance_id} evaluation completed")
        logger.info(f"   Score: {judge_result['overall_summary']['total_score']}/{judge_result['overall_summary']['max_score']} ({judge_result['overall_summary']['percentage']:.1f}%)")
        
        return result, None
        
    except Exception as e:
        logger.error(f"✗ Error evaluating {instance_id}: {str(e)}")
        return None, str(e)




def main():
    """Main evaluation loop."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate inference results using LLM-as-Judge with rubrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluation_rubric.py --input inference/inference_culture-benchmark_claude_step0.json
  python evaluation_rubric.py -i inference/inference_results_20260113_093402.json --workers 4
        """
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the inference results JSON file (generated by inference.py)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of concurrent workers for parallel processing (default: 4)'
    )
    parser.add_argument(
        '--judge-model', '-j',
        type=str,
        default=None,
        help='Judge model to use for evaluation (overrides JUDGE_MODEL env var)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (overrides OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='OpenAI API base URL (overrides OPENAI_BASE_URL env var)'
    )
    
    args = parser.parse_args()
    input_file = args.input
    num_workers = args.workers
    
    # Override configurations from command line if provided
    global JUDGE_MODEL, api_key, base_url
    if args.judge_model:
        JUDGE_MODEL = args.judge_model
    if args.api_key:
        api_key = args.api_key
    if args.base_url:
        base_url = args.base_url
    # Extract filename without extension (handle multiple dots in filename)
    input_name = Path(input_file).stem  # Gets filename without extension, handles multiple dots
    # Create output filename based on current timestamp
    if input_name.startswith('inference_'):
        input_name = input_name[10:]
    output_file = eval_dir / f"evaluation_{input_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Print pipeline information
    logger.info("="*70)
    logger.info("Rubric-Based Latent Belief Evaluation Pipeline (Multi-threaded)")
    logger.info("="*70)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Judge model: {JUDGE_MODEL}")
    logger.info(f"Concurrent workers: {num_workers}")
    logger.info("="*70)
    
    # Load inference results
    logger.info(f"\n📖 Loading inference results from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    logger.info(f"   ✓ Loaded {len(dataset)} instances with inference")
    
    # Select subset if NUM_INSTANCES_TO_EVALUATE is set
    if NUM_INSTANCES_TO_EVALUATE is not None:
        dataset = dataset[:NUM_INSTANCES_TO_EVALUATE]
        logger.info(f"   ℹ️  Evaluating only first {len(dataset)} instances")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Initialize results file with empty list
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f)
    logger.info(f"   ✓ Initialized output file: {output_file}")
    
    file_lock = threading.Lock()
    
    success_count = 0
    failed_count = 0
    failed_instances = []
    completed_count = 0
    
    def process_and_save(instance):
        """Process instance and save result in thread-safe manner."""
        nonlocal success_count, failed_count, completed_count
        
        result, error = evaluate_instance(instance, client)
        
        # Thread-safe file writing
        with file_lock:
            nonlocal completed_count
            completed_count += 1
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Progress: {completed_count}/{len(dataset)} completed")
            logger.info(f"{'='*70}")
            
            if result:
                # Append to results file immediately
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                results.append(result)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                logger.info(f"💾 Saved result for {instance['instance_id']}")
                return {"status": "success", "instance_id": instance['instance_id']}
            else:
                failed_count += 1
                failed_instances.append({
                    "instance_id": instance['instance_id'],
                    "error": error
                })
                return {"status": "failed", "instance_id": instance['instance_id'], "error": error}
    
    # Use ThreadPoolExecutor for concurrent processing
    logger.info(f"\n🚀 Starting concurrent evaluation with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_instance = {executor.submit(process_and_save, instance): instance for instance in dataset}
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_instance):
            instance = future_to_instance[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error(f"✗ Instance {instance['instance_id']} generated an exception: {exc}")
                with file_lock:
                    failed_count += 1
                    failed_instances.append({
                        "instance_id": instance['instance_id'],
                        "error": str(exc)
                    })
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"✓ Successful: {success_count}/{len(dataset)}")
    logger.info(f"✗ Failed: {failed_count}/{len(dataset)}")
    
    if failed_instances:
        logger.warning(f"\nFailed instances:")
        for fail in failed_instances:
            logger.warning(f"  - {fail['instance_id']}: {fail['error']}")
    
    logger.info(f"\n📊 Results saved to: {output_file}")
    
    # Calculate average scores (only for successfully evaluated instances)
    if success_count > 0:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Calculate normalized scores (percentage) for each dimension
        latent_belief_scores = []
        user_profile_scores = []
        resolution_scores = []
        overall_scores = []
        
        for r in results:
            eval_data = r['evaluation']
            
            # Calculate percentage for each dimension
            if 'latent_belief_explanation' in eval_data:
                lb = eval_data['latent_belief_explanation']
                latent_belief_scores.append(lb['total_score'] / lb['max_score'] if lb['max_score'] > 0 else 0)
            
            if 'user_profile_modeling' in eval_data:
                up = eval_data['user_profile_modeling']
                user_profile_scores.append(up['total_score'] / up['max_score'] if up['max_score'] > 0 else 0)
            
            if 'correct_resolution' in eval_data:
                cr = eval_data['correct_resolution']
                resolution_scores.append(cr['total_score'] / cr['max_score'] if cr['max_score'] > 0 else 0)
            
            if 'overall_summary' in eval_data:
                overall_scores.append(eval_data['overall_summary']['percentage'])
        
        # Calculate averages
        latent_belief = sum(latent_belief_scores) / len(latent_belief_scores) if latent_belief_scores else 0
        user_profile = sum(user_profile_scores) / len(user_profile_scores) if user_profile_scores else 0
        resolution = sum(resolution_scores) / len(resolution_scores) if resolution_scores else 0
        avg_percentage = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        avg_score = avg_percentage / 100  # normalized to 0-1
        
        # Save summary to results.json (relative path)
        results_summary_file = Path("evaluation_summary.json")
        total_instances = success_count + failed_count
        summary_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input_file": str(input_file),
            "output_file": str(output_file),
            "judge_model": JUDGE_MODEL,
            "evaluation_method": "rubric_based",
            "total_instances": total_instances,
            "successful_evaluations": success_count,
            "failed_evaluations": failed_count,
            "scores": {
                "latent_belief_explanation": {
                    "average_score": round(latent_belief, 4),
                    "percentage": round(latent_belief * 100, 2)
                },
                "user_profile_modeling": {
                    "average_score": round(user_profile, 4),
                    "percentage": round(user_profile * 100, 2)
                },
                "correct_resolution": {
                    "average_score": round(resolution, 4),
                    "percentage": round(resolution * 100, 2)
                },
                "overall": {
                    "average_score": round(avg_score, 4),
                    "percentage": round(avg_percentage, 2)
                }
            }
        }
        
        try:
            if results_summary_file.exists() and results_summary_file.stat().st_size > 0:
                with open(results_summary_file, 'r', encoding='utf-8') as f:
                    all_summaries = json.load(f)
            else:
                all_summaries = []
        except (json.JSONDecodeError, FileNotFoundError):
            all_summaries = []
        
        all_summaries.append(summary_entry)
        with open(results_summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📝 Summary saved to: {results_summary_file}")
    else:
        logger.warning("\nNo instances were successfully evaluated - cannot calculate average scores")


if __name__ == "__main__":
    main()
