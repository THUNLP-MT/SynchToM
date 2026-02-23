"""
Inference Script for Theory-of-Mind (ToM) Dataset

This script runs inference on instances to predict latent beliefs, user profiles,
and correct resolutions based on observations, instructions, and trajectories.

Usage:
    python inference.py --input <dataset.json> --model <model_name> --steps <num_steps> [options]

Environment Variables:
    OPENAI_API_KEY: API key for the inference model (default: 'EMPTY')
    OPENAI_BASE_URL: Base URL for the OpenAI API (default: 'http://localhost:8080/v1')
    TRAJECTORIES_DIR: Base directory for trajectory files (default: './trajectories')
    IMAGES_DIR: Directory containing images (default: './images')

Example:
    python inference.py --input dataset.json --model gpt-4 --steps 5
    python inference.py -i dataset.json -m qwen3-32b -s 0  # No trajectory
"""

from openai import OpenAI
import json
import datetime
import os
import logging
import time
import re
import argparse
import base64
from pathlib import Path

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables with defaults
api_key = os.environ.get('OPENAI_API_KEY', 'EMPTY')
base_url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:8080/v1')

# Number of instances to process (set to None to process all)
NUM_INSTANCES_TO_PROCESS = None

# Timeout and retry configuration
TIMEOUT_SECONDS = 180
MAX_RETRIES = 3

# Create inference directory (relative path)
inference_dir = Path("inference_results")
inference_dir.mkdir(exist_ok=True)

# Trajectories and images directories (configurable via env vars)
TRAJECTORIES_DIR = Path(os.environ.get('TRAJECTORIES_DIR', './trajectories'))
IMAGES_DIR = Path(os.environ.get('IMAGES_DIR', './images'))


def encode_image_to_base64(image_id: str) -> str:
    """Encode image file to base64 string.
    
    Args:
        image_id: Image identifier (e.g., "11-1")
    
    Returns:
        Base64 encoded string of the image
    """
    # Try direct path first
    image_path = IMAGES_DIR / f"{image_id}.png"
    if image_path.exists():
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Try subdirectories
    for subdir in ["image1", "malay", "new_data", "0502-double_ambiguity"]:
        image_path = IMAGES_DIR / subdir / f"{image_id}.png"
        if image_path.exists():
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    
    raise FileNotFoundError(f"Image not found for image_id: {image_id}")


def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text, handling cases where it's wrapped in markdown or other text.
    
    Enhanced version with multiple fallback strategies for robust JSON extraction.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    think_pattern = r'<think>.*?</think>'
    text_no_think = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    try:
        return json.loads(text_no_think.strip())
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

    raise ValueError(f"Could not extract valid JSON from response after trying all strategies. Response preview: {text[:500]}...")


def load_trajectory(trajectory_path, step_num=None):
    """Load trajectory from file and filter out 'thought' field to prevent information leakage.
    
    Args:
        trajectory_path: Path to the trajectory file (relative to TRAJECTORIES_DIR or absolute)
        step_num: Number of steps to keep (None = keep all steps)
    
    Returns:
        list: Filtered trajectory, or None if file doesn't exist
    """
    # Try as relative path first (from TRAJECTORIES_DIR)
    full_path = TRAJECTORIES_DIR / trajectory_path
    
    # If not found, try as absolute path
    if not full_path.exists():
        full_path = Path(trajectory_path)
    
    # Check if file exists
    if not full_path.exists():
        logger.warning(f"   ⚠️  Trajectory file not found: {trajectory_path}")
        return None
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter out 'thought' field from each turn
        trajectory = data.get('trajectory', data)
        filtered_trajectory = []
        for turn in trajectory:
            filtered_turn = {
                "turn": turn["turn"],
                "action": turn["action"],
                "observation": turn.get("observation", "")  
            }
            filtered_trajectory.append(filtered_turn)
        
        # Keep only first step_num steps if specified
        if step_num is not None and step_num > 0:
            filtered_trajectory = filtered_trajectory[:step_num]
            logger.info(f"   📏 Keeping first {step_num} steps (total available: {len(data.get('trajectory', data))} steps)")
        if step_num == 0:
            filtered_trajectory = []
        return filtered_trajectory
    except Exception as e:
        logger.error(f"   ✗ Error loading trajectory {trajectory_path}: {e}")
        return None


def infer_three_dimensions(observation, explicit_instruction, trajectory, client, inference_model, image_id=None):
    """Use LLM to infer the three dimensions.
    
    Args:
        observation: User's observation
        explicit_instruction: User's explicit instruction
        trajectory: User's behavioral trajectory
        client: OpenAI client
        inference_model: Model name for inference
        image_id: Optional image identifier for visual context
    """
    
    inference_prompt = f"""# Task: 
You need to infer three key aspects about a user based on their observation, explicit instruction, and behavioral trajectory.

## Given Information

### User's Observation:
{observation}

### User's Explicit Instruction:
{explicit_instruction}

### User's Behavioral Trajectory:
{json.dumps(trajectory, indent=2, ensure_ascii=False)}

---

## Your Task

Based on the above information, infer and provide:

### 1. Latent Belief Explanation
What does the user believe is the root cause of the problem? Explain their mental model and why they have this belief.

### 2. User Profile Modeling
What underlying preference, bias, experience level, or worldview does this user have that led them to form this belief? Why did they develop this conception?

### 3. Correct Resolution
What is the actual root cause of the problem, and what is the correct solution? Explain what the user should do to truly fix the issue.

## Requirements
- Be specific and concrete in your answers
- The latent belief should explain the user's observable actions
- The user profile should explain WHY they have this latent belief
- The correct resolution should address the true root cause

## Output Format

Return a JSON object with exactly these fields:
{{
  "latent_belief_explanation": "The user's belief about the problem and why they think their approach works",
  "user_profile_modeling": "The user's background, preferences, and biases that led to this belief",
  "correct_resolution": "The root cause and the correct solution to fix it"
}}
"""
    
    logger.info(f"   🤖 Calling inference model ({inference_model})...")
    # Prepare messages with optional image
    messages = [{"role": "user", "content": inference_prompt}]
    
    # Retry loop with timeout and JSON extraction error handling
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"   🔄 Attempt {attempt}/{MAX_RETRIES}")
            
            inference_response = client.chat.completions.create(
                model=inference_model,
                messages=messages,
                temperature=0.0,
                timeout=TIMEOUT_SECONDS
            )
            
            response_text = inference_response.choices[0].message.content
            print(response_text)
            
            # Extract JSON from response (handles markdown code blocks and incomplete JSON)
            try:
                result = extract_json_from_text(response_text)
                logger.info("   ✓ Inference completed")
                return result
            except ValueError as json_error:
                # JSON extraction failed
                logger.warning(f"   ⚠️  JSON extraction failed on attempt {attempt}/{MAX_RETRIES}: {str(json_error)[:100]}")
                if attempt < MAX_RETRIES:
                    # Save the problematic response for debugging
                    logger.debug(f"   Problematic response: {response_text[:300]}")
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.info(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue  # Retry with new API call
                else:
                    logger.error(f"   ✗ Max retries reached - all JSON extraction attempts failed")
                    raise
            
        except ValueError:
            # Re-raise JSON extraction error (already logged above)
            if attempt >= MAX_RETRIES:
                raise
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                logger.warning(f"   ⏱️  Timeout on attempt {attempt}/{MAX_RETRIES}: {error_msg}")
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.info(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"   ✗ Max retries reached for inference")
                    raise
            else:
                logger.error(f"   ✗ Non-timeout error: {error_msg}")
                raise


def process_instance(instance, client, inference_model, step_num):
    """Process a single instance with inference."""
    instance_id = instance['id']
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {instance_id}")
    logger.info(f"{'='*70}")
    
    try:
        # Load trajectory
        logger.info(f"📂 Loading trajectory from {instance['trajectory']}...")
        trajectory = load_trajectory(instance['trajectory'], step_num=step_num)
        
        # Skip if trajectory file not found
        if trajectory is None:
            logger.warning(f"⏭️  Skipping {instance_id} - trajectory file not found")
            return None, "Trajectory file not found"
        
        logger.info(f"   ✓ Loaded {len(trajectory)} turns")
        
        # Run Inference
        logger.info("📝 Running inference...")
        
        # Check if instance has image
        image_id = instance.get('image', None)
        if image_id:
            logger.info(f"🖼️  Instance has image: {image_id}")
        
        inference_result = infer_three_dimensions(
            observation=instance['observation'],
            explicit_instruction=instance['explicit_instruction'],
            trajectory=trajectory,
            client=client,
            inference_model=inference_model,
            image_id=image_id
        )
        
        # Combine original data with inference
        result = {
            "instance_id": instance_id,
            "domain": instance['domain'],
            "observation": instance['observation'],
            "explicit_instruction": instance['explicit_instruction'],
            "trajectory_path": instance['trajectory'],
            "ground_truth": {
                'user_profile': instance['user_profile'],
                'user_latent_belief': instance['user_latent_belief'],
                'true_latent_state': instance['true_latent_state'],
                'root_cause_of_misconception': instance['root_cause_of_misconception']
            },
            "rubrics": instance.get('rubrics', {}),
            "inference": inference_result,
            "inferred_at": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"✓ {instance_id} inference completed")
        
        return result, None
        
    except Exception as e:
        logger.error(f"✗ Error processing {instance_id}: {str(e)}")
        return None, str(e)


def main():
    """Main inference loop."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run inference on latent belief dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python inference.py --input pref_latent_belief_dataset_new1.json --model gpt-5-2025-08-07 --steps 5
  python inference.py -i dataset.json -m qwen3-32b -s 10
  python inference.py -i pref_latent_belief_dataset_new1.json -m qwen3-32b -s 0  # Use only observation and instruction, no trajectory
        """
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input dataset JSON file'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Name of the inference model to use (e.g., gpt-5-2025-08-07, qwen3-32b)'
    )
    parser.add_argument(
        '--steps', '-s',
        type=int,
        required=True,
        help='Number of trajectory steps to keep (0 = no trajectory, positive number = first N steps)'
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
    parser.add_argument(
        '--trajectories-dir',
        type=str,
        default=None,
        help='Base directory for trajectory files (overrides TRAJECTORIES_DIR env var)'
    )
    
    args = parser.parse_args()
    input_file = args.input
    inference_model = args.model
    step_num = args.steps
    
    # Override configurations from command line if provided
    global api_key, base_url, TRAJECTORIES_DIR
    if args.api_key:
        api_key = args.api_key
    if args.base_url:
        base_url = args.base_url
    if args.trajectories_dir:
        TRAJECTORIES_DIR = Path(args.trajectories_dir)
    
    # Create output filename with model name, steps, and timestamp
    # Sanitize model name for filename (replace special characters)
    safe_model_name = inference_model.replace('/', '_').replace(':', '_').replace(' ', '_')
    input_data_name = input_file.split('/')[-1].split('.')[0]

    output_file = inference_dir / f"inference_{input_data_name}_{safe_model_name}_step{step_num}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Print pipeline information
    logger.info("="*70)
    logger.info("Latent Belief Inference Pipeline")
    logger.info("="*70)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Inference model: {inference_model}")
    logger.info(f"Trajectory steps: {step_num if step_num > 0 else 'No trajectory (observation & instruction only)'}")
    logger.info("="*70)
    
    # Load dataset
    logger.info(f"\n📖 Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    logger.info(f"   ✓ Loaded {len(dataset)} instances")
    
    # Select subset if NUM_INSTANCES_TO_PROCESS is set
    if NUM_INSTANCES_TO_PROCESS is not None:
        dataset = dataset[:NUM_INSTANCES_TO_PROCESS]
        logger.info(f"   ℹ️  Processing only first {len(dataset)} instances")
    
    # Initialize OpenAI client
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # Initialize results file with empty list
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f)
    logger.info(f"   ✓ Initialized output file: {output_file}")
    
    # Process each instance
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_instances = []
    skipped_instances = []
    
    for idx, instance in enumerate(dataset):
        logger.info(f"\n{'='*70}")
        logger.info(f"Progress: {idx + 1}/{len(dataset)}")
        logger.info(f"{'='*70}")
        
        result, error = process_instance(instance, client, inference_model, step_num)
        
        if result:
            # Append to results file immediately
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            results.append(result)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            success_count += 1
            logger.info(f"💾 Saved result for {instance['id']}")
        else:
            # Check if it's a skip (trajectory not found) or real error
            if error == "Trajectory file not found":
                skipped_count += 1
                skipped_instances.append(instance['id'])
            else:
                failed_count += 1
                failed_instances.append({
                    "instance_id": instance['id'],
                    "error": error
                })
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("INFERENCE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"✓ Successful: {success_count}/{len(dataset)}")
    logger.info(f"⏭️  Skipped (no trajectory): {skipped_count}/{len(dataset)}")
    logger.info(f"✗ Failed: {failed_count}/{len(dataset)}")
    
    if skipped_instances:
        logger.info(f"\nSkipped instances (trajectory not found):")
        for instance_id in skipped_instances:
            logger.info(f"  - {instance_id}")
    
    if failed_instances:
        logger.warning(f"\nFailed instances:")
        for fail in failed_instances:
            logger.warning(f"  - {fail['instance_id']}: {fail['error']}")
    
    logger.info(f"\n📊 Inference results saved to: {output_file}")
    logger.info(f"   Use this file as input for evaluation.py to score the inferences")


if __name__ == "__main__":
    main()

