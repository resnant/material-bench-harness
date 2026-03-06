#!/usr/bin/env python3
"""
Material Bench Benchmark Runner

Usage:
    python scripts/run_benchmark.py --dataset material-figbench --api-base http://localhost:8001
    python scripts/run_benchmark.py --dataset materialbench-choice --api-base http://localhost:8001
    python scripts/run_benchmark.py --dataset materialbench-free --api-base http://localhost:8001
    python scripts/run_benchmark.py --dataset all --api-base http://localhost:8001
    python scripts/run_benchmark.py --dataset all --api-base http://localhost:8001 --use-llm-judge
"""

import argparse
import base64
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


SYSTEM_PROMPT = """You are an expert in materials science and engineering. 
Answer the following questions accurately based on your knowledge and the provided information.
For multiple choice questions, respond with only the letter (a, b, c, or d).
For other questions, provide your final answer in the format: **Answer:** [your answer]
Keep your explanation concise and put the final answer at the end."""


def extract_answer_from_text(text):
    """
    Extract the final answer from model's verbose response.
    
    Priority:
    1. Text after **Answer:** or **Answer**
    2. Bolded numbers
    3. Last number in text
    """
    if text is None:
        return ""
    
    text = str(text)
    
    # Pattern 1: Extract after **Answer:**
    answer_patterns = [
        r'\*\*Answer:\*\*\s*[:\s]*(.*?)(?:\n|$)',
        r'\*\*Answer\*\*\s*[:\s]*(.*?)(?:\n|$)',
        r'Answer:\s*(.*?)(?:\n|$)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
            if answer_part and len(answer_part) < 100:
                # Extract number from answer part if it's long
                num_match = re.search(r'(\d+\.?\d*\s*(?:×|x|\\times)?\s*10\^?[-+]?\d+|\d+\.?\d*)', answer_part)
                if num_match:
                    return num_match.group(1)
                return answer_part
    
    # Pattern 2: Bolded numbers
    bold_numbers = re.findall(r'\*\*\s*(\d+\.?\d*\s*(?:×|x|\\times)?\s*10\^?[-+]?\d+|\d+\.?\d*)\s*\*\*', text)
    if bold_numbers:
        return bold_numbers[-1]
    
    # Pattern 3: Last bold block
    bold_blocks = re.findall(r'\*\*(.+?)\*\*', text)
    if bold_blocks:
        last_block = bold_blocks[-1].strip()
        num_match = re.search(r'(\d+\.?\d*\s*(?:×|x|\\times)?\s*10\^?[-+]?\d+|\d+\.?\d*)', last_block)
        if num_match:
            return num_match.group(1)
    
    # Pattern 4: Last number in text
    all_numbers = re.findall(r'(\d+\.?\d*)', text)
    if all_numbers:
        return all_numbers[-1]
    
    return text


def normalize(text):
    """Normalize text: lowercase, whitespace removal, and numeric notation normalization"""
    if text is None:
        return ""
    
    text = str(text).lower()
    
    # Remove all whitespace
    text = re.sub(r'\s+', '', text)
    
    # Normalize scientific notation: 10^{-9} -> 10^-9, 10^(-9) -> 10^-9
    text = re.sub(r'10\^\{?(-?\d+)\}?', r'10^\1', text)
    
    # Normalize multiplication symbols
    text = text.replace('×', 'x')
    
    # Remove common punctuation
    text = text.replace(',', '').replace('.', '')
    
    return text


def check_correct(prediction, ground_truth, answer_range_2=None):
    """
    Check if prediction is correct.
    - First extract the answer from verbose model response
    - For numeric ranges (e.g., "0.0825-0.0975"): check if answer is within range
    - For numeric answers: check with tolerance
    - Otherwise: normalized exact match
    """
    # Extract answer from verbose response
    extracted_answer = extract_answer_from_text(prediction)
    
    pred = normalize(extracted_answer)
    truth = normalize(ground_truth)
    
    # Numeric range check
    if answer_range_2 and answer_range_2 != '-' and answer_range_2 != 'None':
        match = re.match(r'([\d.]+)-([\d.]+)', str(answer_range_2))
        if match:
            try:
                low, high = float(match.group(1)), float(match.group(2))
                value = float(pred)
                return low <= value <= high
            except (ValueError, TypeError):
                pass
    
    # Try numeric comparison with tolerance
    try:
        # Remove non-numeric characters for comparison (except . and -)
        pred_clean = re.sub(r'[^\d.\-]', '', pred)
        truth_clean = re.sub(r'[^\d.\-]', '', truth)
        
        pred_num = float(pred_clean)
        truth_num = float(truth_clean)
        # 3% tolerance for numeric answers
        tolerance = abs(truth_num) * 0.03
        return abs(pred_num - truth_num) <= tolerance
    except (ValueError, TypeError):
        pass
    
    # Exact match after normalization
    return pred == truth


def load_material_figbench(hf_token=None, max_samples=0, seed=42):
    """Load MaterialFigBench dataset"""
    headers = {}
    if hf_token:
        headers['Authorization'] = f'Bearer {hf_token}'
    
    # Load JSON metadata
    json_url = "https://huggingface.co/datasets/omron-sinicx/MaterialFigBench/resolve/main/MaterialFigBench.json"
    resp = requests.get(json_url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to load MaterialFigBench.json: {resp.status_code}")
    
    data = resp.json()
    
    # Load image list
    api_url = "https://huggingface.co/api/datasets/omron-sinicx/MaterialFigBench/tree/main/MaterialFigBench_Figs_images"
    resp = requests.get(api_url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to load image list: {resp.status_code}")
    
    image_files = {f['path'].split('/')[-1]: f['path'] for f in resp.json()}
    
    # Prepare samples
    samples = []
    for item in data:
        sample = {
            'question_id': item['problem#'],
            'problem_sentence': item['problem_sentence'],
            'image_files': [],
            'ground_truth': item.get('answer_range_1', ''),
            'answer_range_2': item.get('answer_range_2', ''),
            'original_problem': item.get('original_problem_#', ''),
            'textbook': item.get('original_textbook', '')
        }
        
        # Collect image files
        for key in ['image_file_1', 'image_file_2', 'image_file_3']:
            if item.get(key):
                sample['image_files'].append(item[key])
        
        samples.append(sample)
    
    # Random sampling
    if max_samples > 0 and max_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, max_samples)
    
    return samples, image_files, headers


def get_image_cache_dir():
    """Get image cache directory"""
    cache_dir = Path.home() / ".cache" / "material_bench" / "images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_image_base64(image_filename, image_files_map, headers, use_cache=True):
    """Load image and return base64 encoded string with caching"""
    cache_dir = get_image_cache_dir()
    cache_file = cache_dir / image_filename
    
    # Try to load from cache
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                img_data = f.read()
        except:
            cache_file.unlink(missing_ok=True)
            img_data = None
    else:
        img_data = None
    
    # Download if not cached
    if img_data is None:
        image_path = image_files_map.get(image_filename)
        if not image_path:
            image_path = f"MaterialFigBench_Figs_images/{image_filename}"
        
        image_url = f"https://huggingface.co/datasets/omron-sinicx/MaterialFigBench/resolve/main/{image_path}"
        resp = requests.get(image_url, headers=headers)
        if resp.status_code != 200:
            return None
        
        img_data = resp.content
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                f.write(img_data)
        except:
            pass
    
    # Encode to base64
    img = Image.open(BytesIO(img_data))
    buffered = BytesIO()
    img.save(buffered, format='PNG')
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


def load_materialbench_choice(hf_token=None, max_samples=0, seed=42):
    """Load MaterialBENCH choice dataset"""
    headers = {}
    if hf_token:
        headers['Authorization'] = f'Bearer {hf_token}'
    
    json_url = "https://huggingface.co/datasets/omron-sinicx/MaterialBENCH/resolve/main/choice_dataset.json"
    resp = requests.get(json_url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to load choice_dataset.json: {resp.status_code}")
    
    data = resp.json()
    
    samples = []
    for i, item in enumerate(data):
        sample = {
            'question_id': i + 1,
            'problem_sentence': item['problem sentence'],
            'choices_a': item['a'],
            'choices_b': item['b'],
            'choices_c': item['c'],
            'choices_d': item['d'],
            'correct_choice': item['correct choice'],
            'textbook': item.get('textbook', ''),
            'question_number': item.get('question number in the text book', '')
        }
        samples.append(sample)
    
    # Random sampling
    if max_samples > 0 and max_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, max_samples)
    
    return samples


def load_materialbench_free(hf_token=None, max_samples=0, seed=42):
    """Load MaterialBENCH free dataset"""
    headers = {}
    if hf_token:
        headers['Authorization'] = f'Bearer {hf_token}'
    
    json_url = "https://huggingface.co/datasets/omron-sinicx/MaterialBENCH/resolve/main/free_dataset.json"
    resp = requests.get(json_url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to load free_dataset.json: {resp.status_code}")
    
    data = resp.json()
    
    samples = []
    for i, item in enumerate(data):
        sample = {
            'question_id': i + 1,
            'problem_sentence': item['problem sentence'],
            'correct_answer': item['correct answer'],
            'textbook': item.get('textbook', ''),
            'question_number': item.get('question number in the text book', '')
        }
        samples.append(sample)
    
    # Random sampling
    if max_samples > 0 and max_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, max_samples)
    
    return samples


def call_api(client, model, messages, timeout, max_retries=2):
    """Call OpenAI API with retry logic"""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4...
                print(f"\nAPI error: {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    return None


def process_figbench_sample(args_tuple):
    """Process a single MaterialFigBench sample (for parallel execution)"""
    sample, model, image_files_map, headers, timeout, max_retries, pbar_lock = args_tuple
    
    # Load images
    image_contents = []
    image_error = False
    for img_file in sample['image_files']:
        img_base64 = load_image_base64(img_file, image_files_map, headers)
        if img_base64 is None:
            image_error = True
            break
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
        })
    
    if image_error:
        return None
    
    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": sample['problem_sentence']},
            *image_contents
        ]}
    ]
    
    # Create a new client for this thread
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="not-needed"
    )
    
    # Call API
    try:
        prediction = call_api(client, model, messages, timeout, max_retries)
    except Exception as e:
        prediction = None
    
    # Check correctness
    correct = check_correct(prediction, sample['ground_truth'], sample['answer_range_2'])
    
    return {
        'question_id': sample['question_id'],
        'problem_sentence': sample['problem_sentence'],
        'image_files': ';'.join(sample['image_files']),
        'prediction': prediction,
        'ground_truth': sample['ground_truth'],
        'answer_range_2': sample['answer_range_2'],
        'correct': correct
    }


def process_choice_sample(args_tuple):
    """Process a single MaterialBENCH choice sample (for parallel execution)"""
    sample, model, timeout, max_retries = args_tuple
    
    prompt = f"""{sample['problem_sentence']}

a) {sample['choices_a']}
b) {sample['choices_b']}
c) {sample['choices_c']}
d) {sample['choices_d']}

Respond with only the letter (a, b, c, or d)."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    # Create a new client for this thread
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="not-needed"
    )
    
    # Call API
    try:
        prediction = call_api(client, model, messages, timeout, max_retries)
        if prediction:
            match = re.search(r'\b([abcd])\b', prediction.lower())
            if match:
                prediction = match.group(1)
            else:
                prediction = prediction.strip().lower()
    except Exception as e:
        prediction = None
    
    # Check correctness
    correct = normalize(prediction) == normalize(sample['correct_choice'])
    
    return {
        'question_id': sample['question_id'],
        'problem_sentence': sample['problem_sentence'],
        'choices_a': sample['choices_a'],
        'choices_b': sample['choices_b'],
        'choices_c': sample['choices_c'],
        'choices_d': sample['choices_d'],
        'prediction': prediction,
        'correct_choice': sample['correct_choice'],
        'correct': correct
    }


def process_free_sample(args_tuple):
    """Process a single MaterialBENCH free sample (for parallel execution)"""
    sample, model, timeout, max_retries = args_tuple
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample['problem_sentence']}
    ]
    
    # Create a new client for this thread
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="not-needed"
    )
    
    # Call API
    try:
        prediction = call_api(client, model, messages, timeout, max_retries)
    except Exception as e:
        prediction = None
    
    # Check correctness
    correct = check_correct(prediction, sample['correct_answer'])
    
    return {
        'question_id': sample['question_id'],
        'problem_sentence': sample['problem_sentence'],
        'prediction': prediction,
        'correct_answer': sample['correct_answer'],
        'correct': correct
    }


def run_material_figbench(client, model, samples, image_files_map, headers, output_dir, timeout, max_retries, num_threads=1):
    """Run benchmark for MaterialFigBench"""
    results = []
    failed_count = 0
    
    print(f"\nRunning MaterialFigBench benchmark ({len(samples)} samples)...")
    print(f"Images will be cached in ~/.cache/material_bench/images/")
    print(f"Using {num_threads} thread(s)...")
    
    # Prepare arguments for each sample
    sample_args = [
        (sample, model, image_files_map, headers, timeout, max_retries, None)
        for sample in samples
    ]
    
    if num_threads == 1:
        # Sequential execution
        for sample in tqdm(samples, desc="MaterialFigBench"):
            args = (sample, model, image_files_map, headers, timeout, max_retries, None)
            result = process_figbench_sample(args)
            if result is None:
                failed_count += 1
            else:
                results.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_figbench_sample, args): args[0] for args in sample_args}
            
            with tqdm(total=len(samples), desc="MaterialFigBench") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        failed_count += 1
                    else:
                        results.append(result)
                    pbar.update(1)
    
    if failed_count > 0:
        print(f"Warning: {failed_count} samples failed")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"material-figbench_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results


def run_materialbench_choice(client, model, samples, output_dir, timeout, max_retries, num_threads=1):
    """Run benchmark for MaterialBENCH choice"""
    results = []
    failed_count = 0
    
    print(f"\nRunning MaterialBENCH choice benchmark ({len(samples)} samples)...")
    print(f"Using {num_threads} thread(s)...")
    
    # Prepare arguments for each sample
    sample_args = [
        (sample, model, timeout, max_retries)
        for sample in samples
    ]
    
    if num_threads == 1:
        # Sequential execution
        for sample in tqdm(samples, desc="MaterialBENCH choice"):
            args = (sample, model, timeout, max_retries)
            result = process_choice_sample(args)
            results.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_choice_sample, args): args[0] for args in sample_args}
            
            with tqdm(total=len(samples), desc="MaterialBENCH choice") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        failed_count += 1
                    else:
                        results.append(result)
                    pbar.update(1)
    
    if failed_count > 0:
        print(f"Warning: {failed_count} samples failed")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"materialbench-choice_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results


def run_materialbench_free(client, model, samples, output_dir, timeout, max_retries, num_threads=1):
    """Run benchmark for MaterialBENCH free"""
    results = []
    failed_count = 0
    
    print(f"\nRunning MaterialBENCH free benchmark ({len(samples)} samples)...")
    print(f"Using {num_threads} thread(s)...")
    
    # Prepare arguments for each sample
    sample_args = [
        (sample, model, timeout, max_retries)
        for sample in samples
    ]
    
    if num_threads == 1:
        # Sequential execution
        for sample in tqdm(samples, desc="MaterialBENCH free"):
            args = (sample, model, timeout, max_retries)
            result = process_free_sample(args)
            results.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_free_sample, args): args[0] for args in sample_args}
            
            with tqdm(total=len(samples), desc="MaterialBENCH free") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        failed_count += 1
                    else:
                        results.append(result)
                    pbar.update(1)
    
    if failed_count > 0:
        print(f"Warning: {failed_count} samples failed")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"materialbench-free_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results


JUDGE_SYSTEM_PROMPT = """You are evaluating answers for a materials science benchmark.
Determine if the model's prediction is correct compared to the ground truth.

Criteria:
- Meaning is correct (minor wording differences are OK)
- Numeric values within 1% error are acceptable
- Notation variations are acceptable

Output format:
- If correct: "CORRECT"
- If incorrect: "INCORRECT: [reason]"

Examples:
- CORRECT
- INCORRECT: The model predicted option (c), but the correct answer is (b).
- INCORRECT: The predicted value 5.2 differs from the ground truth 3.8 by more than 1%."""


def llm_judge_answer(problem, prediction, ground_truth, client, model, timeout=60):
    """
    Use LLM to judge if the prediction is correct compared to ground truth.
    Returns tuple of (is_correct: bool, reason: str or None).
    """
    if prediction is None or prediction == '':
        return (False, "Prediction is empty")
    
    prompt = f"""Problem: {problem}
Model Prediction: {prediction}
Ground Truth: {ground_truth}"""
    
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout
        )
        judgment = response.choices[0].message.content.strip()
        
        if judgment.upper().startswith("CORRECT"):
            return (True, None)
        elif judgment.upper().startswith("INCORRECT"):
            reason = judgment.split(":", 1)[1].strip() if ":" in judgment else "Incorrect answer"
            return (False, reason)
        else:
            return (False, f"Unable to parse judgment: {judgment}")
    except Exception as e:
        return (False, f"Error: {str(e)}")


def llm_judge_sample(args_tuple):
    """Process a single sample for LLM judge (for parallel execution)"""
    sample, client, model, problem_column, timeout = args_tuple
    
    problem = sample.get(problem_column, '')
    prediction = sample.get('prediction', '')
    
    if 'ground_truth' in sample:
        ground_truth = sample.get('ground_truth', '')
    elif 'correct_answer' in sample:
        ground_truth = sample.get('correct_answer', '')
    elif 'correct_choice' in sample:
        ground_truth = sample.get('correct_choice', '')
    else:
        ground_truth = ''
    
    is_correct, reason = llm_judge_answer(problem, prediction, ground_truth, client, model, timeout)
    
    result = sample.copy()
    result['llm_judge_correct'] = is_correct
    result['llm_judge_reason'] = reason
    return result


def run_llm_judge_on_results(results, problem_column, client, model, num_threads=1, timeout=60):
    """Run LLM judge on a list of results"""
    if not results:
        return []
    
    print(f"\nRunning LLM judge on {len(results)} samples...")
    print(f"Using {num_threads} thread(s)...")
    
    sample_args = [
        (result, client, model, problem_column, timeout)
        for result in results
    ]
    
    judged_results = []
    
    if num_threads == 1:
        for result in tqdm(results, desc="LLM Judge"):
            args = (result, client, model, problem_column, timeout)
            judged_result = llm_judge_sample(args)
            judged_results.append(judged_result)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(llm_judge_sample, args): args[0] for args in sample_args}
            
            with tqdm(total=len(results), desc="LLM Judge") as pbar:
                for future in as_completed(futures):
                    judged_result = future.result()
                    judged_results.append(judged_result)
                    pbar.update(1)
    
    correct_count = sum(1 for r in judged_results if r.get('llm_judge_correct', False))
    incorrect_with_reason = [(r.get('question_id'), r.get('llm_judge_reason')) 
                              for r in judged_results if not r.get('llm_judge_correct', False) and r.get('llm_judge_reason')]
    
    print(f"LLM Judge Accuracy: {correct_count}/{len(judged_results)} ({correct_count/len(judged_results):.2%})")
    if incorrect_with_reason:
        print(f"\nIncorrect samples with reasons:")
        for qid, reason in incorrect_with_reason[:5]:
            print(f"  Q{qid}: {reason}")
        if len(incorrect_with_reason) > 5:
            print(f"  ... and {len(incorrect_with_reason) - 5} more")
    
    return judged_results


def load_and_judge_csv(csv_file, client, model, num_threads=1, timeout=60):
    """Load a CSV file, run LLM judge, and save the results"""
    df = pd.read_csv(csv_file)
    results = df.to_dict('records')
    
    if 'problem_sentence' in df.columns:
        problem_column = 'problem_sentence'
    else:
        print(f"Warning: problem_sentence column not found in {csv_file}")
        return None
    
    judged_results = run_llm_judge_on_results(results, problem_column, client, model, num_threads, timeout)
    
    if not judged_results:
        return None
    
    df_judged = pd.DataFrame(judged_results)
    
    output_file = str(csv_file).replace('.csv', '_judged.csv')
    df_judged.to_csv(output_file, index=False)
    print(f"Judged results saved to {output_file}")
    
    return judged_results


def calculate_summary(all_results, output_dir, llm_judge_results=None):
    """Calculate and save summary statistics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'datasets': {},
        'overall': {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0
        }
    }
    
    for dataset_name, results in all_results.items():
        if not results:
            continue
        
        total = len(results)
        correct = sum(1 for r in results if r.get('correct', False))
        accuracy = correct / total if total > 0 else 0.0
        
        summary['datasets'][dataset_name] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }
        
        summary['overall']['total'] += total
        summary['overall']['correct'] += correct
        
        if llm_judge_results and dataset_name in llm_judge_results:
            llm_results = llm_judge_results[dataset_name]
            llm_correct = sum(1 for r in llm_results if r.get('llm_judge_correct', False))
            summary['datasets'][dataset_name]['llm_judge_correct'] = llm_correct
            summary['datasets'][dataset_name]['llm_judge_accuracy'] = llm_correct / len(llm_results) if len(llm_results) > 0 else 0.0
    
    if summary['overall']['total'] > 0:
        summary['overall']['accuracy'] = summary['overall']['correct'] / summary['overall']['total']
    
    if llm_judge_results:
        summary['overall']['llm_judge_total'] = sum(
            len(r) for r in llm_judge_results.values()
        )
        summary['overall']['llm_judge_correct'] = sum(
            sum(1 for r in results if r.get('llm_judge_correct', False))
            for results in llm_judge_results.values()
        )
        if summary['overall']['llm_judge_total'] > 0:
            summary['overall']['llm_judge_accuracy'] = (
                summary['overall']['llm_judge_correct'] / summary['overall']['llm_judge_total']
            )
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for dataset_name, stats in summary['datasets'].items():
        print(f"\n{dataset_name}:")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Accuracy: {stats['accuracy']:.2%}")
        if 'llm_judge_accuracy' in stats:
            print(f"  LLM Judge Correct: {stats['llm_judge_correct']}")
            print(f"  LLM Judge Accuracy: {stats['llm_judge_accuracy']:.2%}")
    
    print(f"\nOverall:")
    print(f"  Total: {summary['overall']['total']}")
    print(f"  Correct: {summary['overall']['correct']}")
    print(f"  Accuracy: {summary['overall']['accuracy']:.2%}")
    if 'llm_judge_accuracy' in summary['overall']:
        print(f"  LLM Judge Total: {summary['overall']['llm_judge_total']}")
        print(f"  LLM Judge Correct: {summary['overall']['llm_judge_correct']}")
        print(f"  LLM Judge Accuracy: {summary['overall']['llm_judge_accuracy']:.2%}")
    
    output_file = output_dir / f"summary_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Material Bench Benchmark Runner')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['material-figbench', 'materialbench-choice', 'materialbench-free', 'all'],
                        help='Dataset to run benchmark on')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--api-base', type=str, default='http://localhost:8001',
                        help='OpenAI-compatible API base URL')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-397B-A17B-FP8',
                        help='Model name')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Number of samples to randomly select (0 for all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--api-key', type=str, default='',
                        help='API key')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace token for higher rate limits')
    parser.add_argument('--timeout', type=int, default=300,
                        help='API timeout in seconds')
    parser.add_argument('--max-retries', type=int, default=2,
                        help='Maximum number of retries')
    parser.add_argument('--num-threads', type=int, default=1,
                        help='Number of parallel threads for API calls')
    parser.add_argument('--use-llm-judge', action='store_true',
                        help='Run LLM judge on saved results after inference')
    parser.add_argument('--judge-api-base', type=str, default=None,
                        help='API base URL for LLM judge (default: same as --api-base)')
    parser.add_argument('--judge-model', type=str, default=None,
                        help='Model name for LLM judge (default: same as --model)')
    parser.add_argument('--judge-num-threads', type=int, default=None,
                        help='Number of parallel threads for LLM judge (default: same as --num-threads)')
    parser.add_argument('--judge-timeout', type=int, default=60,
                        help='API timeout for LLM judge in seconds')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize API client
    client = OpenAI(
        base_url=args.api_base.rstrip('/') + '/v1',
        api_key=args.api_key if args.api_key else 'not-needed'
    )
    
    all_results = {}
    
    # Determine which datasets to run
    datasets_to_run = []
    if args.dataset == 'all':
        datasets_to_run = ['material-figbench', 'materialbench-choice', 'materialbench-free']
    else:
        datasets_to_run = [args.dataset]
    
    for dataset_name in datasets_to_run:
        if dataset_name == 'material-figbench':
            samples, image_files_map, headers = load_material_figbench(
                args.hf_token, args.max_samples, args.seed
            )
            results = run_material_figbench(
                client, args.model, samples, image_files_map, headers,
                output_dir, args.timeout, args.max_retries, args.num_threads
            )
            all_results['material-figbench'] = results
        
        elif dataset_name == 'materialbench-choice':
            samples = load_materialbench_choice(
                args.hf_token, args.max_samples, args.seed
            )
            results = run_materialbench_choice(
                client, args.model, samples, output_dir, args.timeout, args.max_retries, args.num_threads
            )
            all_results['materialbench-choice'] = results
        
        elif dataset_name == 'materialbench-free':
            samples = load_materialbench_free(
                args.hf_token, args.max_samples, args.seed
            )
            results = run_materialbench_free(
                client, args.model, samples, output_dir, args.timeout, args.max_retries, args.num_threads
            )
            all_results['materialbench-free'] = results
    
    # Calculate and save summary
    if all_results:
        if args.use_llm_judge:
            judge_api_base = args.judge_api_base if args.judge_api_base else args.api_base
            judge_model = args.judge_model if args.judge_model else args.model
            judge_num_threads = args.judge_num_threads if args.judge_num_threads else args.num_threads
            
            judge_client = OpenAI(
                base_url=judge_api_base.rstrip('/') + '/v1',
                api_key=args.api_key if args.api_key else 'not-needed'
            )
            
            llm_judge_results = {}
            
            for dataset_name in all_results.keys():
                if dataset_name == 'material-figbench':
                    csv_pattern = output_dir / "material-figbench_*.csv"
                elif dataset_name == 'materialbench-choice':
                    csv_pattern = output_dir / "materialbench-choice_*.csv"
                elif dataset_name == 'materialbench-free':
                    csv_pattern = output_dir / "materialbench-free_*.csv"
                else:
                    continue
                
                csv_files = list(output_dir.glob(csv_pattern.name))
                csv_files = [f for f in csv_files if '_judged' not in str(f)]
                
                if csv_files:
                    csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                    print(f"\nRunning LLM judge on {dataset_name} results: {csv_file}")
                    
                    judged_results = load_and_judge_csv(
                        csv_file, judge_client, judge_model,
                        judge_num_threads, args.judge_timeout
                    )
                    
                    if judged_results:
                        llm_judge_results[dataset_name] = judged_results
            
            calculate_summary(all_results, output_dir, llm_judge_results)
        else:
            calculate_summary(all_results, output_dir)


if __name__ == '__main__':
    main()
