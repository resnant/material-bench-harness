#!/usr/bin/env python3
"""
Material Bench Benchmark Runner

Usage:
    python scripts/run_benchmark.py --dataset material-figbench --api-base http://localhost:8001
    python scripts/run_benchmark.py --dataset materialbench-choice --api-base http://localhost:8001
    python scripts/run_benchmark.py --dataset materialbench-free --api-base http://localhost:8001
    python scripts/run_benchmark.py --dataset all --api-base http://localhost:8001
"""

import argparse
import base64
import json
import os
import random
import re
import time
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
For multiple choice questions, respond with only the letter (a, b, c, or d)."""


def normalize(text):
    """Normalize text: lowercase and whitespace normalization"""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_correct(prediction, ground_truth, answer_range_2=None):
    """
    Check if prediction is correct.
    - For numeric ranges (e.g., "0.0825-0.0975"): check if answer is within range
    - Otherwise: normalized exact match
    """
    pred = normalize(prediction)
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


def run_material_figbench(client, model, samples, image_files_map, headers, output_dir, timeout, max_retries):
    """Run benchmark for MaterialFigBench"""
    results = []
    
    print(f"\nRunning MaterialFigBench benchmark ({len(samples)} samples)...")
    print(f"Images will be cached in ~/.cache/material_bench/images/")
    
    for sample in tqdm(samples, desc="MaterialFigBench"):
        # Load images
        image_contents = []
        image_error = False
        for img_file in sample['image_files']:
            img_base64 = load_image_base64(img_file, image_files_map, headers)
            if img_base64 is None:
                print(f"\nWarning: Failed to load image {img_file}, skipping this sample")
                image_error = True
                break
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })
        
        if image_error:
            continue
        
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": sample['problem_sentence']},
                *image_contents
            ]}
        ]
        
        # Call API
        try:
            prediction = call_api(client, model, messages, timeout, max_retries)
        except Exception as e:
            print(f"\nError calling API for question {sample['question_id']}: {e}")
            prediction = None
        
        # Check correctness
        correct = check_correct(prediction, sample['ground_truth'], sample['answer_range_2'])
        
        results.append({
            'question_id': sample['question_id'],
            'problem_sentence': sample['problem_sentence'],
            'image_files': ';'.join(sample['image_files']),
            'prediction': prediction,
            'ground_truth': sample['ground_truth'],
            'answer_range_2': sample['answer_range_2'],
            'correct': correct
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"material-figbench_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results


def run_materialbench_choice(client, model, samples, output_dir, timeout, max_retries):
    """Run benchmark for MaterialBENCH choice"""
    results = []
    
    print(f"\nRunning MaterialBENCH choice benchmark ({len(samples)} samples)...")
    
    for sample in tqdm(samples, desc="MaterialBENCH choice"):
        # Build prompt
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
        
        # Call API
        try:
            prediction = call_api(client, model, messages, timeout, max_retries)
            # Extract just the letter
            if prediction:
                match = re.search(r'\b([abcd])\b', prediction.lower())
                if match:
                    prediction = match.group(1)
                else:
                    prediction = prediction.strip().lower()
        except Exception as e:
            print(f"\nError calling API for question {sample['question_id']}: {e}")
            prediction = None
        
        # Check correctness
        correct = normalize(prediction) == normalize(sample['correct_choice'])
        
        results.append({
            'question_id': sample['question_id'],
            'problem_sentence': sample['problem_sentence'],
            'choices_a': sample['choices_a'],
            'choices_b': sample['choices_b'],
            'choices_c': sample['choices_c'],
            'choices_d': sample['choices_d'],
            'prediction': prediction,
            'correct_choice': sample['correct_choice'],
            'correct': correct
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"materialbench-choice_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results


def run_materialbench_free(client, model, samples, output_dir, timeout, max_retries):
    """Run benchmark for MaterialBENCH free"""
    results = []
    
    print(f"\nRunning MaterialBENCH free benchmark ({len(samples)} samples)...")
    
    for sample in tqdm(samples, desc="MaterialBENCH free"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample['problem_sentence']}
        ]
        
        # Call API
        try:
            prediction = call_api(client, model, messages, timeout, max_retries)
        except Exception as e:
            print(f"\nError calling API for question {sample['question_id']}: {e}")
            prediction = None
        
        # Check correctness
        correct = check_correct(prediction, sample['correct_answer'])
        
        results.append({
            'question_id': sample['question_id'],
            'problem_sentence': sample['problem_sentence'],
            'prediction': prediction,
            'correct_answer': sample['correct_answer'],
            'correct': correct
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"materialbench-free_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results


def calculate_summary(all_results, output_dir):
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
    
    if summary['overall']['total'] > 0:
        summary['overall']['accuracy'] = summary['overall']['correct'] / summary['overall']['total']
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for dataset_name, stats in summary['datasets'].items():
        print(f"\n{dataset_name}:")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Accuracy: {stats['accuracy']:.2%}")
    
    print(f"\nOverall:")
    print(f"  Total: {summary['overall']['total']}")
    print(f"  Correct: {summary['overall']['correct']}")
    print(f"  Accuracy: {summary['overall']['accuracy']:.2%}")
    
    # Save summary
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
                output_dir, args.timeout, args.max_retries
            )
            all_results['material-figbench'] = results
        
        elif dataset_name == 'materialbench-choice':
            samples = load_materialbench_choice(
                args.hf_token, args.max_samples, args.seed
            )
            results = run_materialbench_choice(
                client, args.model, samples, output_dir, args.timeout, args.max_retries
            )
            all_results['materialbench-choice'] = results
        
        elif dataset_name == 'materialbench-free':
            samples = load_materialbench_free(
                args.hf_token, args.max_samples, args.seed
            )
            results = run_materialbench_free(
                client, args.model, samples, output_dir, args.timeout, args.max_retries
            )
            all_results['materialbench-free'] = results
    
    # Calculate and save summary
    if all_results:
        calculate_summary(all_results, output_dir)


if __name__ == '__main__':
    main()
