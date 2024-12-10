# eval.py
import concurrent.futures
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import time
from typing import List, Tuple, Dict, Set
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

from washdetector.generator.models import GeneratorParams, WashCamouflageParams, WashCamouflageStrategy
from washdetector.generator.generator import TradingDataGenerator
from washdetector.utils.validation import validate_transactions
from washdetector.visualization.graph import create_transaction_graph
from washdetector.agent.wash_detector import analyze_wash_trading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    execution_time: float
    ground_truth_info: Dict
    detection_info: Dict

@dataclass
class TestCase:
    difficulty: Difficulty
    params: GeneratorParams
    seed: int

def setup_output_directory() -> Path:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"evaluation_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_params_by_difficulty(difficulty: Difficulty, seed: int) -> GeneratorParams:
    """Generate parameters based on difficulty level"""
    base_params = {
        'seed': seed,
        'token_symbol': 'TEST',
        'price_volatility': 0.1,
        'normal_trade_size_mean': 15.0,
        'normal_trade_size_std': 5.0,
    }
    
    if difficulty == Difficulty.EASY:
        return GeneratorParams(
            **base_params,
            num_transactions=300,
            num_normal_traders=100,
            num_wash_groups=1,
            wash_group_sizes=(4,),
            wash_amounts=(20.0,),
            wash_tx_counts=(30,),
            time_span_days=30,
        )
    elif difficulty == Difficulty.MEDIUM:
        return GeneratorParams(
            **base_params,
            num_transactions=500,
            num_normal_traders=100,
            num_wash_groups=1,
            wash_group_sizes=(4,),
            wash_amounts=(20.0,),
            wash_tx_counts=(30,),
            time_span_days=30,
            num_market_makers=5,
            market_maker_activeness=0.6,
        )
    else:  # HARD
        return GeneratorParams(
            **base_params,
            num_transactions=1000,
            num_normal_traders=100,
            num_wash_groups=2,
            wash_group_sizes=(4, 7),
            wash_amounts=(20.0, 10.0),
            wash_tx_counts=(30, 20),
            time_span_days=30,
            num_market_makers=5,
            market_maker_activeness=0.6,
            camouflage=WashCamouflageParams(
                strategy=WashCamouflageStrategy.HYBRID,
                normal_trade_probability=0.3,
                volume_leakage=0.2,
                normal_connections_per_washer=5,
                wash_amount_variance=0.3,
                min_time_between_wash=1.0
            )
        )

def generate_test_cases(num_cases_per_difficulty: int) -> List[TestCase]:
    """Generate test cases for each difficulty level"""
    test_cases = []
    base_seed = 42

    for difficulty in Difficulty:
        for i in range(num_cases_per_difficulty):
            seed = base_seed + (i * 100) + (list(Difficulty).index(difficulty) * 1000)
            params = get_params_by_difficulty(difficulty, seed)
            test_cases.append(TestCase(difficulty, params, seed))
    
    return test_cases

def compute_metrics(ground_truth: Set[str], predicted: Set[str], all_addresses: Set[str], 
                   execution_time: float, ground_truth_info: Dict, detection_info: Dict) -> EvaluationMetrics:
    """Compute evaluation metrics comparing ground truth to predicted wash traders"""
    true_positives = len(ground_truth.intersection(predicted))
    false_positives = len(predicted - ground_truth)
    false_negatives = len(ground_truth - predicted)
    true_negatives = len(all_addresses - ground_truth - predicted)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        execution_time=execution_time,
        ground_truth_info=ground_truth_info,
        detection_info=detection_info
    )

def plot_metrics(results: List[Dict], output_dir: Path):
    """Generate and save visualization plots"""
    # Prepare data
    df = pd.DataFrame(results)
    
    # Plot 1: Precision, Recall, F1 by Difficulty
    plt.figure(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1_score']
    
    df_melted = df.melt(
        id_vars=['difficulty'],
        value_vars=metrics,
        var_name='Metric',
        value_name='Score'
    )
    
    sns.barplot(data=df_melted, x='difficulty', y='Score', hue='Metric')
    plt.title('Detection Performance by Difficulty')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_difficulty.png')
    plt.close()
    
    # Plot 2: Confusion Matrices
    for difficulty in df['difficulty'].unique():
        diff_data = df[df['difficulty'] == difficulty]
        conf_matrix = np.array([
            [diff_data['true_negatives'].mean(), diff_data['false_positives'].mean()],
            [diff_data['false_negatives'].mean(), diff_data['true_positives'].mean()]
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=['Normal', 'Wash Trade'],
            yticklabels=['Normal', 'Wash Trade']
        )
        plt.title(f'Average Confusion Matrix - {difficulty} Difficulty')
        plt.tight_layout()
        plt.savefig(output_dir / f'confusion_matrix_{difficulty}.png')
        plt.close()
    
    # Plot 3: Performance vs Time
    plt.figure(figsize=(10, 6))
    for difficulty in df['difficulty'].unique():
        diff_data = df[df['difficulty'] == difficulty]
        plt.scatter(
            diff_data['execution_time'],
            diff_data['f1_score'],
            label=difficulty,
            alpha=0.7
        )
    
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('F1 Score')
    plt.title('Detection Performance vs Execution Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_time.png')
    plt.close()
    
    # Plot 4: Volume Distribution
    plt.figure(figsize=(10, 6))
    volume_data = []
    for result in results:
        gt_info = result['ground_truth_info']
        volume_data.append({
            'difficulty': result['difficulty'],
            'wash_volume_pct': gt_info['metrics']['wash_volume_percentage'],
            'normal_volume_pct': 1 - gt_info['metrics']['wash_volume_percentage']
        })
    
    volume_df = pd.DataFrame(volume_data)
    volume_df_melted = volume_df.melt(
        id_vars=['difficulty'],
        var_name='Volume Type',
        value_name='Percentage'
    )
    
    sns.boxplot(data=volume_df_melted, x='difficulty', y='Percentage', hue='Volume Type')
    plt.title('Volume Distribution by Difficulty')
    plt.tight_layout()
    plt.savefig(output_dir / 'volume_distribution.png')
    plt.close()

def run_single_evaluation(test_case: TestCase, api_key: str) -> Dict:
    """Run evaluation for a single test case"""
    start_time = time.time()
    
    try:
        # Generate and validate transactions
        generator = TradingDataGenerator(test_case.params)
        transactions = generator.generate_transactions()
        is_valid = validate_transactions(transactions)
        
        # Get ground truth information
        ground_truth_info = generator.get_ground_truth_info()
        ground_truth_addresses = set()
        for group in generator.wash_groups:
            ground_truth_addresses.update(group)
        
        # Run analysis
        analysis = analyze_wash_trading(transactions, api_key)
        predicted_addresses = set(analysis.suspicious_addresses)
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Compute metrics
        metrics = compute_metrics(
            ground_truth_addresses,
            predicted_addresses,
            set(generator.addresses),
            execution_time,
            ground_truth_info,
            analysis.dict()
        )
        
        # Generate visualization
        graph_path = create_transaction_graph(transactions)
        
        return {
            "difficulty": test_case.difficulty.value,
            "seed": test_case.seed,
            "num_transactions": len(transactions),
            "is_valid": is_valid,
            "graph_path": str(graph_path),
            "execution_time": metrics.execution_time,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "true_negatives": metrics.true_negatives,
            "ground_truth_info": metrics.ground_truth_info,
            "detection_info": metrics.detection_info
        }
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

def save_results(results: List[Dict], output_dir: Path):
    """Save evaluation results to files"""
    # Save detailed results as JSON
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)
    
    # Generate LaTeX tables
    metrics_by_difficulty = summary_df.groupby('difficulty').agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'execution_time': ['mean', 'std']
    }).round(3)
    
    with open(output_dir / "latex_tables.tex", "w") as f:
        f.write("% Detection Performance\n")
        f.write(metrics_by_difficulty.to_latex())
    
    # Print summary statistics
    logger.info("\nEvaluation Summary:")
    logger.info("==================")
    logger.info(f"Total test cases: {len(results)}")
    logger.info("\nMetrics by difficulty:")
    logger.info("\n" + str(metrics_by_difficulty))
    
    # Generate plots
    plot_metrics(results, output_dir)

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Setup output directory
    output_dir = setup_output_directory()
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Generate test cases (4 cases per difficulty level)
    test_cases = generate_test_cases(4)
    logger.info(f"Generated {len(test_cases)} test cases")
    
    # Run evaluations in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_case = {
            executor.submit(run_single_evaluation, case, api_key): case 
            for case in test_cases
        }
        
        for future in concurrent.futures.as_completed(future_to_case):
            case = future_to_case[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed evaluation for {case.difficulty.value} case with seed {case.seed}")
            except Exception as e:
                logger.error(f"Error processing {case.difficulty.value} case with seed {case.seed}: {str(e)}")
    
    # Save and display results
    save_results(results, output_dir)

if __name__ == "__main__":
    main()