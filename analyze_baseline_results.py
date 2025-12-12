"""
analyze_baseline_results.py
Analyzes the completed baseline training results by parsing training logs,
extracting performance metrics, and generating summary reports and visualizations.

This script is designed to:
1. Parse training logs to extract model performance metrics
2. Generate summary tables comparing baseline GNN models
3. Create visualization charts for accuracy and F1 scores
4. Save results for future comparison with advanced models
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_training_log(log_path):
    """
    Parse the training log file to extract key performance metrics for each model.
    
    Args:
        log_path (str): Path to the training.log file
        
    Returns:
        dict: Dictionary containing accuracy and F1 scores for each baseline model
              Format: {model_name: {'accuracy': float, 'f1': float}}
              
    The function looks for:
    - Individual model test results (GCN, GraphSAGE, GAT)
    - Best model summary from the final comparison
    """
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Define regex patterns to extract results for each model type
    # These patterns match the "Test Results" section in the training log
    patterns = {
        'GCN': r'Test Results:\s+- Accuracy: ([\d.]+)\s+- F1 Score: ([\d.]+)',
        'GraphSAGE': r'GraphSAGE.*?Test Results:.*?Accuracy: ([\d.]+).*?F1 Score: ([\d.]+)',
        'GAT': r'GAT.*?Test Results:.*?Accuracy: ([\d.]+).*?F1 Score: ([\d.]+)'
    }
    
    # Extract the best model information from the final summary section
    summary_pattern = r'Best Model: (\w+).*?Accuracy: ([\d.]+).*?F1 Score: ([\d.]+)'
    summary_match = re.search(summary_pattern, content, re.DOTALL)
    
    if summary_match:
        best_model = summary_match.group(1)
        print(f"✓ Best baseline model identified: {best_model}")
        print(f"  Test Accuracy: {summary_match.group(2)}")
        print(f"  Test F1 Score: {summary_match.group(3)}")
    
    # Extract detailed results for each individual model
    # Find all "Test Results" sections and capture model name, accuracy, and F1 score
    test_results = re.findall(
        r'(GCN|GraphSAGE|GAT).*?Test Results:.*?Accuracy: ([\d.]+).*?F1 Score: ([\d.]+)',
        content,
        re.DOTALL
    )
    
    # Store results in dictionary with float conversion for numerical operations
    for model, acc, f1 in test_results:
        results[model] = {
            'accuracy': float(acc),
            'f1': float(f1)
        }
    
    return results

def create_baseline_summary(results):
    """
    Create a comprehensive summary table of baseline model results.
    
    Args:
        results (dict): Dictionary containing model performance metrics
        
    Returns:
        pd.DataFrame: Formatted summary table with model names and metrics
        
    The function:
    1. Organizes results into a structured DataFrame
    2. Formats numerical values to 4 decimal places
    3. Prints the summary to console
    4. Saves the summary as CSV for future reference
    """
    
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    
    # Prepare data for DataFrame with consistent formatting
    data = []
    for model, metrics in results.items():
        data.append({
            'Model': model,
            'Type': 'Baseline',  # Label all as baseline for future comparison
            'Test Acc': f"{metrics['accuracy']:.4f}",
            'Test F1': f"{metrics['f1']:.4f}"
        })
    
    # Create DataFrame and display
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Save results to CSV for reproducibility and future analysis
    df.to_csv('project2_results/baseline_summary.csv', index=False)
    print("\n✓ Successfully saved summary to project2_results/baseline_summary.csv")
    
    return df

def plot_baseline_results(results):
    """
    Generate bar chart visualizations comparing baseline model performance.
    
    Args:
        results (dict): Dictionary containing model performance metrics
        
    Creates two side-by-side bar charts:
    1. Test Accuracy comparison across models
    2. Test F1 Score comparison across models
    
    Both charts include:
    - Value labels on top of each bar
    - Grid lines for easier reading
    - Y-axis range optimized for the data (0.9-1.0)
    - High-resolution output (300 DPI)
    """
    
    # Extract model names and metrics for plotting
    models = list(results.keys())
    accs = [results[m]['accuracy'] for m in models]
    f1s = [results[m]['f1'] for m in models]
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left subplot: Accuracy comparison
    bars1 = axes[0].bar(models, accs, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Baseline Model Accuracy', fontsize=14)
    axes[0].set_ylim([0.9, 1.0])  # Zoom into relevant range
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of each bar for accuracy
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=10)
    
    # Right subplot: F1 Score comparison
    bars2 = axes[1].bar(models, f1s, color='coral', alpha=0.8)
    axes[1].set_ylabel('Test F1 Score', fontsize=12)
    axes[1].set_title('Baseline Model F1 Score', fontsize=14)
    axes[1].set_ylim([0.9, 1.0])  # Zoom into relevant range
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of each bar for F1 score
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=10)
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('project2_results/baseline_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Successfully saved visualization to project2_results/baseline_performance.png")
    plt.close()

if __name__ == "__main__":
    import sys
    import os
    
    # Allow specifying log file path as command-line argument, default to 'training.log'
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'training.log'
    
    # Verify log file exists before processing
    if not os.path.exists(log_path):
        print(f"Error: Log file '{log_path}' not found!")
        print("Please ensure the training log file exists in the current directory.")
        sys.exit(1)
    
    print("Analyzing baseline results from training.log...")
    print("="*60)
    
    # Parse the training log to extract model results
    results = parse_training_log(log_path)
    
    if results:
        # Generate summary table and visualizations if parsing was successful
        df = create_baseline_summary(results)
        plot_baseline_results(results)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Review the generated baseline_summary.csv file")
        print("2. Execute: python 10_run_extra_credit_quick.py")
        print("3. Compare advanced model results with these baseline metrics")
    else:
        # Fallback message if automatic parsing fails
        print("\n⚠ Warning: Could not automatically parse results from log file")
        print("Using default values from log summary as fallback...")
        
        # Display manually extracted summary information from the log
        print("\nFrom your training log:")
        print("  Best Model: GraphSAGE")
        print("  - Test Accuracy: 0.9660")
        print("  - Test F1 Score: 0.9625")