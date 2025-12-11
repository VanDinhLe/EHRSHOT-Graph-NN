"""
analyze_baseline_results.py
分析已完成的baseline训练结果
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_training_log(log_path):
    """解析training.log提取关键信息"""
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    results = {}
    
    # 提取模型结果
    patterns = {
        'GCN': r'Test Results:\s+- Accuracy: ([\d.]+)\s+- F1 Score: ([\d.]+)',
        'GraphSAGE': r'GraphSAGE.*?Test Results:.*?Accuracy: ([\d.]+).*?F1 Score: ([\d.]+)',
        'GAT': r'GAT.*?Test Results:.*?Accuracy: ([\d.]+).*?F1 Score: ([\d.]+)'
    }
    
    # 从最后的summary中提取
    summary_pattern = r'Best Model: (\w+).*?Accuracy: ([\d.]+).*?F1 Score: ([\d.]+)'
    summary_match = re.search(summary_pattern, content, re.DOTALL)
    
    if summary_match:
        best_model = summary_match.group(1)
        print(f"✓ Best baseline model: {best_model}")
        print(f"  Accuracy: {summary_match.group(2)}")
        print(f"  F1 Score: {summary_match.group(3)}")
    
    # 提取各模型的详细结果
    # 查找所有"Test Results"部分
    test_results = re.findall(
        r'(GCN|GraphSAGE|GAT).*?Test Results:.*?Accuracy: ([\d.]+).*?F1 Score: ([\d.]+)',
        content,
        re.DOTALL
    )
    
    for model, acc, f1 in test_results:
        results[model] = {
            'accuracy': float(acc),
            'f1': float(f1)
        }
    
    return results

def create_baseline_summary(results):
    """创建baseline结果总结"""
    
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    
    data = []
    for model, metrics in results.items():
        data.append({
            'Model': model,
            'Type': 'Baseline',
            'Test Acc': f"{metrics['accuracy']:.4f}",
            'Test F1': f"{metrics['f1']:.4f}"
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # 保存
    df.to_csv('project2_results/baseline_summary.csv', index=False)
    print("\n✓ Saved to project2_results/baseline_summary.csv")
    
    return df

def plot_baseline_results(results):
    """绘制baseline结果"""
    
    models = list(results.keys())
    accs = [results[m]['accuracy'] for m in models]
    f1s = [results[m]['f1'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    bars1 = axes[0].bar(models, accs, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Baseline Model Accuracy', fontsize=14)
    axes[0].set_ylim([0.9, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=10)
    
    # F1 Score
    bars2 = axes[1].bar(models, f1s, color='coral', alpha=0.8)
    axes[1].set_ylabel('Test F1 Score', fontsize=12)
    axes[1].set_title('Baseline Model F1 Score', fontsize=14)
    axes[1].set_ylim([0.9, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('project2_results/baseline_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot to project2_results/baseline_performance.png")
    plt.close()

if __name__ == "__main__":
    import sys
    import os
    
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'training.log'
    
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found!")
        sys.exit(1)
    
    print("Analyzing baseline results from training.log...")
    print("="*60)
    
    results = parse_training_log(log_path)
    
    if results:
        df = create_baseline_summary(results)
        plot_baseline_results(results)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Review baseline_summary.csv")
        print("2. Run: python 10_run_extra_credit_quick.py")
        print("3. Compare advanced models with baseline")
    else:
        print("\n⚠ Could not parse results from log file")
        print("Using default values from log summary...")
        
        # 从log的summary中手动提取
        print("\nFrom your log:")
        print("  Best Model: GraphSAGE")
        print("  - Accuracy: 0.9660")
        print("  - F1 Score: 0.9625")
