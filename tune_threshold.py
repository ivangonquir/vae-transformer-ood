"""
Threshold Tuning and Analysis
==============================

This script helps you find the optimal threshold for anomaly detection
and diagnose why you're getting lots of False Negatives.

Usage:
    python tune_threshold.py --dataset sms_spam --output_dir ./output
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
import argparse
import os
import json


def load_results(dataset_name, output_dir):
    """Load cached results from previous run"""
    
    results_path = os.path.join(output_dir, 'results', f'{dataset_name}_results.json')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results found for {dataset_name}. Run experiment first.")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def compute_errors_from_embeddings(dataset_name, output_dir, model_path=None):
    """
    Recompute reconstruction errors if we have a saved model
    """
    # For now, we'll use a simpler approach - just analyze different thresholds
    # on the error distributions we can infer from the results
    
    # This is a placeholder - in practice, you'd load the actual model and embeddings
    pass


def analyze_threshold_sensitivity(dataset_name, output_dir):
    """
    Analyze how performance changes with different thresholds
    """
    
    print("\n" + "="*70)
    print(f"THRESHOLD ANALYSIS: {dataset_name}")
    print("="*70)
    
    results = load_results(dataset_name, output_dir)
    
    # Get error statistics
    normal_mean = results['normal_error_mean']
    normal_std = results['normal_error_std']
    anomaly_mean = results['anomaly_error_mean']
    anomaly_std = results['anomaly_error_std']
    current_threshold = results['threshold']
    
    print(f"\nCurrent Configuration:")
    print(f"  Threshold: {current_threshold:.6f}")
    print(f"  F1-Score: {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    
    print(f"\nError Distributions:")
    print(f"  Normal:  {normal_mean:.6f} ± {normal_std:.6f}")
    print(f"  Anomaly: {anomaly_mean:.6f} ± {anomaly_std:.6f}")
    print(f"  Separation: {(anomaly_mean - normal_mean) / normal_std:.2f}σ")
    
    # Simulate different thresholds
    print(f"\n" + "-"*70)
    print("SIMULATED PERFORMANCE AT DIFFERENT THRESHOLDS:")
    print("-"*70)
    print(f"{'Threshold':<12} {'Method':<20} {'Est. Precision':<15} {'Est. Recall':<15} {'Est. F1':<10}")
    print("-"*70)
    
    # Assume normal errors ~ N(normal_mean, normal_std)
    # Assume anomaly errors ~ N(anomaly_mean, anomaly_std)
    
    from scipy.stats import norm
    
    thresholds_to_test = [
        (normal_mean + 0.5 * normal_std, "Mean + 0.5σ"),
        (normal_mean + 1.0 * normal_std, "Mean + 1.0σ"),
        (normal_mean + 1.5 * normal_std, "Mean + 1.5σ"),
        (normal_mean + 2.0 * normal_std, "Mean + 2.0σ"),
        (normal_mean + 2.5 * normal_std, "Mean + 2.5σ"),
        (np.percentile([normal_mean] * 100, 90), "90th percentile"),
        (np.percentile([normal_mean] * 100, 95), "95th percentile"),
        (np.percentile([normal_mean] * 100, 99), "99th percentile"),
        ((normal_mean + anomaly_mean) / 2, "Midpoint"),
    ]
    
    best_f1 = 0
    best_threshold = None
    best_method = None
    
    for threshold, method in thresholds_to_test:
        # Estimate false positive rate (normal flagged as anomaly)
        fpr = 1 - norm.cdf(threshold, normal_mean, normal_std)
        
        # Estimate true positive rate (anomaly correctly flagged)
        tpr = 1 - norm.cdf(threshold, anomaly_mean, anomaly_std)
        
        # Estimate precision and recall (assuming balanced classes for now)
        if (tpr + fpr) > 0:
            precision = tpr / (tpr + fpr) if (tpr + fpr) > 0 else 0
            recall = tpr
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0
        
        print(f"{threshold:<12.6f} {method:<20} {precision:<15.4f} {recall:<15.4f} {f1:<10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_method = method
    
    print("-"*70)
    print(f"\n✅ RECOMMENDED:")
    print(f"   Threshold: {best_threshold:.6f} ({best_method})")
    print(f"   Expected F1: {best_f1:.4f}")
    print(f"   Current F1: {results['f1']:.4f}")
    
    if best_f1 > results['f1']:
        improvement = ((best_f1 - results['f1']) / results['f1']) * 100
        print(f"   💡 Potential improvement: +{improvement:.1f}%")
    
    # Analyze current performance
    print(f"\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    cm = results['confusion_matrix']
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:4d} (normal correctly identified)")
    print(f"  False Positives: {fp:4d} (normal wrongly flagged) ⚠️")
    print(f"  False Negatives: {fn:4d} (anomaly missed) ⚠️")
    print(f"  True Positives:  {tp:4d} (anomaly correctly detected)")
    
    total_anomalies = tp + fn
    total_normal = tn + fp
    
    print(f"\nBreakdown:")
    print(f"  Of {total_normal} normal samples:")
    print(f"    {tn} ({100*tn/total_normal:.1f}%) correctly classified as normal")
    print(f"    {fp} ({100*fp/total_normal:.1f}%) wrongly flagged as anomaly")
    
    print(f"\n  Of {total_anomalies} anomaly samples:")
    print(f"    {tp} ({100*tp/total_anomalies:.1f}%) correctly detected")
    print(f"    {fn} ({100*fn/total_anomalies:.1f}%) missed (classified as normal) ⚠️")
    
    # Diagnosis
    print(f"\n💡 Problem Analysis:")
    
    if fn > tp:
        print(f"  ❌ HIGH FALSE NEGATIVES: Missing {fn} out of {total_anomalies} anomalies!")
        print(f"     → Threshold is TOO HIGH (too lenient)")
        print(f"     → Current: {current_threshold:.6f}")
        print(f"     → Try: {best_threshold:.6f} ({best_method})")
        print(f"     → This will catch more anomalies but increase false alarms")
    
    if fp > 0.1 * total_normal:
        print(f"  ⚠️  HIGH FALSE POSITIVES: Flagging {fp} normal samples")
        print(f"     → Threshold might be too low (too strict)")
    
    if tp < 0.5 * total_anomalies:
        print(f"  ⚠️  LOW RECALL: Only detecting {100*tp/total_anomalies:.1f}% of anomalies")
        print(f"     → VAE is not learning good representations")
        print(f"     → Semantic distance might be too low")
        print(f"     → Consider: longer training, different architecture, or different method")
    
    separation = (anomaly_mean - normal_mean) / normal_std
    if separation < 2.0:
        print(f"  ⚠️  LOW SEPARATION: Only {separation:.2f}σ between classes")
        print(f"     → Error distributions overlap significantly")
        print(f"     → Perfect separation is impossible with this data")
        print(f"     → Expected F1 ceiling: ~{min(0.70, 0.4 + separation*0.15):.2f}")
    
    return best_threshold, best_method


def create_threshold_analysis_plot(dataset_name, output_dir):
    """
    Create visualization showing how metrics change with threshold
    """
    
    results = load_results(dataset_name, output_dir)
    
    normal_mean = results['normal_error_mean']
    normal_std = results['normal_error_std']
    anomaly_mean = results['anomaly_error_mean']
    anomaly_std = results['anomaly_error_std']
    current_threshold = results['threshold']
    
    # Generate threshold range
    min_threshold = normal_mean - normal_std
    max_threshold = anomaly_mean + anomaly_std
    thresholds = np.linspace(min_threshold, max_threshold, 100)
    
    from scipy.stats import norm
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        fpr = 1 - norm.cdf(threshold, normal_mean, normal_std)
        tpr = 1 - norm.cdf(threshold, anomaly_mean, anomaly_std)
        
        precision = tpr / (tpr + fpr) if (tpr + fpr) > 0 else 0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Metrics vs Threshold
    ax1.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    ax1.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
    ax1.plot(thresholds, f1_scores, 'g-', linewidth=2.5, label='F1-Score')
    
    # Mark current threshold
    ax1.axvline(current_threshold, color='black', linestyle='--', linewidth=2, 
               label=f'Current ({current_threshold:.6f})')
    
    # Mark optimal threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    ax1.axvline(best_threshold, color='purple', linestyle='--', linewidth=2,
               label=f'Optimal ({best_threshold:.6f})')
    
    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'{dataset_name}: Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Error distributions with thresholds
    x = np.linspace(min_threshold - normal_std, max_threshold + anomaly_std, 200)
    
    normal_dist = norm.pdf(x, normal_mean, normal_std)
    anomaly_dist = norm.pdf(x, anomaly_mean, anomaly_std)
    
    ax2.fill_between(x, normal_dist, alpha=0.5, color='green', label='Normal errors')
    ax2.fill_between(x, anomaly_dist, alpha=0.5, color='red', label='Anomaly errors')
    
    ax2.axvline(current_threshold, color='black', linestyle='--', linewidth=2,
               label='Current threshold')
    ax2.axvline(best_threshold, color='purple', linestyle='--', linewidth=2,
               label='Optimal threshold')
    
    ax2.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'{dataset_name}: Error Distributions', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'images', f'{dataset_name}_threshold_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nThreshold analysis plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    try:
        best_threshold, best_method = analyze_threshold_sensitivity(args.dataset, args.output_dir)
        create_threshold_analysis_plot(args.dataset, args.output_dir)
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        print(f"\n1. Update your threshold:")
        print(f"   detector.threshold = {best_threshold:.6f}  # {best_method}")
        
        print(f"\n2. Or modify the percentile in your code:")
        print(f"   detector.set_threshold(train_data, percentile=85)  # Instead of 95")
        
        print(f"\n3. Check the visualization:")
        print(f"   {args.output_dir}/images/{args.dataset}_threshold_analysis.png")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure you've run the experiment first:")
        print(f"  python vae_text_anomaly.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()