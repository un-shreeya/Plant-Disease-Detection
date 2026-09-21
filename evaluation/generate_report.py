import os
import json

def generate_report():
    print("Generating comprehensive evaluation report...")
    
    # Load summary JSON
    summary_path = 'evaluation/outputs/evaluation_summary.json'
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Please run evaluate.py first.")
        return

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Load robustness JSON
    robust_path = 'evaluation/outputs/robustness_results.json'
    if not os.path.exists(robust_path):
        print(f"Error: {robust_path} not found. Please run evaluate.py first.")
        return
        
    with open(robust_path, 'r') as f:
        robustness = json.load(f)

    report_md = f"""# Plant Disease Detection Model - Final Evaluation Report

## 1. Executive Summary
The model evaluation demonstrates highly accurate performance across the 38 classes of the PlantVillage dataset.

- **Overall Accuracy**: {summary['accuracy']}%
- **Weighted Precision**: {summary['precision']}%
- **Weighted Recall**: {summary['recall']}%
- **Weighted F1-Score**: {summary['f1_score']}%
- **Average Inference Time**: {summary['inference_time_ms']} ms/image
- **Model Size**: {summary['model_size_mb']} MB

## 2. Robustness Testing
The model was tested against simulated real-world conditions (blur, poor lighting, sensor noise) to evaluate its reliability outside of controlled laboratory settings.

| Condition | Description | Accuracy | Drop from Baseline |
|-----------|-------------|----------|--------------------|
| Clean (Baseline) | Standard validation set | {robustness['clean']}% | 0.0% |
| Blurry | Gaussian blur (σ=3) | {robustness['blurry']}% | {(robustness['clean'] - robustness['blurry']):.2f}% |
| Dark | Brightness reduced by 60% | {robustness['dark']}% | {(robustness['clean'] - robustness['dark']):.2f}% |
| Noisy | Gaussian sensor noise | {robustness['noisy']}% | {(robustness['clean'] - robustness['noisy']):.2f}% |
| Low Res | 56x56 upscaled to 224x224 | {robustness['low_resolution']}% | {(robustness['clean'] - robustness['low_resolution']):.2f}% |

## 3. Visualizations Generated
The following visualizations have been saved in `evaluation/outputs/` for inclusion in the research paper:
1. `confusion_matrix.png`: Shows exact misclassifications between visually similar diseases.
2. `roc_curves.png`: One-vs-Rest AUC metrics for the top 10 classes.
3. `per_class_accuracy.png`: Bar chart identifying specific diseases the model struggles to identify.
4. `classification_report.txt`: Detailed raw statistical dump.

## 4. Conclusion
The EfficientNet-B0 backbone provides an excellent trade-off between accuracy and computational efficiency ({summary['inference_time_ms']} ms per image), making it suitable for edge deployment.
"""

    report_path = 'evaluation/final_report.md'
    os.makedirs('evaluation', exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_md)
        
    print(f"✅ Report generated successfully at {report_path}")

if __name__ == "__main__":
    generate_report()
