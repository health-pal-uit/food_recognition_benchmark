import json
import os
import time
import matplotlib.pyplot as plt
from metrics import calculate_policy_metrics
import numpy as np
import pandas as pd

# 10 TARGET CLASSES DEFINED IN THESIS PROPOSAL
TARGET_CLASSES = [
    'banh_mi',
    'banh_trang_tron',
    'banh_xeo',
    'bun_bo_hue',
    'bun_dau',
    'com_tam',
    'goi_cuon',
    'pho',
    'hu_tieu',
    'xoi',
]


class FoodRecognitionBenchmark:
    def __init__(self, checkpoint_path, output_dir='outputs'):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.p1_conf_threshold = 0.65  # Confidence threshold selected from validation

    def _simulate_detector_inference(self, image_path, is_known=True):
        """Simulate YOLO detector inference pipeline.

        Replace this simulation logic with actual HTTP/gRPC API call when the AI
        service is ready.
        """
        start_time = time.time()

        if is_known:
            filename = os.path.basename(image_path)
            gt_class = filename.split('_')[0] if '_' in filename else TARGET_CLASSES[0]
            if gt_class not in TARGET_CLASSES:
                gt_class = TARGET_CLASSES[0]

            pred_class = gt_class
            confidence = float(np.random.uniform(0.55, 0.95))
        else:
            gt_class = 'unknown_or_non_food'
            pred_class = np.random.choice(
                TARGET_CLASSES + ['other_class'], p=[0.03] * 10 + [0.7]
            )
            confidence = float(np.random.uniform(0.30, 0.75))

        iou_luminance = float(np.random.uniform(0.60, 0.98))
        latency_ms = (time.time() - start_time) * 1000 + np.random.uniform(15, 35)

        return {
            'gt_class': gt_class,
            'pred_class': pred_class,
            'confidence': confidence,
            'iou_luminance': iou_luminance,
            'latency_ms': latency_ms,
        }

    def apply_policy(self, detection_res, policy_name):
        """Apply selective classification logic for P0, P1, and P2 policies."""
        pred_class = detection_res['pred_class']
        conf = detection_res['confidence']
        iou_lum = detection_res['iou_luminance']

        # P0 Policy (Baseline): Top candidate > Floor threshold (0.25)
        if policy_name == 'P0':
            if conf < 0.25:
                return 'No detection', None
            return ('Accepted', pred_class) if pred_class in TARGET_CLASSES else (
                'Uncertain',
                pred_class,
            )

        # P1 Policy: Floor threshold + Selected Confidence threshold
        elif policy_name == 'P1':
            if conf < self.p1_conf_threshold:
                return 'No detection', None
            return ('Accepted', pred_class) if pred_class in TARGET_CLASSES else (
                'Uncertain',
                pred_class,
            )

        # P2 Policy: P1 + Luminance stability check (Luminance +15%, IoU >= 0.75)
        elif policy_name == 'P2':
            if conf < self.p1_conf_threshold:
                return 'No detection', None
            if pred_class in TARGET_CLASSES:
                if iou_lum >= 0.75:
                    return 'Accepted', pred_class
                else:
                    return 'Uncertain', pred_class
            return 'Uncertain', pred_class

    def run_benchmark(self, known_dir, ood_dir):
        print(
            '=== STARTING HEALTHPAL FOOD RECOGNITION BENCHMARK EVALUATION ===\n'
        )
        policies = ['P0', 'P1', 'P2']
        final_summary = {}

        known_images = [f'pho_{i}.jpg' for i in range(100)]
        ood_images = [f'ood_{i}.jpg' for i in range(100)]

        for policy in policies:
            print(f'-> Evaluating Policy: {policy}...')
            policy_results = {'known': [], 'ood': []}

            for img in known_images:
                det = self._simulate_detector_inference(img, is_known=True)
                status, final_pred = self.apply_policy(det, policy)
                det['status'] = status
                det['final_pred'] = final_pred
                policy_results['known'].append(det)

        for img in ood_images:
            det = self._simulate_detector_inference(img, is_known=False)
            status, final_pred = self.apply_policy(det, policy)
            det['status'] = status
            det['final_pred'] = final_pred
            policy_results['ood'].append(det)

        metrics = calculate_policy_metrics(policy_results)
        final_summary[policy] = metrics

        self._export_results(final_summary)

    def _export_results(self, summary_data):
        df = pd.DataFrame(summary_data).T
        csv_path = os.path.join(self.output_dir, 'benchmark_summary.csv')
        df.to_csv(csv_path, encoding='utf-8-sig')

        json_path = os.path.join(self.output_dir, 'benchmark_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

        print('\n' + '=' * 70)
        print('BENCHMARK EVALUATION SUMMARY (P0 / P1 / P2)')
        print('=' * 70)
        print(df.to_string())
        print('=' * 70)
        print(f'\n[SUCCESS] CSV report saved to: {csv_path}')
        print(f'[SUCCESS] JSON report saved to: {json_path}')

        self._plot_charts(df)

    def _plot_charts(self, df):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        df['Known Coverage (%)'].plot(
            kind='bar', color=['#3498db', '#2ecc71', '#e74c3c']
        )
        plt.title('Known Coverage (%)')
        plt.ylabel('%')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        df['Selective Risk (%)'].plot(
            kind='bar', color=['#3498db', '#2ecc71', '#e74c3c']
        )
        plt.title('Selective Risk (%) - Lower is Better')
        plt.ylabel('%')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        chart_path = os.path.join(self.output_dir, 'benchmark_comparison_chart.png')
        plt.savefig(chart_path, dpi=300)
        print(f'[SUCCESS] Comparison chart saved to: {chart_path}\n')


    if __name__ == '__main__':
        benchmark = FoodRecognitionBenchmark(
            checkpoint_path='checkpoints/baseline_yolov8.pt'
        )
        benchmark.run_benchmark(
            known_dir='datasets/known_test', ood_dir='datasets/ood_test'
        )