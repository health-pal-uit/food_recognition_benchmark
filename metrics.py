import numpy as np


def calculate_policy_metrics(results):
    """Calculate selective classification metrics for P0/P1/P2 policies.

    Args:
        results (dict): Dictionary containing 'known' and 'ood' evaluation
        results.

    Returns:
        dict: Calculated performance and latency metrics.
    """
    total_known = len(results['known'])
    total_ood = len(results['ood'])

    # 1. Known Dataset Evaluation
    known_accepted = [r for r in results['known'] if r['status'] == 'Accepted']
    known_coverage = (
        len(known_accepted) / total_known if total_known > 0 else 0.0
    )

    known_correct = [
        r for r in known_accepted if r['pred_class'] == r['gt_class']
    ]
    known_incorrect = len(known_accepted) - len(known_correct)
    selective_risk = (
        known_incorrect / len(known_accepted)
        if len(known_accepted) > 0
        else 0.0
    )

    correct_and_accepted_rate = (
        len(known_correct) / total_known if total_known > 0 else 0.0
    )

    # 2. Unknown / OOD Dataset Evaluation
    ood_rejected = [
        r for r in results['ood'] if r['status'] in ['Uncertain', 'No detection']
    ]
    ood_rejection_rate = len(ood_rejected) / total_ood if total_ood > 0 else 0.0

    ood_false_accepted = total_ood - len(ood_rejected)
    ood_false_accept_rate = (
        ood_false_accepted / total_ood if total_ood > 0 else 0.0
    )

    # 3. Latency Evaluation
    latencies = [r['latency_ms'] for r in results['known'] + results['ood']]
    p50_latency = (
        float(np.percentile(latencies, 50)) if len(latencies) > 0 else 0.0
    )
    p95_latency = (
        float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0
    )

    return {
        'Known Coverage (%)': round(known_coverage * 100, 2),
        'Selective Risk (%)': round(selective_risk * 100, 2),
        'Correct & Accepted (%)': round(correct_and_accepted_rate * 100, 2),
        'OOD Rejection Rate (%)': round(ood_rejection_rate * 100, 2),
        'OOD False Accept Rate (%)': round(ood_false_accept_rate * 100, 2),
        'p50 Latency (ms)': round(p50_latency, 2),
        'p95 Latency (ms)': round(p95_latency, 2),
    }