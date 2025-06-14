import numpy as np

def compute_aurc(y_true, y_pred, y_confidence):
    """Compute the Area Under the Risk-Coverage Curve (AURC)."""
     # Convert to smaller data types for memory efficiency
    y_true = np.array(y_true, dtype=np.int8)
    y_pred = np.array(y_pred, dtype=np.int8)
    y_confidence = np.array(y_confidence, dtype=np.float32)
    
    sorted_indices = np.argsort(-y_confidence)  # Sort by confidence (descending)

    total_samples = len(y_true)
    risk_list = np.zeros(total_samples, dtype=np.float32)
    coverage_list = np.zeros(total_samples, dtype=np.float32)
    incorrect_count = 0

    # Compute in batches
    batch_size = 1000
    for i in range(0, total_samples, batch_size):
        end = min(i + batch_size, total_samples)
        batch_indices = sorted_indices[i:end]
        
        # Results for the current batch
        for j, idx in enumerate(batch_indices):
            pos = i + j
            if y_pred[idx] != y_true[idx]:
                incorrect_count += 1
            risk_list[pos] = incorrect_count / (pos + 1)
            coverage_list[pos] = (pos + 1) / total_samples
    
    aurc = np.trapz(risk_list, coverage_list)
    
    return aurc