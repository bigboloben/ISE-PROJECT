import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time


def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def batch_generate_sample_pairs(X_test, sensitive_columns, non_sensitive_columns, batch_size):
    """Generate multiple sample pairs at once for batch processing"""
    samples_a = []
    samples_b = []

    # Select random indices
    indices = np.random.choice(len(X_test), size=batch_size, replace=True)

    for idx in indices:
        sample_a = X_test.iloc[idx].copy()
        sample_b = sample_a.copy()

        sample_a = sample_a.astype(float)
        sample_b = sample_b.astype(float)

        # Apply perturbation on sensitive features
        for col in sensitive_columns:
            if col in X_test.columns:
                unique_values = X_test[col].unique()
                sample_b[col] = np.random.choice(unique_values)

        # Apply perturbation on non-sensitive features
        for col in non_sensitive_columns:
            if col in X_test.columns:
                min_val = X_test[col].min()
                max_val = X_test[col].max()
                perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
                sample_a[col] = np.clip(sample_a[col] + perturbation, min_val, max_val)
                sample_b[col] = np.clip(sample_b[col] + perturbation, min_val, max_val)

        samples_a.append(sample_a)
        samples_b.append(sample_b)

    return np.array([list(s) for s in samples_a]), np.array([list(s) for s in samples_b])


def batch_evaluate_discrimination(model, samples_a, samples_b, threshold=0.01):
    """Evaluate discrimination in batches for efficiency"""
    # Model predictions
    predictions_a = model.predict(samples_a, verbose=0)
    predictions_b = model.predict(samples_b, verbose=0)

    # Calculate differences
    prediction_diffs = np.abs(predictions_a - predictions_b).flatten()

    # Count discriminatory instances
    discriminatory_mask = prediction_diffs > threshold
    discriminatory_count = np.sum(discriminatory_mask)

    # Return both the count and the mask
    return discriminatory_count, discriminatory_mask


def calculate_idi_ratio(model, X_test, sensitive_columns, non_sensitive_columns,
                        total_evaluations, batch_size=32, threshold=0.01):
    """Calculate IDI ratio with batch processing and tracking metrics"""
    start_time = time.time()
    discrimination_count = 0
    processed_samples = 0
    unique_inputs = set()
    discriminatory_pairs = set()

    # Track performance over time
    performance_metrics = {
        'time': [],
        'discriminatory_pairs_found': [],
        'unique_inputs': [],
        'evaluations': []
    }

    # Calculate number of batches needed to match total_evaluations
    remaining_evaluations = total_evaluations

    while remaining_evaluations > 0:
        # Adjust batch size for last iteration if needed
        current_batch_size = min(batch_size, remaining_evaluations // 2)
        if current_batch_size <= 0:
            break

        # Generate sample pairs
        samples_a, samples_b = batch_generate_sample_pairs(
            X_test, sensitive_columns, non_sensitive_columns, current_batch_size
        )

        # Store unique inputs
        for i in range(current_batch_size):
            sample_a_tuple = tuple(samples_a[i])
            sample_b_tuple = tuple(samples_b[i])
            unique_inputs.add(sample_a_tuple)
            unique_inputs.add(sample_b_tuple)

        # Evaluate discrimination
        disc_count, disc_mask = batch_evaluate_discrimination(model, samples_a, samples_b, threshold)
        discrimination_count += disc_count

        # Store discriminatory pairs
        for i in range(current_batch_size):
            if disc_mask[i]:
                sample_a_tuple = tuple(samples_a[i])
                sample_b_tuple = tuple(samples_b[i])
                discriminatory_pairs.add((sample_a_tuple, sample_b_tuple))

        # Update processed samples and remaining evaluations
        processed_samples += current_batch_size
        remaining_evaluations -= 2 * current_batch_size  # 2 evaluations per pair

        # Record metrics periodically
        if processed_samples % (10 * batch_size) == 0 or remaining_evaluations <= 0:
            current_time = time.time() - start_time
            current_evaluations = total_evaluations - remaining_evaluations

            performance_metrics['time'].append(current_time)
            performance_metrics['discriminatory_pairs_found'].append(len(discriminatory_pairs))
            performance_metrics['unique_inputs'].append(len(unique_inputs))
            performance_metrics['evaluations'].append(current_evaluations)

            print(f"Processed {processed_samples} samples, Found {len(discriminatory_pairs)} discriminatory pairs")
            print(
                f"Current IDI ratio: {len(discriminatory_pairs) / max(1, len(unique_inputs)):.4f}, Time: {current_time:.2f}s")

    # Calculate final metrics
    total_time = time.time() - start_time
    idi_ratio = len(discriminatory_pairs) / len(unique_inputs) if unique_inputs else 0

    print(f"Random Search IDI Ratio: {idi_ratio:.4f}, Total time: {total_time:.2f}s")
    print(f"Total evaluations: {total_evaluations - remaining_evaluations}, Total samples: {processed_samples}")

    # Return results in the same format as the GA for fair comparison
    results = {
        'discriminatory_pairs': discriminatory_pairs,
        'unique_inputs': unique_inputs,
        'idi_ratio': idi_ratio,
        'total_evaluations': total_evaluations - remaining_evaluations,
        'total_time': total_time,
        'performance_metrics': performance_metrics
    }

    return results