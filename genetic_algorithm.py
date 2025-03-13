import numpy as np
import random
import time


def initialize_population(X_test, sensitive_features, population_size):
    """Create initial population of sample pairs with batch processing"""
    population = []
    # Sample indices with replacement
    indices = np.random.choice(len(X_test), size=population_size, replace=True)

    for idx in indices:
        base_sample = X_test.iloc[idx].copy().astype(float)
        variant = base_sample.copy()

        # Change a sensitive feature
        for feature in sensitive_features:
            unique_values = X_test[feature].unique()
            current_value = variant[feature]
            other_values = [v for v in unique_values if v != current_value]
            if other_values:
                variant[feature] = np.random.choice(other_values)

        population.append((base_sample, variant))
    return population


def batch_calculate_fitness(model, population, batch_size=32):
    """Calculate fitness for entire population in batches"""
    total_pairs = len(population)
    all_fitnesses = np.zeros(total_pairs)

    # Process in batches to avoid memory issues
    for i in range(0, total_pairs, batch_size):
        end_idx = min(i + batch_size, total_pairs)
        current_batch = population[i:end_idx]

        # Prepare batch data
        samples_a = np.array([list(pair[0]) for pair in current_batch])
        samples_b = np.array([list(pair[1]) for pair in current_batch])

        # Get predictions in batches
        preds_a = model.predict(samples_a, verbose=0)
        preds_b = model.predict(samples_b, verbose=0)

        # Calculate fitness (absolute difference)
        batch_fitnesses = np.abs(preds_a - preds_b).flatten()
        all_fitnesses[i:end_idx] = batch_fitnesses

    return all_fitnesses


def selection(population, fitnesses):
    """Select parents using tournament selection"""
    selected = []
    for _ in range(len(population)):
        # Tournament selection
        idx1, idx2 = random.sample(range(len(population)), 2)
        winner = idx1 if fitnesses[idx1] > fitnesses[idx2] else idx2
        selected.append(population[winner])
    return selected


def crossover(parent1, parent2, non_sensitive_features, sensitive_features, crossover_rate=0.7):
    """Perform crossover between two parent sample pairs"""
    if random.random() > crossover_rate:
        return parent1, parent2

    # Unpack parent pairs
    parent1_a, parent1_b = parent1
    parent2_a, parent2_b = parent2

    # Create children by crossing over non-sensitive features
    crossover_point = random.randint(0, len(non_sensitive_features))
    features_to_swap = non_sensitive_features[:crossover_point]

    child1_a, child1_b = parent1_a.copy(), parent1_b.copy()
    child2_a, child2_b = parent2_a.copy(), parent2_b.copy()

    for feature in features_to_swap:
        child1_a[feature] = parent2_a[feature]
        child1_b[feature] = parent2_b[feature]
        child2_a[feature] = parent1_a[feature]
        child2_b[feature] = parent1_b[feature]

    # Ensure consistent sensitive feature differences
    for feature in sensitive_features:
        child1_b[feature] = child1_a[feature] if parent1_b[feature] == parent1_a[feature] else child1_b[feature]
        child2_b[feature] = child2_a[feature] if parent2_b[feature] == parent2_a[feature] else child2_b[feature]

    return (child1_a, child1_b), (child2_a, child2_b)


def mutate(sample_pair, X_test, non_sensitive_features, mutation_rate=0.1):
    """Apply mutation to a sample pair"""
    sample_a, sample_b = sample_pair
    mutated_a = sample_a.copy()
    mutated_b = sample_b.copy()

    # Mutate non-sensitive features
    for feature in non_sensitive_features:
        if random.random() < mutation_rate:
            min_val = X_test[feature].min()
            max_val = X_test[feature].max()
            range_val = max_val - min_val

            # Apply same perturbation to both samples to keep them comparable
            perturbation = np.random.uniform(-0.2 * range_val, 0.2 * range_val)
            mutated_a[feature] = np.clip(mutated_a[feature] + perturbation, min_val, max_val)
            mutated_b[feature] = np.clip(mutated_b[feature] + perturbation, min_val, max_val)

    return (mutated_a, mutated_b)


def run_genetic_algorithm(model, X_test, sensitive_features, non_sensitive_features,
                          population_size=50, generations=100, mutation_rate=0.1,
                          crossover_rate=0.7, discrimination_threshold=0.01,
                          batch_size=32):
    """Run the genetic algorithm with batch processing and performance tracking"""
    # Initialize tracking
    discriminatory_pairs = set()
    unique_inputs = set()
    model_evaluations = 0
    performance_metrics = {
        'time': [],
        'discriminatory_pairs_found': [],
        'unique_inputs': [],
        'evaluations': []
    }
    start_time = time.time()

    # Initialize population
    population = initialize_population(X_test, sensitive_features, population_size)

    for generation in range(generations):
        gen_start = time.time()

        # Evaluate fitness in batches
        fitnesses = batch_calculate_fitness(model, population, batch_size)
        model_evaluations += 2 * len(population)  # 2 predictions per pair

        # Add discriminatory instances to results
        for pair, fitness in zip(population, fitnesses):
            sample_a, sample_b = pair
            sample_a_tuple = tuple(sample_a)
            sample_b_tuple = tuple(sample_b)

            unique_inputs.add(sample_a_tuple)
            unique_inputs.add(sample_b_tuple)

            if fitness > discrimination_threshold:
                discriminatory_pairs.add((sample_a_tuple, sample_b_tuple))

        # Record metrics for this generation
        current_time = time.time() - start_time
        performance_metrics['time'].append(current_time)
        performance_metrics['discriminatory_pairs_found'].append(len(discriminatory_pairs))
        performance_metrics['unique_inputs'].append(len(unique_inputs))
        performance_metrics['evaluations'].append(model_evaluations)

        # Status update
        if generation % 10 == 0:
            print(
                f"Generation {generation}: Found {len(discriminatory_pairs)} discriminatory pairs, {len(unique_inputs)} unique inputs")
            print(
                f"Current IDI ratio: {len(discriminatory_pairs) / max(1, len(unique_inputs)):.4f}, Time: {current_time:.2f}s")

        # Selection
        selected = selection(population, fitnesses)

        # Create next generation
        next_population = []

        # Elitism - keep best individual
        best_idx = np.argmax(fitnesses)
        next_population.append(population[best_idx])

        # Crossover and mutation
        for i in range(0, len(selected) - 1, 2):
            if i + 1 < len(selected):  # Make sure there's a pair
                child1, child2 = crossover(selected[i], selected[i + 1],
                                           non_sensitive_features, sensitive_features,
                                           crossover_rate)
                next_population.append(mutate(child1, X_test, non_sensitive_features, mutation_rate))
                if len(next_population) < population_size:
                    next_population.append(mutate(child2, X_test, non_sensitive_features, mutation_rate))

        population = next_population

    # Calculate final IDI ratio
    idi_ratio = len(discriminatory_pairs) / len(unique_inputs) if unique_inputs else 0
    total_time = time.time() - start_time
    print(f"Final IDI ratio: {idi_ratio:.4f}, Total time: {total_time:.2f}s, Total evaluations: {model_evaluations}")

    # Return additional metrics for comparison
    results = {
        'discriminatory_pairs': discriminatory_pairs,
        'unique_inputs': unique_inputs,
        'idi_ratio': idi_ratio,
        'total_evaluations': model_evaluations,
        'total_time': total_time,
        'performance_metrics': performance_metrics
    }

    return results

