import argparse
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import genetic_algorithm
import random_search_baseline


def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def load_model(model_path):
    return keras.models.load_model(model_path)


def run_multiple_trials(model, X_test, sensitive_columns, non_sensitive_columns, num_trials=10,
                        population_size=100, generations=50, batch_size=32):
    """
    Run multiple trials of both algorithms with fair comparison metrics
    """
    ga_results = []
    rs_results = []
    ga_times = []
    rs_times = []

    # Additional metrics for more comprehensive comparison
    ga_efficiency = []  # Discriminatory pairs per evaluation
    rs_efficiency = []
    ga_metrics_history = []
    rs_metrics_history = []

    for i in range(num_trials):
        print(f"Running trial {i + 1}/{num_trials}")

        # Calculate the expected total evaluations for the GA
        expected_evaluations = 2 * population_size * generations

        # Run genetic algorithm
        ga_result = genetic_algorithm.run_genetic_algorithm(
            model=model,
            X_test=X_test,
            sensitive_features=sensitive_columns,
            non_sensitive_features=non_sensitive_columns,
            population_size=population_size,
            generations=generations,
            mutation_rate=0.1,
            crossover_rate=0.7,
            batch_size=batch_size
        )

        # Extract results
        ga_metrics_history.append(ga_result['performance_metrics'])
        ga_results.append(ga_result['idi_ratio'])
        ga_times.append(ga_result['total_time'])
        ga_efficiency.append(len(ga_result['discriminatory_pairs']) / max(1, ga_result['total_evaluations']))

        # Run random search with equal computational budget
        rs_result = random_search_baseline.calculate_idi_ratio(
            model,
            X_test,
            sensitive_columns,
            non_sensitive_columns,
            total_evaluations=expected_evaluations,
            batch_size=batch_size
        )

        # Extract results
        rs_metrics_history.append(rs_result['performance_metrics'])
        rs_results.append(rs_result['idi_ratio'])
        rs_times.append(rs_result['total_time'])
        rs_efficiency.append(len(rs_result['discriminatory_pairs']) / max(1, rs_result['total_evaluations']))

    return {
        'ga_results': ga_results,
        'rs_results': rs_results,
        'ga_times': ga_times,
        'rs_times': rs_times,
        'ga_efficiency': ga_efficiency,
        'rs_efficiency': rs_efficiency,
        'ga_metrics_history': ga_metrics_history,
        'rs_metrics_history': rs_metrics_history
    }


def perform_statistical_analysis(trial_results):
    """
    Perform comprehensive statistical analysis on trial results
    """
    ga_results = trial_results['ga_results']
    rs_results = trial_results['rs_results']
    ga_times = trial_results['ga_times']
    rs_times = trial_results['rs_times']
    ga_efficiency = trial_results['ga_efficiency']
    rs_efficiency = trial_results['rs_efficiency']

    # Calculate descriptive statistics
    results = {
        'ga_median': np.median(ga_results),
        'rs_median': np.median(rs_results),
        'ga_q1': np.percentile(ga_results, 25),
        'ga_q3': np.percentile(ga_results, 75),
        'rs_q1': np.percentile(rs_results, 25),
        'rs_q3': np.percentile(rs_results, 75),
        'ga_mean_time': np.mean(ga_times),
        'rs_mean_time': np.mean(rs_times),
        'ga_mean_efficiency': np.mean(ga_efficiency),
        'rs_mean_efficiency': np.mean(rs_efficiency)
    }

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(ga_results, ddof=1) ** 2 + np.std(rs_results, ddof=1) ** 2) / 2)
    results['effect_size'] = (np.mean(ga_results) - np.mean(rs_results)) / pooled_std if pooled_std > 0 else float(
        'inf')

    print(f"==== IDI Ratio Analysis ====")
    print(
        f"Genetic Algorithm: Median IDI Ratio = {results['ga_median']:.4f}, Q1 = {results['ga_q1']:.4f}, Q3 = {results['ga_q3']:.4f}")
    print(
        f"Random Search: Median IDI Ratio = {results['rs_median']:.4f}, Q1 = {results['rs_q1']:.4f}, Q3 = {results['rs_q3']:.4f}")
    print(f"Improvement: {((results['ga_median'] - results['rs_median']) / results['rs_median']) * 100:.2f}%")
    print(f"Effect size (Cohen's d): {results['effect_size']:.4f}")

    print(f"\n==== Time Efficiency Analysis ====")
    print(f"Genetic Algorithm: Mean time = {results['ga_mean_time']:.2f}s")
    print(f"Random Search: Mean time = {results['rs_mean_time']:.2f}s")
    print(f"Time ratio (GA/RS): {results['ga_mean_time'] / results['rs_mean_time']:.2f}x")

    print(f"\n==== Discovery Efficiency Analysis ====")
    print(f"Genetic Algorithm: Mean discriminatory pairs per evaluation = {results['ga_mean_efficiency']:.6f}")
    print(f"Random Search: Mean discriminatory pairs per evaluation = {results['rs_mean_efficiency']:.6f}")
    print(f"Efficiency ratio (GA/RS): {results['ga_mean_efficiency'] / results['rs_mean_efficiency']:.2f}x")

    # Use Wilcoxon signed-rank test for IDI ratio
    stat, p_value = stats.wilcoxon(ga_results, rs_results)
    results['statistic'] = stat
    results['p_value'] = p_value

    print(f"\n==== Statistical Test Results ====")
    print(f"Wilcoxon signed-rank test (IDI ratio):")
    print(f"Statistic: {stat:.4e}")  # Using scientific notation
    print(f"p-value: {p_value:.4e}")  # Using scientific notation

    alpha = 0.05
    if p_value < alpha:
        print(f"The difference between the approaches is statistically significant (p < {alpha})")
    else:
        print(f"The difference between the approaches is not statistically significant (p >= {alpha})")

    # Also test efficiency metrics
    stat_eff, p_value_eff = stats.wilcoxon(ga_efficiency, rs_efficiency)
    results['eff_statistic'] = stat_eff
    results['eff_p_value'] = p_value_eff

    print(f"\nWilcoxon signed-rank test (Efficiency):")
    print(f"Statistic: {stat_eff:.4e}")  # Using scientific notation
    print(f"p-value: {p_value_eff:.4e}")  # Using scientific notation

    if p_value_eff < alpha:
        print(f"The efficiency difference is statistically significant (p < {alpha})")
    else:
        print(f"The efficiency difference is not statistically significant (p >= {alpha})")

    return results


def create_comparison_table(trial_results, dataset_name):
    """
    Create a clean table for presenting comparison results between GA and Random Search
    """
    ga_results = trial_results['ga_results']
    rs_results = trial_results['rs_results']

    # Calculate improvement percentage
    improvement = ((np.median(ga_results) - np.median(rs_results)) / np.median(rs_results)) * 100

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(ga_results, ddof=1) ** 2 + np.std(rs_results, ddof=1) ** 2) / 2)
    effect_size = (np.mean(ga_results) - np.mean(rs_results)) / pooled_std if pooled_std > 0 else float('inf')

    # Run Wilcoxon test
    stat, p_value = stats.wilcoxon(ga_results, rs_results)

    # Format p-value with scientific notation
    p_value_str = f"{p_value:.4e}"
    stat_str = f"{stat:.4e}"

    # Is it statistically significant at α = 0.05?
    is_significant = "Yes" if p_value < 0.05 else "No"

    result = {
        "Dataset": dataset_name,
        "GA Median IDI": f"{np.median(ga_results):.4f}",
        "RS Median IDI": f"{np.median(rs_results):.4f}",
        "Improvement (%)": f"{improvement:.2f}%",
        "Effect Size": f"{effect_size:.2f}",
        "Statistic": stat_str,
        "p-value": p_value_str,
        "Statistically Significant": is_significant
    }

    return result


def create_summary_table(all_datasets_results):
    """
    Create a summary table for all datasets
    """
    return pd.DataFrame(all_datasets_results)


def plot_simplified_comparison(all_results, save_path="results/simplified_comparison.png"):
    """
    Create a simplified visualization showing the key results across all datasets
    """
    # Convert to DataFrame if not already
    if not isinstance(all_results, pd.DataFrame):
        results_df = pd.DataFrame(all_results)
    else:
        results_df = all_results.copy()

    # Convert string percentages to float
    results_df['Improvement (%)'] = results_df['Improvement (%)'].str.rstrip('%').astype(float)

    # Sort datasets by improvement percentage
    results_df = results_df.sort_values('Improvement (%)', ascending=False)

    # Create the figure
    plt.figure(figsize=(14, 8))

    # Create a bar chart for improvement percentage
    plt.subplot(1, 2, 1)

    # Set colors based on statistical significance
    colors = ['#1f77b4' if sig == 'Yes' else '#d3d3d3' for sig in results_df['Statistically Significant']]

    # Plot improvement
    bars = plt.bar(results_df['Dataset'], results_df['Improvement (%)'], color=colors)

    # Add labels
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.ylabel('Improvement (%)')
    plt.title('GA vs Random Search: Improvement by Dataset')
    plt.xticks(rotation=45, ha='right')

    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Statistically Significant'),
        Patch(facecolor='#d3d3d3', label='Not Significant')
    ]
    plt.legend(handles=legend_elements)

    # Create a comparison scatter plot
    plt.subplot(1, 2, 2)

    # Convert IDI values to float
    ga_values = [float(val) for val in results_df['GA Median IDI']]
    rs_values = [float(val) for val in results_df['RS Median IDI']]

    # Plot the values
    for i, dataset in enumerate(results_df['Dataset']):
        plt.scatter(rs_values[i], ga_values[i], label=dataset, s=100)

    # Add diagonal line for reference
    max_val = max(max(ga_values), max(rs_values))
    min_val = min(min(ga_values), min(rs_values))
    padding = (max_val - min_val) * 0.05  # 5% padding
    plt.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'r--', alpha=0.5)

    plt.xlabel('Random Search IDI')
    plt.ylabel('Genetic Algorithm IDI')
    plt.title('GA vs Random Search: IDI Comparison')

    # Add text annotations for each point
    for i, dataset in enumerate(results_df['Dataset']):
        plt.annotate(dataset, (rs_values[i], ga_values[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Simplified visualization saved to {save_path}")

    return plt


def plot_comprehensive_results(trial_results, stats_results, dataset):
    """
    Create comprehensive visualizations comparing the approaches
    """
    ga_results = trial_results['ga_results']
    rs_results = trial_results['rs_results']
    ga_times = trial_results['ga_times']
    rs_times = trial_results['rs_times']
    ga_efficiency = trial_results['ga_efficiency']
    rs_efficiency = trial_results['rs_efficiency']

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f'Comprehensive Comparison: Genetic Algorithm vs Random Search - {dataset}',
                 fontsize=16, y=0.98)

    # Plot 1: IDI Ratio comparison (box plot)
    plt.subplot(2, 3, 1)
    box = plt.boxplot([ga_results, rs_results], labels=['Genetic Algorithm', 'Random Search'],
                      patch_artist=True)
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('IDI Ratio Comparison')
    plt.ylabel('IDI Ratio')

    # Plot 2: Runtime comparison (box plot)
    plt.subplot(2, 3, 2)
    box = plt.boxplot([ga_times, rs_times], labels=['Genetic Algorithm', 'Random Search'],
                      patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Runtime Comparison')
    plt.ylabel('Time (seconds)')

    # Plot 3: Efficiency comparison (box plot)
    plt.subplot(2, 3, 3)
    box = plt.boxplot([ga_efficiency, rs_efficiency], labels=['GA', 'RS'],
                      patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Discovery Efficiency')
    plt.ylabel('Discriminatory pairs / evaluation')

    # Plot 4: Paired IDI comparison
    plt.subplot(2, 3, 4)
    for i in range(len(ga_results)):
        plt.plot([1, 2], [ga_results[i], rs_results[i]], 'ro-', alpha=0.3)
    plt.xticks([1, 2], ['Genetic Algorithm', 'Random Search'])
    plt.ylabel('IDI Ratio')
    plt.title('Paired Trial Comparison')

    # Plot 5: IDI Ratio vs. Evaluations (performance over time from histories)
    plt.subplot(2, 3, 5)

    # Average the metrics across trials
    if trial_results['ga_metrics_history'] and trial_results['rs_metrics_history']:
        # Get the first trial's data for plotting
        ga_history = trial_results['ga_metrics_history'][0]
        rs_history = trial_results['rs_metrics_history'][0]

        plt.plot(ga_history['evaluations'],
                 [d / max(1, u) for d, u in zip(ga_history['discriminatory_pairs_found'], ga_history['unique_inputs'])],
                 'b-', label='GA')
        plt.plot(rs_history['evaluations'],
                 [d / max(1, u) for d, u in zip(rs_history['discriminatory_pairs_found'], rs_history['unique_inputs'])],
                 'g-', label='RS')

        plt.xlabel('Model Evaluations')
        plt.ylabel('IDI Ratio')
        plt.title('Performance vs. Computation')
        plt.legend()

    # Plot 6: Distribution of differences
    plt.subplot(2, 3, 6)
    differences = [ga - rs for ga, rs in zip(ga_results, rs_results)]
    plt.hist(differences, bins=10, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Differences (GA - RS)')
    plt.xlabel('Difference in IDI Ratio')
    plt.ylabel('Frequency')

    # Add statistical test results
    plt.figtext(0.5, 0.01,
                f"Wilcoxon test (IDI): p-value = {stats_results['p_value']:.4e}, Effect size = {stats_results['effect_size']:.2f}\n" +
                f"Efficiency: GA = {stats_results['ga_mean_efficiency']:.6f}, RS = {stats_results['rs_mean_efficiency']:.6f}, " +
                f"Ratio = {stats_results['ga_mean_efficiency'] / stats_results['rs_mean_efficiency']:.2f}x",
                ha='center', fontsize=12, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_location = f'results/comprehensive_comparison_{dataset}.png'
    plt.savefig(save_location)
    print(f"Visualization saved to {save_location}")


def main(file_paths, targets, sensitive_columns_list, num_trials=3, population_size=50, generations=20):
    """Main function to run the comparison across all datasets"""
    all_results = []

    for i in range(len(file_paths)):
        file_path = f"dataset/{file_paths[i]}.csv"
        model_path = f"DNN/model_{file_paths[i]}.h5"
        target = targets[i]
        sensitive_columns = sensitive_columns_list[i]

        print(f"\n{'=' * 20} Dataset: {file_paths[i]} {'=' * 20}")

        # Load data and model
        print(f"Loading data and model...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target)
        model = load_model(model_path)

        # Define non-sensitive columns
        non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

        # Run comparison trials
        print(f"Running comparison trials...")
        trial_results = run_multiple_trials(
            model, X_test, sensitive_columns, non_sensitive_columns,
            num_trials=num_trials, population_size=population_size, generations=generations
        )

        # Analyze results
        print(f"\nPerforming statistical analysis...")
        stats_results = perform_statistical_analysis(trial_results)

        # Create table row for this dataset
        dataset_result = create_comparison_table(trial_results, file_paths[i])
        all_results.append(dataset_result)

        # Generate visualizations
        print(f"\nGenerating visualizations...")
        plot_comprehensive_results(trial_results, stats_results, file_paths[i])

    # Create and save summary table
    summary_table = create_summary_table(all_results)
    print("\n===== Summary Results =====")
    print(summary_table)

    # Save to CSV
    summary_table.to_csv("results/summary_comparison.csv", index=False)

    # Generate simplified comparison visualization
    plot_simplified_comparison(summary_table, save_path="results/simplified_comparison.png")

    return summary_table


if __name__ == "__main__":
    if __name__ == "__main__":
        # Set up command-line argument parsing
        parser = argparse.ArgumentParser(description='AI Model Fairness Testing Tool')
        parser.add_argument('--trials', type=int, default=30, help='Number of trials to run (default: 30)')
        parser.add_argument('--population', type=int, default=50, help='Population size for GA (default: 50)')
        parser.add_argument('--generations', type=int, default=100, help='Number of generations for GA (default: 100)')
        parser.add_argument('--dataset', type=str, help='Run only on a specific dataset (optional)')
        args = parser.parse_args()

        # Configure datasets
        file_paths = []
        targets = []
        sensitive_columns_list = []

        file_paths.append('processed_adult')
        targets.append('Class-label')
        sensitive_columns_list.append(['race', 'gender', 'age'])

        file_paths.append('processed_communities_crime')
        targets.append('class')
        sensitive_columns_list.append(['Black', 'FemalePctDiv'])

        file_paths.append('processed_compas')
        targets.append('Recidivism')
        sensitive_columns_list.append(['Sex', 'Race'])

        file_paths.append('processed_credit')
        targets.append('class')
        sensitive_columns_list.append(['SEX', 'AGE', 'MARRIAGE'])

        file_paths.append('processed_dutch')
        targets.append('occupation')
        sensitive_columns_list.append(['sex', 'age'])

        file_paths.append('processed_german')
        targets.append('CREDITRATING')
        sensitive_columns_list.append(['PersonStatusSex', 'AgeInYears'])

        file_paths.append('processed_kdd')
        targets.append('income')
        sensitive_columns_list.append(['sex', 'race'])

        file_paths.append('processed_law_school')
        targets.append('pass_bar')
        sensitive_columns_list.append(['male', 'race'])

        # Filter datasets if a specific one is provided
        if args.dataset:
            try:
                idx = file_paths.index(args.dataset)
                file_paths = [file_paths[idx]]
                targets = [targets[idx]]
                sensitive_columns_list = [sensitive_columns_list[idx]]
            except ValueError:
                print(f"Dataset '{args.dataset}' not found. Available datasets: {', '.join(file_paths)}")
                exit(1)

        # Run the main function with command-line arguments
        main(file_paths, targets, sensitive_columns_list,
             num_trials=args.trials,
             population_size=args.population,
             generations=args.generations)

