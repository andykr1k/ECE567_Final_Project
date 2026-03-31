import pandas as pd
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
import numpy as np

# ==========================================
# Configuration
# ==========================================
# Directories
CACHE_DIR = Path("./cache")
RESULTS_DIR = Path("./reproduction_results")
# Your Weights & Biases Entity (username or team name) and Project name
WANDB_ENTITY = "veldrovive"  # e.g., "my-team"
WANDB_PROJECT = "craftax_baselines_new_jax" # e.g., "craftax-reproduction"

# The exact metric name as logged in W&B
METRIC_NAME = "episode_return"
X_AXIS_METRIC = "_step" # Or "global_step", "Step", depending on how it was logged

# Adjust the smoothing window; larger numbers = smoother curves
SMOOTHING_WINDOW = 200  

# Map the display labels to the exact Run Names in your W&B project
# MAIN_RUNS = {
#     'ICM': 'Craftax-ICM-v1-1000M',
#     'PPO': 'Craftax-Symbolic-v1-PPO-10000M',
#     'RND': 'Craftax-Symbolic-v1-PPO_RND-10000M',
#     'PPO-RNN': 'Craftax-Symbolic-v1-PPO_RNN-5000M',
# }
MAIN_RUNS = {
    'PPO': 'Craftax-PPO-v1-1000M',
    'PPO-RNN': 'Craftax-Symbolic-v1-PPO_RNN-1000M',
    'PPO-ICM': 'Craftax-Symbolic-v1-ICM-1000M',
    'PPO-E3B': 'Craftax-ICM-E3B-v1-1000M',
    'PPO-RND': 'Craftax-Symbolic-v1-PPO_RND-1000M'
}

RNN_VARIANCE_RUNS = {
    'Seed 1': 'Craftax-Symbolic-v1-PPO_RNN-10000M',
    'Seed 2': 'Craftax-Symbolic-v1-PPO_RNN-5000M',
    'Seed 3': 'Craftax-Symbolic-v1-PPO_RNN-1000M'
}

ACHIEVEMENTS_BY_DIFFICULTY = {
    "Basic": [
        "COLLECT_WOOD", "PLACE_TABLE", "EAT_COW", "COLLECT_SAPLING", "COLLECT_DRINK", 
        "MAKE_WOOD_PICKAXE", "MAKE_WOOD_SWORD", "PLACE_PLANT", "DEFEAT_ZOMBIE", 
        "COLLECT_STONE", "PLACE_STONE", "EAT_PLANT", "DEFEAT_SKELETON", 
        "MAKE_STONE_PICKAXE", "MAKE_STONE_SWORD", "WAKE_UP", "PLACE_FURNACE", 
        "COLLECT_COAL", "COLLECT_IRON", "COLLECT_DIAMOND", "MAKE_IRON_PICKAXE", 
        "MAKE_IRON_SWORD", "MAKE_ARROW", "MAKE_TORCH", "PLACE_TORCH"
    ],
    "Intermediate": [
        "MAKE_DIAMOND_SWORD", "MAKE_IRON_ARMOUR", "MAKE_DIAMOND_ARMOUR", 
        "ENTER_GNOMISH_MINES", "ENTER_DUNGEON", "DEFEAT_GNOME_WARRIOR", 
        "DEFEAT_GNOME_ARCHER", "DEFEAT_ORC_SOLIDER", "DEFEAT_ORC_MAGE", 
        "EAT_BAT", "EAT_SNAIL", "FIND_BOW", "FIRE_BOW", "COLLECT_SAPPHIRE", 
        "COLLECT_RUBY", "MAKE_DIAMOND_PICKAXE", "OPEN_CHEST", "DRINK_POTION"
    ],
    "Advanced": [
        "ENTER_SEWERS", "ENTER_VAULT", "ENTER_TROLL_MINES", "DEFEAT_LIZARD", 
        "DEFEAT_KOBOLD", "DEFEAT_TROLL", "DEFEAT_DEEP_THING", "LEARN_FIREBALL", 
        "CAST_FIREBALL", "LEARN_ICEBALL", "CAST_ICEBALL", "ENCHANT_SWORD", 
        "ENCHANT_ARMOUR", "DEFEAT_KNIGHT", "DEFEAT_ARCHER"
    ],
    "Very Advanced": [
        "ENTER_FIRE_REALM", "ENTER_ICE_REALM", "ENTER_GRAVEYARD", "DEFEAT_PIGMAN", 
        "DEFEAT_FIRE_ELEMENTAL", "DEFEAT_FROST_TROLL", "DEFEAT_ICE_ELEMENTAL", 
        "DAMAGE_NECROMANCER", "DEFEAT_NECROMANCER"
    ]
}
achievement_wandb_keys = {}
for tier, achievements in ACHIEVEMENTS_BY_DIFFICULTY.items():
    for achievement in achievements:
        achievement_wandb_keys[achievement] = f"Achievements/{achievement.lower()}"

# ==========================================
# Helper Functions
# ==========================================
def smooth(series, window):
    """Applies a rolling mean to smooth out spiky W&B data."""
    return series.rolling(window=window, min_periods=1).mean()

def fetch_run_data(api, entity, project, run_name, metrics):
    """Fetches the history of specific metrics for a given run name."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_run_name = run_name.replace('/', '_').replace(' ', '_')
    
    if isinstance(metrics, str):
        metrics = [metrics]
        
    metrics_str = "_".join(metrics).replace('/', '_') # Replaces the slash to avoid directory traversal errors
    cache_path = CACHE_DIR / f"{safe_run_name}_{metrics_str}.csv"
    
    if cache_path.exists():
        print(f"Loading cached data for run: {run_name} ({metrics_str})...")
        return pd.read_csv(cache_path)
        
    print(f"Fetching data for run: {run_name} ({metrics_str})...")
    
    # Search for the run by its display name
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    
    if len(runs) == 0:
        print(f"  -> WARNING: Run '{run_name}' not found! Skipping.")
        return None
    elif len(runs) > 1:
        print(f"  -> WARNING: Multiple runs found with name '{run_name}'. Using the most recent one.")
    
    run = runs[0]
    
    # Use scan_history to get all data points (history() limits to 500 by default)
    history = run.scan_history(keys=[X_AXIS_METRIC] + metrics)
    
    # Convert generator to list of dicts, then to DataFrame
    data = list(history)
    if not data:
        print(f"  -> WARNING: No data found for metrics '{metrics_str}' in run '{run_name}'.")
        return None
        
    df = pd.DataFrame(data)
    
    # Sort by the x-axis step just to be safe
    df = df.sort_values(by=X_AXIS_METRIC).reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df

# ==========================================
# Plotting Functions
# ==========================================
def plot_main_figure(api):
    """Produces the main figure comparing PPO, RND, and the longer PPO-RNN run."""
    plt.figure(figsize=(10, 6))
    
    for label, run_name in MAIN_RUNS.items():
        df = fetch_run_data(api, WANDB_ENTITY, WANDB_PROJECT, run_name, METRIC_NAME)
        
        if df is not None and METRIC_NAME in df.columns:
            # Drop NaNs just in case W&B returns sparse rows
            valid_data = df[[X_AXIS_METRIC, METRIC_NAME]].dropna()
            smoothed_y = smooth(valid_data[METRIC_NAME], SMOOTHING_WINDOW)
            
            plt.plot(valid_data[X_AXIS_METRIC], smoothed_y, label=label, linewidth=1.5)

    plt.xlabel('Timestep')
    # Note: If you want to match the paper's "% of max" scale (where max=226), 
    # update the plotting line to: plt.plot(..., smoothed_y / 2.26, ...)
    plt.ylabel('Episode Return') 
    plt.title('Main Benchmark: Reward vs Timestep')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = RESULTS_DIR / 'main_figure_wandb.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_filename}\n")

def plot_rnn_variance_figure(api):
    """Produces a figure specifically comparing the two PPO-RNN seeds."""
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Keep colors distinct for comparison
    assert len(colors) >= len(RNN_VARIANCE_RUNS), "Not enough colors for all runs"
    
    for (label, run_name), color in zip(RNN_VARIANCE_RUNS.items(), colors):
        df = fetch_run_data(api, WANDB_ENTITY, WANDB_PROJECT, run_name, METRIC_NAME)
        
        if df is not None and METRIC_NAME in df.columns:
            valid_data = df[[X_AXIS_METRIC, METRIC_NAME]].dropna()
            smoothed_y = smooth(valid_data[METRIC_NAME], SMOOTHING_WINDOW)
            
            plt.plot(valid_data[X_AXIS_METRIC], smoothed_y, label=label, color=color, linewidth=1.5)

    plt.xlabel('Timestep')
    plt.ylabel('Episode Return')
    plt.title('PPO-RNN Variance: Reward vs Timestep\n(Same Method and Hyperparameters, Different Seeds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = RESULTS_DIR / 'rnn_variance_figure_wandb.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_filename}\n")

def generate_latex_results_table(api):
    """
    Fetches the final smoothed reward for each run and outputs a 
    ready-to-use LaTeX table comparing their performance.
    """
    print("\n--- Generating LaTeX Results Table ---")
    
    # Max possible reward in Craftax is 226. We use this to calculate % of Max
    MAX_REWARD = 226.0 
    
    results = {}
    
    # We combine the dictionaries to get results for all tracked runs
    all_runs = {**MAIN_RUNS}
    
    for label, run_name in all_runs.items():
        df = fetch_run_data(api, WANDB_ENTITY, WANDB_PROJECT, run_name, METRIC_NAME)
        
        if df is not None and METRIC_NAME in df.columns:
            valid_data = df[[X_AXIS_METRIC, METRIC_NAME]].dropna()
            # Calculate rolling mean to get a stable final value
            smoothed_y = smooth(valid_data[METRIC_NAME], SMOOTHING_WINDOW)
            
            # Extract the final smoothed value
            final_val = smoothed_y.iloc[-1]
            percent_max = (final_val / MAX_REWARD) * 100
            
            results[label] = {
                "final_reward": final_val,
                "percent_max": percent_max
            }
            
    # Constructing the LaTeX Table
    latex_str = "\\begin{table}[h]\n"
    latex_str += "\\centering\n"
    latex_str += "\\begin{tabular}{lcc}\n"
    latex_str += "\\hline\n"
    latex_str += "\\textbf{Algorithm} & \\textbf{Final Reward} & \\textbf{\\% of Max Reward} \\\\\n"
    latex_str += "\\hline\n"
    
    for label, metrics in results.items():
        # Escape underscores for LaTeX
        clean_label = label.replace("_", "\\_")
        latex_str += f"{clean_label} & {metrics['final_reward']:.2f} & {metrics['percent_max']:.2f}\\% \\\\\n"
        
    latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\caption{Final performance of evaluated algorithms after training.}\n"
    latex_str += "\\label{tab:final_results}\n"
    latex_str += "\\end{table}\n"
    
    print("\n" + latex_str)
    
    # Optionally save to a file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = RESULTS_DIR / "final_results_table.tex"
    with open(output_filename, "w") as f:
        f.write(latex_str)
    print(f"Saved table to: {output_filename}\n")

def plot_selected_achievements_over_time(api):
    """
    Recreates Figure 5 from the paper: plots the success rate of a selected 
    subset of interesting achievements over time.
    """
    print("\n--- Generating Selected Achievements Over Time Figure ---")
    
    # The 6 achievements highlighted in Figure 5 of the paper
    selected_achievements = [
        "MAKE_WOOD_PICKAXE", "ENTER_DUNGEON", "DEFEAT_ZOMBIE",
        "ENTER_GNOMISH_MINES", "EAT_PLANT", "MAKE_DIAMOND_SWORD"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, achievement in enumerate(selected_achievements):
        metric_key = achievement_wandb_keys[achievement]
        ax = axes[i]
        
        for label, run_name in MAIN_RUNS.items():
            df = fetch_run_data(api, WANDB_ENTITY, WANDB_PROJECT, run_name, metric_key)
            
            if df is not None and metric_key in df.columns:
                valid_data = df[[X_AXIS_METRIC, metric_key]].dropna()
                smoothed_y = smooth(valid_data[metric_key], SMOOTHING_WINDOW)
                
                # Logged data is already 0-100, no need to multiply
                ax.plot(valid_data[X_AXIS_METRIC], smoothed_y, label=label, linewidth=1.5)
                
        ax.set_title(achievement.lower())
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Success Rate (%)")
        
        # Only put the legend on the first subplot to save space
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = RESULTS_DIR / 'selected_achievements_over_time.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_filename}")


def plot_achievement_bar_chart(api):
    """
    Recreates Figure 4 from the paper: Bar chart of achievement success rates 
    grouped by difficulty category.
    """
    print("\n--- Generating Achievement Difficulty Bar Chart ---")
    
    categories = list(ACHIEVEMENTS_BY_DIFFICULTY.keys())
    run_labels = list(MAIN_RUNS.keys())
    
    # Dictionaries to hold the aggregated final success rates and standard errors
    final_rates = {label: {cat: 0.0 for cat in categories} for label in run_labels}
    std_errors = {label: {cat: 0.0 for cat in categories} for label in run_labels}
    
    for label, run_name in MAIN_RUNS.items():
        for category, achievements in ACHIEVEMENTS_BY_DIFFICULTY.items():
            cat_rates = []
            for ach in achievements:
                metric_key = achievement_wandb_keys[ach]
                df = fetch_run_data(api, WANDB_ENTITY, WANDB_PROJECT, run_name, metric_key)
                
                if df is not None and metric_key in df.columns:
                    valid_data = df[metric_key].dropna()
                    if not valid_data.empty:
                        # Average the last 50 steps to get a stable final value resisting end-of-run noise
                        final_val = valid_data.tail(50).mean()
                        cat_rates.append(final_val)
            
            if cat_rates:
                # Data is already in percentages (0-100)
                final_rates[label][category] = np.mean(cat_rates)
                std_errors[label][category] = np.std(cat_rates) / np.sqrt(len(cat_rates))

    # Grouped Bar Chart Setup
    x = np.arange(len(categories))
    width = 0.8 / len(run_labels)  # Dynamic bar width based on number of baseline algorithms
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, label in enumerate(run_labels):
        rates = [final_rates[label][cat] for cat in categories]
        errors = [std_errors[label][cat] for cat in categories]
        
        # Calculate X offset for side-by-side grouped bars
        offset = (i - len(run_labels)/2) * width + width/2
        
        ax.bar(x + offset, rates, width, yerr=errors, label=label, capsize=3, alpha=0.9)

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Achievement Success Rate by Difficulty Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    output_filename = RESULTS_DIR / 'achievement_difficulty_bar_chart.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_filename}")

if __name__ == "__main__":
    # Initialize the W&B API client
    public_api = wandb.Api()
    
    print("--- Generating Main Figure ---")
    plot_main_figure(public_api)
    
    print("--- Generating RNN Variance Figure ---")
    plot_rnn_variance_figure(public_api)
    
    plot_selected_achievements_over_time(public_api)
    plot_achievement_bar_chart(public_api)
    
    generate_latex_results_table(public_api)
    
    print("\nDone!")