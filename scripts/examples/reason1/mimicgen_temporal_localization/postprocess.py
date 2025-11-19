import pickle
import numpy as np
import os
import pandas as pd
import argparse

import re

def extract_two_floats(text):
    """
    Extract two float numbers from a string in format like "1.73s - 2.00s".
    
    Args:
        text (str): Input string containing patterns like "X.XXs - Y.YYs"
        
    Returns:
        tuple: Two float numbers found, or (None, None) if not found
    """
    # Pattern to match float numbers followed by 's' (seconds)
    # Matches patterns like "1.73 - 2.00"
    pattern = r'([+-]?\d*\.?\d+)'
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    if len(matches) >= 2:
        try:
            # Convert first two matches to floats
            float1 = float(matches[0])
            float2 = float(matches[1])
            return float1, float2
        except ValueError:
            return None, None
    else:
        return None, None

def get_num_events(gt_timestamps_list):
    """
    Determine the number of events based on ground truth timestamps.
    
    Args:
        gt_timestamps_list (list): List of timestamps for a demo
        
    Returns:
        int: Number of events
    """
    return len(gt_timestamps_list)

# demo6
gt_timestamps_nut_pouring = {
    "0": [1.7, 6.2, 8.3],
    "1": [1.9, 6.0, 8.1],
    "2": [1.8, 5.9, 7.6],
    "3": [1.7, 5.9, 7.6],
    "4": [1.8, 5.8, 7.9]
}

gt_timestamps_cube_stacking = {
    "0": [2.1, 3.5, 5.37],
    "1": [1.9, 3.2, 5.13],
    "2": [0.83, 2.87, 3.93],
    "3": [1.33, 2.73, 4.27],
    "4": [1.53, 3.17, 5.2],
    "5": [0.97, 2.77, 4.23],
    "6": [1.7, 4.07, 5.53],
    "7": [1.27, 2.67, 4.9],
    "8": [1.37, 2.7, 4.43],
    "9": [1.33, 3.17, 4.17],
}

gt_timestamps_bridge = {
    "0": [0.6, 1.1],
    "1": [0.8, 1.1],
    "2": [0.73, 1.23],
    "3": [0.57, 1.2],
    "4": [0.67, 1.4]
}

gt_timestamps_toaster = {
    'cut_682545_head_color_fixed': [1.8333333333333333, 6.433333333333334, 9.833333333333334, 11.166666666666666], 
    'cut_681882_head_color_fixed': [1.5, 5.5, 9.833333333333334, 11.166666666666666], 
    'cut_688144_head_color_fixed': [1.6666666666666667, 7.833333333333333, 11.0, 11.9], 
    'cut_683236_head_color_fixed': [2.8333333333333335, 7.466666666666667, 10.666666666666666, 11.666666666666666], 
    'cut_688068_head_color_fixed': [2.0, 8.0, 11.066666666666666, 11.8], 
    'cut_682655_head_color_fixed': [1.6666666666666667, 6.1, 9.6, 10.433333333333334], 
    'cut_683270_head_color_fixed': [2.1333333333333333, 5.966666666666667, 9.533333333333333, 10.266666666666667], 
    'cut_687781_head_color_fixed': [1.8666666666666667, 7.433333333333334, 11.666666666666666, 14.233333333333333], 
    'cut_683347_head_color_fixed': [2.2333333333333334, 7.733333333333333, 12.066666666666666, 12.733333333333333], 
    'cut_686921_head_color_fixed': [2.2666666666666666, 7.3, 11.2, 12.0]
    }

gt_timestamps_chips = {
    'cut_724566_head_color_fixed': [3.6666666666666665, 13.0],
    'cut_725207_head_color_fixed': [2.5, 9.666666666666666],
    'cut_725261_head_color_fixed': [2.1666666666666665, 10.5],
    'cut_723523_head_color_fixed': [2.3333333333333335, 11.666666666666666],
    'cut_724955_head_color_fixed': [4.0, 14.533333333333333],
    'cut_725248_head_color_fixed': [2.3333333333333335, 10.0],
    'cut_723493_head_color_fixed': [2.466666666666667, 8.833333333333334],
    'cut_725170_head_color_fixed': [2.5, 9.833333333333334],
    'cut_724153_head_color_fixed': [2.6666666666666665, 11.166666666666666],
    'cut_723270_head_color_fixed': [4.766666666666667, 13.833333333333334]
}

gt_timestamps_fork = {
    'cut_686443_head_color_fixed': [4.766666666666667, 9.0, 11.166666666666666, 14.0], 
    'cut_686066_head_color_fixed': [4.2, 9.6, 11.5, 14.0], 
    'cut_686488_head_color_fixed': [3.3333333333333335, 9.0, 11.466666666666667, 14.266666666666667], 
    'cut_685938_head_color_fixed': [2.0, 7.666666666666667, 10.666666666666666, 14.666666666666666], 
    'cut_686045_head_color_fixed': [5.666666666666667, 9.333333333333334, 11.166666666666666, 13.333333333333334], 
    'cut_686094_head_color_fixed': [3.066666666666667, 7.166666666666667, 10.5, 13.933333333333334], 
    'cut_686180_head_color_fixed': [3.3333333333333335, 8.0, 10.533333333333333, 14.166666666666666], 
    'cut_686320_head_color_fixed': [2.1666666666666665, 6.933333333333334, 10.633333333333333, 14.5], 
    'cut_686245_head_color_fixed': [5.0, 9.0, 10.933333333333334, 14.0], 
    'cut_686470_head_color_fixed': [2.3, 7.6, 10.6, 14.333333333333334]
    }

gt_timestamps_cup = {
    'cut_709524_head_color_fixed': [4.333333333333333, 8.166666666666666, 13.5], 
    'cut_709556_head_color_fixed': [2.3333333333333335, 7.5, 14.433333333333334], 
    'cut_709507_head_color_fixed': [3.6666666666666665, 7.166666666666667, 14.166666666666666], 
    'cut_709325_head_color_fixed': [2.6666666666666665, 6.466666666666667, 14.0], 
    'cut_709342_head_color_fixed': [3.3333333333333335, 6.766666666666667, 13.833333333333334], 
    'cut_710323_head_color_fixed': [2.0, 6.666666666666667, 14.333333333333334], 
    'cut_710150_head_color_fixed': [3.2, 8.233333333333333, 14.333333333333334], 
    'cut_709286_head_color_fixed': [3.3333333333333335, 7.0, 14.333333333333334], 
    'cut_709987_head_color_fixed': [3.0, 7.6, 14.333333333333334], 
    'cut_709401_head_color_fixed': [2.1666666666666665, 6.166666666666667, 14.333333333333334]
    }

if __name__ == "__main__":
    # Mapping of timestamp types to their dictionaries
    gt_timestamps_map = {
        'nut': gt_timestamps_nut_pouring,
        'cube': gt_timestamps_cube_stacking,
        'bridge': gt_timestamps_bridge,
        'toaster': gt_timestamps_toaster,
        'chips': gt_timestamps_chips,
        'fork': gt_timestamps_fork,
        'cup': gt_timestamps_cup
    }
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Postprocess video timestamp predictions and compute accuracy metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available ground truth timestamp types:
  {', '.join(gt_timestamps_map.keys())}

Examples:
  python postprocess.py /path/to/results --gt-timestamps toaster --fps 8
  python postprocess.py /path/to/results --gt-timestamps bridge --use-start-time
  python postprocess.py /path/to/results --gt-timestamps cube_stacking --fps 4
        """
    )
    
    parser.add_argument(
        'results_dir',
        type=str,
        help='Directory containing the results pickle files'
    )
    
    parser.add_argument(
        '--gt-timestamps',
        type=str,
        required=True,
        choices=list(gt_timestamps_map.keys()),
        help='Type of ground truth timestamps to use'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=8,
        help='Frames per second for video analysis (default: 8)'
    )
    
    parser.add_argument(
        '--use-start-time',
        action='store_true',
        help='Use start times (even indices) instead of end times (odd indices) for predictions'
    )
    
    parser.add_argument(
        '--num-trials',
        type=int,
        default=10,
        help='Number of trials to process per demo (default: 10)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set variables from parsed arguments
    results_dir = args.results_dir
    output_dir = results_dir
    gt_timestamps_type = args.gt_timestamps
    fps = args.fps
    use_end_time = not args.use_start_time
    num_trials = args.num_trials
    
    # Get the ground truth timestamps
    gt_timestamps = gt_timestamps_map[gt_timestamps_type]
    
    # Print configuration
    print(f"Using {'END' if use_end_time else 'START'} times ({'odd' if use_end_time else 'even'} indices)")
    print(f"FPS: {fps}")
    print(f"GT Timestamps Type: {gt_timestamps_type}")
    print(f"Number of trials: {num_trials}")
    print(f"Results directory: {results_dir}")
    print()

    # Store aggregate statistics across all demos and trials
    all_demos_errors_seconds = []
    all_demos_errors_percent = []
    all_demos_hit_rates = []
    all_demos_subtask_durations = []
    all_trials_data = []  # Store per-trial data for analysis
    time_window = 4.1 * (1 / fps) / 2

    for demo_name in gt_timestamps.keys():
        print(f"\n{'='*70}")
        print(f"Processing Demo {demo_name}...")
        print(f"GT Timestamps: {gt_timestamps[demo_name]}")
        print(f"{'='*70}")
        
        # Determine number of events for this demo
        num_events = get_num_events(gt_timestamps[demo_name])
        print(f"Number of events: {num_events}")
        
        results_file = f'{results_dir}/results_{demo_name}.pkl'
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            continue
        with open(results_file, 'rb') as f:
            loaded_dict = pickle.load(f)
        
        
        # Calculate subtask durations from ground truth timestamps dynamically
        subtask_durations = []
        for i in range(num_events):
            if i == 0:
                subtask_durations.append(gt_timestamps[demo_name][0])  # Duration until first event
            else:
                subtask_durations.append(gt_timestamps[demo_name][i] - gt_timestamps[demo_name][i-1])  # Duration between events
        
        # Store errors and hits across all trials for this demo
        demo_trials_errors_seconds = []
        demo_trials_errors_percent = []
        demo_trials_hit_rates = []
        demo_trials_predictions = []
        
        for trial in range(num_trials):
            print(f"\n--- Trial {trial} ---")
            if not f"fps{fps}" in loaded_dict:
                print(f"FPS {fps} not found in results file")
                continue
            output_text_fps = loaded_dict[f"fps{fps}"]
            if not output_text_fps:
                print(f"Demo {demo_name} fps {fps} trial {trial}: No output text, skipping...")
                continue
            # Check if trial index is within bounds
            if trial >= len(output_text_fps):
                print(f"Trial {trial}: Index out of range (only {len(output_text_fps)} trials available), skipping...")
                continue
            output_text = loaded_dict[f"fps{fps}"][trial]
            # print(output_text)
            
            if not output_text:
                print(f"Trial {trial}: No output text, skipping...")
                continue
            
            # Handle case where output_text might be a list
            if isinstance(output_text, list):
                if len(output_text) > 0:
                    output_text = output_text[0]
                else:
                    print(f"Trial {trial}: Empty output text list, skipping...")
                    continue
            
            # Extract predictions from output
            tmp_list = []
            tmp_list.append(np.ravel([list(extract_two_floats(i)) for i in output_text.split('\n') if 'Event' in i]))
            expected_length = num_events * 2  # Each event has start and end time
            tmp_list = [i for i in tmp_list if len(i) == expected_length and all(x is not None for x in i)]
            
            if len(tmp_list) == 0:
                print(f"Trial {trial}: No valid predictions extracted, skipping...")
                continue
            
            print(f"Trial {trial}: Extracted predictions: {tmp_list[0]}")
            demo_trials_predictions.append(tmp_list[0])
            
            # Calculate hits and errors for this trial
            hit_table = []
            error_table = []
            
            for i in tmp_list:
                # Dynamically build hit and error tables based on number of events
                hit_row = []
                error_row = []
                for event_idx in range(num_events):
                    # Use flag to determine which index to use
                    if use_end_time:
                        pred_end_time = i[event_idx * 2 + 1]  # End time is at odd indices
                    else:
                        pred_end_time = i[event_idx * 2]  # Start time is at even indices
                    
                    gt_time = gt_timestamps[demo_name][event_idx]
                    hit_row.append(np.abs(pred_end_time - gt_time) < time_window)
                    error_row.append(np.abs(pred_end_time - gt_time))
                
                hit_table.append(hit_row)
                error_table.append(error_row)
            
            hit_table = np.array(hit_table)
            error_table = np.array(error_table)
            
            # Calculate metrics for this trial
            trial_errors_seconds = np.mean(error_table, axis=0)
            trial_errors_percent = [
                (trial_errors_seconds[i] / subtask_durations[i] * 100) if subtask_durations[i] > 0 else 0
                for i in range(num_events)
            ]
            trial_hit_rates = np.average(hit_table, axis=0)
            
            demo_trials_errors_seconds.append(trial_errors_seconds)
            demo_trials_errors_percent.append(trial_errors_percent)
            demo_trials_hit_rates.append(trial_hit_rates)
            
            print(f"Trial {trial} - Errors (s): {trial_errors_seconds}")
            print(f"Trial {trial} - Hit rates: {trial_hit_rates}")
        
        # Calculate average across all trials for this demo
        if len(demo_trials_errors_seconds) > 0:
            demo_avg_errors_seconds = np.mean(demo_trials_errors_seconds, axis=0)
            demo_avg_errors_percent = np.mean(demo_trials_errors_percent, axis=0)
            demo_avg_hit_rates = np.mean(demo_trials_hit_rates, axis=0)
            demo_std_errors_seconds = np.std(demo_trials_errors_seconds, axis=0)
            demo_std_errors_percent = np.std(demo_trials_errors_percent, axis=0)
            
            print(f"\n=== Demo {demo_name} Summary (across {len(demo_trials_errors_seconds)} trials) ===")
            print(f"Subtask durations (seconds): {subtask_durations}")
            print(f"Average errors (seconds): {demo_avg_errors_seconds} ± {demo_std_errors_seconds}")
            print(f"Average errors (percent): {demo_avg_errors_percent} ± {demo_std_errors_percent}")
            
            # Print event-specific metrics dynamically
            for event_idx in range(num_events):
                print(f"Event {event_idx + 1} - Avg error: {demo_avg_errors_seconds[event_idx]:.3f}s ± {demo_std_errors_seconds[event_idx]:.3f}s ({demo_avg_errors_percent[event_idx]:.1f}% of {subtask_durations[event_idx]:.2f}s subtask)")
            
            print(f"Overall avg error: {np.mean(demo_avg_errors_seconds):.3f}s ({np.mean(demo_avg_errors_percent):.1f}%)")
            
            # Print hit rates dynamically
            hit_rate_str = ", ".join([f"Event{i+1}={demo_avg_hit_rates[i]:.3f}" for i in range(num_events)])
            print(f"Average hit rates: {hit_rate_str}")
            print("=" * 70)
            
            # Save per-demo results
            os.makedirs(output_dir, exist_ok=True)
            
            # Save all predictions for this demo
            if len(demo_trials_predictions) > 0:
                # Dynamically generate column names based on number of events
                column_names = []
                for event_idx in range(num_events):
                    column_names.extend([f"event{event_idx+1}_start", f"event{event_idx+1}_end"])
                
                predictions_df = pd.DataFrame(demo_trials_predictions, columns=column_names)
                predictions_df['trial'] = range(len(demo_trials_predictions))
                predictions_df.to_csv(f'{output_dir}/results_stat_demo_{demo_name}_fps{fps}_all_trials.csv', index=False)
            
            # Save per-trial error statistics
            trials_error_dict = {'trial': range(len(demo_trials_errors_seconds))}
            
            # Dynamically add columns for each event
            for event_idx in range(num_events):
                trials_error_dict[f'event{event_idx+1}_error_s'] = [e[event_idx] for e in demo_trials_errors_seconds]
                trials_error_dict[f'event{event_idx+1}_error_pct'] = [e[event_idx] for e in demo_trials_errors_percent]
                trials_error_dict[f'event{event_idx+1}_hit_rate'] = [h[event_idx] for h in demo_trials_hit_rates]
            
            trials_error_df = pd.DataFrame(trials_error_dict)
            trials_error_df.to_csv(f'{output_dir}/per_trial_stats_demo_{demo_name}_fps{fps}.csv', index=False)
            
            # Save demo summary statistics
            demo_summary_df = pd.DataFrame({
                'event': [f'Event {i+1}' for i in range(num_events)],
                'subtask_duration_s': subtask_durations,
                'avg_error_s': demo_avg_errors_seconds,
                'std_error_s': demo_std_errors_seconds,
                'avg_error_percent': demo_avg_errors_percent,
                'std_error_percent': demo_std_errors_percent,
                'avg_hit_rate': demo_avg_hit_rates
            })
            demo_summary_df.to_csv(f'{output_dir}/summary_demo_{demo_name}_fps{fps}.csv', index=False)
            
            # Store data for overall aggregate calculation
            all_demos_errors_seconds.append(demo_avg_errors_seconds)
            all_demos_errors_percent.append(demo_avg_errors_percent)
            all_demos_hit_rates.append(demo_avg_hit_rates)
            all_demos_subtask_durations.append(subtask_durations)
        else:
            print(f"Demo {demo_name}: No valid trials, skipping...")

    # Calculate and display aggregate statistics across all demos
    print("\n" + "=" * 70)
    print("=== AGGREGATE STATISTICS ACROSS ALL DEMOS ===")
    print("=" * 70)

    if len(all_demos_errors_seconds) > 0:
        all_demos_errors_seconds = np.array(all_demos_errors_seconds)
        all_demos_errors_percent = np.array(all_demos_errors_percent)
        all_demos_hit_rates = np.array(all_demos_hit_rates)
        all_demos_subtask_durations = np.array(all_demos_subtask_durations)
        
        # Calculate mean and std across all demos
        mean_errors_seconds = np.mean(all_demos_errors_seconds, axis=0)
        std_errors_seconds = np.std(all_demos_errors_seconds, axis=0)
        mean_errors_percent = np.mean(all_demos_errors_percent, axis=0)
        std_errors_percent = np.std(all_demos_errors_percent, axis=0)
        mean_hit_rates = np.mean(all_demos_hit_rates, axis=0)
        std_hit_rates = np.std(all_demos_hit_rates, axis=0)
        mean_subtask_durations = np.mean(all_demos_subtask_durations, axis=0)
        
        print(f"\nNumber of demos analyzed: {len(all_demos_errors_seconds)}")
        print(f"Number of trials per demo: {num_trials}")
        
        # Get the maximum number of events across all demos
        max_events = mean_errors_seconds.shape[0] if len(mean_errors_seconds.shape) > 0 else len(mean_errors_seconds)
        
        # Print subtask durations
        duration_str = ", ".join([f"{mean_subtask_durations[i]:.2f}s" for i in range(max_events)])
        print(f"Average subtask durations across demos: [{duration_str}]")
        
        print("\n--- Average Errors Across All Demos ---")
        for event_idx in range(max_events):
            print(f"Event {event_idx + 1} - Mean error: {mean_errors_seconds[event_idx]:.3f}s ± {std_errors_seconds[event_idx]:.3f}s ({mean_errors_percent[event_idx]:.1f}% ± {std_errors_percent[event_idx]:.1f}%)")
        
        print(f"\nOverall mean error: {np.mean(mean_errors_seconds):.3f}s ± {np.mean(std_errors_seconds):.3f}s")
        print(f"Overall mean error: {np.mean(mean_errors_percent):.1f}% ± {np.mean(std_errors_percent):.1f}%")
        
        # Print hit rates dynamically
        hit_rate_str = ", ".join([f"Event{i+1}={mean_hit_rates[i]:.3f}±{std_hit_rates[i]:.3f}" for i in range(max_events)])
        print(f"\nMean hit rates: {hit_rate_str}")
        
        # Save aggregate statistics
        aggregate_df = pd.DataFrame({
            'event': [f'Event {i+1}' for i in range(max_events)],
            'mean_subtask_duration_s': mean_subtask_durations,
            'mean_error_s': mean_errors_seconds,
            'std_error_s': std_errors_seconds,
            'mean_error_percent': mean_errors_percent,
            'std_error_percent': std_errors_percent,
            'mean_hit_rate': mean_hit_rates,
            'std_hit_rate': std_hit_rates
        })
        aggregate_df.to_csv(f'{output_dir}/aggregate_stats_all_demos_fps{fps}.csv', index=False)
        
        # Save detailed per-demo comparison (averaged across trials for each demo)
        demo_keys = [k for k, v in zip(gt_timestamps.keys(), all_demos_errors_seconds) if v is not None]
        comparison_dict = {'demo': demo_keys}
        
        # Dynamically add columns for each event
        for event_idx in range(max_events):
            comparison_dict[f'event{event_idx+1}_error_s'] = all_demos_errors_seconds[:, event_idx]
            comparison_dict[f'event{event_idx+1}_error_pct'] = all_demos_errors_percent[:, event_idx]
            comparison_dict[f'event{event_idx+1}_hit_rate'] = all_demos_hit_rates[:, event_idx]
        
        comparison_df = pd.DataFrame(comparison_dict)
        comparison_df.to_csv(f'{output_dir}/per_demo_comparison_fps{fps}.csv', index=False)
        
        print(f"\nAggregate statistics saved to: {output_dir}/aggregate_stats_all_demos_fps{fps}.csv")
        print(f"Per-demo comparison saved to: {output_dir}/per_demo_comparison_fps{fps}.csv")
        print("=" * 70)
    else:
        print("\nNo valid data to aggregate!")
        print("=" * 70)