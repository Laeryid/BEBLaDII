import pandas as pd
import sys

csv_path = r"C:\Experiments\BEBLaDII\experiments\phase 4\local_checkpoints\diffusion_trajectory.csv"
df = pd.read_csv(csv_path)

# Filter for a specific run
run_df = df[(df['RunDate'] == '13.09.2026 10:06:00') & (df['ModelVersion'] == 'phase4_step_14995_LIVE') & (df['Phrase'].str.contains('The quick brown'))]

for idx in range(10): # Look at first 10 tokens
    token_df = run_df[run_df['TokenIndex'] == idx].sort_values('Iteration')
    if token_df.empty: continue
    start_noise = token_df.iloc[0]['NoiseValue']
    end_noise = token_df.iloc[-1]['NoiseValue']
    start_word = token_df.iloc[0]['TokenString']
    end_word = token_df.iloc[-1]['TokenString']
    print(f"Token {idx}: StartNoise={start_noise:.4f} ('{start_word}'), EndNoise={end_noise:.4f} ('{end_word}')")
    
    # Print trajectory if it started high
    if start_noise > 0.6:
        print(f"  Trajectory for Token {idx}:")
        for _, row in token_df.iterrows():
            print(f"    Iter {row['Iteration']}: Noise={row['NoiseValue']:.4f}, Word='{row['TokenString']}'")