import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory containing the CSV files
csv_dir = "/home/ubuntu/GCN_FINBERT_LSTM/GCN_LSTM_STOCK_TREND_PREDICTION/stock_data/tsla_Simon"
output_dir = "/home/ubuntu/GCN_FINBERT_LSTM/GCN_LSTM_STOCK_TREND_PREDICTION/histogram_outputs"
os.makedirs(output_dir, exist_ok=True)

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

for csv_file in csv_files:
    # Read the CSV file
    file_path = os.path.join(csv_dir, csv_file)
    df = pd.read_csv(file_path)

    # Check if required columns exist
    required_columns = ['Positive_BERT', 'Negative_BERT', 'Neutral_BERT', 'Titles']
    if not all(col in df.columns for col in required_columns):
        print(f"Skipping {csv_file} as it does not contain the required columns.")
        continue

    # Extract the day from the filename
    day = os.path.splitext(csv_file)[0]

    # Calculate absolute sum of Positive and Negative scores
    df['Abs_Pos_Neg_Sum'] = abs(df['Positive_BERT'] + df['Negative_BERT'])

    # Identify the most influential news
    most_influential = df.loc[df['Abs_Pos_Neg_Sum'].idxmax()]

    # Create a single plot with three histograms
    plt.figure(figsize=(15, 15))

    # 1. Histogram of Scores for Each News (X-axis: Index)
    plt.subplot(3, 1, 1)
    x_indices = range(len(df))
    plt.bar(x_indices, df['Positive_BERT'], label='Positive', color='green', alpha=0.7)
    plt.bar(x_indices, df['Negative_BERT'], label='Negative', color='red', alpha=0.7, bottom=df['Positive_BERT'])
    plt.bar(x_indices, df['Neutral_BERT'], label='Neutral', color='blue', alpha=0.7, bottom=df['Positive_BERT'] + df['Negative_BERT'])
    plt.title(f"Histogram of Scores for Each News - {day}")
    plt.xlabel("News Index")
    plt.ylabel("Score")
    plt.legend()

    # 2. Histogram of Absolute Sum of Positive and Negative Scores
    plt.subplot(3, 1, 2)
    categories = ['Positive', 'Negative']
    scores = [df['Positive_BERT'].sum(), df['Negative_BERT'].sum()]
    plt.bar(categories, scores, color=['green', 'red'], alpha=0.7)
    plt.title(f"Histogram of Absolute Sum of Positive and Negative Scores - {day}")
    plt.xlabel("Categories")
    plt.ylabel("Score")

    # 3. Highlighted Most Influential News (X-axis: Titles)
    plt.subplot(3, 1, 3)
    highlight_color = ['yellow' if i == most_influential.name else 'gray' for i in df.index]
    x_labels = df['Titles']
    x_positions = range(len(x_labels))
    plt.bar(x_positions, df['Positive_BERT'], label='Positive', color=highlight_color, alpha=0.7)
    plt.bar(x_positions, df['Negative_BERT'], label='Negative', color=highlight_color, alpha=0.7, bottom=df['Positive_BERT'])
    plt.bar(x_positions, df['Neutral_BERT'], label='Neutral', color=highlight_color, alpha=0.7, bottom=df['Positive_BERT'] + df['Negative_BERT'])
    plt.xticks(x_positions, x_labels, rotation=90)
    plt.title(f"Highlighted Most Influential News - {day}")
    plt.xlabel("News Titles")
    plt.ylabel("Score")
    plt.legend()

    # Save the combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{day}_combined_histogram.png"))
    plt.close()

print(f"Combined histograms have been saved to {output_dir}")
