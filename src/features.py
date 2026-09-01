import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def engineer_features(df):
    # 1. Follower-to-Following Ratio
    df['follower_following_ratio'] = df['#followers'] / (df['#follows'] + 1)

    # 2. Profile Completeness Score
    df['completeness_score'] = (
        df['profile pic'].astype(int) +
        (df['description length'] > 0).astype(int) +
        df['external URL'].astype(int) +
        (1 - df['private']).astype(int)
    ) / 4

    # 3. Posts per Follower
    df['posts_per_follower'] = df['#posts'] / (df['#followers'] + 1)

    # 4. Username digit ratio (strong bot signal)
    df['username_digit_ratio'] = df['nums/length username']

    # 5. Fullname digit ratio
    df['fullname_digit_ratio'] = df['nums/length fullname']

    # 6. Fullname word count
    df['fullname_word_count'] = df['fullname words']

    # 7. Name matches username (strong bot signal)
    df['name_equals_username'] = df['name==username']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df
    

FEATURES = ['follower_following_ratio', 'completeness_score', 'posts_per_follower',
            'username_digit_ratio', 'fullname_digit_ratio', 'fullname_word_count',
            'name_equals_username']

def plot_feature_separation(df):
    os.makedirs('reports', exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, col in zip(axes.flatten(), FEATURES):
        sns.boxplot(x='fake', y=col, data=df, ax=ax)
    # hide unused subplot(s) -- 7 features in a 3x3 grid leaves 2 empty
    for ax in axes.flatten()[len(FEATURES):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('reports/feature_separation.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv('data/train.csv')
    df = engineer_features(df)
    df.to_csv('data/processed_accounts.csv', index=False)
    plot_feature_separation(df)
    print("Feature engineering complete. Saved to data/processed_accounts.csv")
    print("Feature separation chart saved to reports/feature_separation.png")
