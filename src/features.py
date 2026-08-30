import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def engineer_features(df):
    # 1. Follower-to-Following Ratio
    df['follower_following_ratio'] = df['#followers'] / (df['#follows'] + 1)

    # 2. Profile Completeness Score
    # (no bio_length/has_location/has_website in this dataset --
    #  using description length, external URL, and inverse of private instead)
    df['completeness_score'] = (
        df['profile pic'].astype(int) +
        (df['description length'] > 0).astype(int) +
        df['external URL'].astype(int) +
        (1 - df['private']).astype(int)
    ) / 4

    # 3. Posts per Follower
    # (replaces "engagement_rate" -- likes/comments data isn't available here,
    #  so this uses post activity relative to follower count instead)
    df['posts_per_follower'] = df['#posts'] / (df['#followers'] + 1)

    # Note: account_age_days is dropped entirely -- this dataset has no
    # account creation date or data collection date to compute it from.

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

FEATURES = ['follower_following_ratio', 'completeness_score', 'posts_per_follower']

def plot_feature_separation(df):
    os.makedirs('reports', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes.flatten(), FEATURES):
        sns.boxplot(x='fake', y=col, data=df, ax=ax)
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
