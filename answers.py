import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Load the data
users_df = pd.read_csv('users.csv')
repos_df = pd.read_csv('repositories.csv')

# Convert dates
users_df['created_at'] = pd.to_datetime(users_df['created_at'])
repos_df['created_at'] = pd.to_datetime(repos_df['created_at'])

# 1. Top 5 users by followers
top_followers = users_df.nlargest(5, 'followers')['login'].tolist()
print("1.", ",".join(top_followers))

# 2. 5 earliest registered users
earliest_users = users_df.nsmallest(5, 'created_at')['login'].tolist()
print("2.", ",".join(earliest_users))

# 3. Most popular licenses
top_licenses = repos_df['license_name'].dropna().value_counts().head(3).index.tolist()
print("3.", ",".join(top_licenses))

# 4. Most common company
most_common_company = users_df['company'].value_counts().index[0]
print("4.", most_common_company)

# 5. Most popular language
most_popular_lang = repos_df['language'].value_counts().index[0]
print("5.", most_popular_lang)

# 6. Second most popular language for users after 2020
recent_users = users_df[users_df['created_at'].dt.year > 2020]['login']
recent_repos = repos_df[repos_df['login'].isin(recent_users)]
second_popular_lang = recent_repos['language'].value_counts().index[1]
print("6.", second_popular_lang)

# 7. Language with highest average stars
avg_stars = repos_df.groupby('language')['stargazers_count'].mean()
top_lang_by_stars = avg_stars.nlargest(1).index[0]
print("7.", top_lang_by_stars)

# 8. Leader strength
users_df['leader_strength'] = users_df['followers'] / (1 + users_df['following'])
top_leaders = users_df.nlargest(5, 'leader_strength')['login'].tolist()
print("8.", ",".join(top_leaders))

# 9. Correlation between followers and public repos
corr = users_df['followers'].corr(users_df['public_repos'])
print("9.", f"{corr:.3f}")

# 10. Regression slope of followers on repos
X = users_df['public_repos'].values.reshape(-1, 1)
y = users_df['followers'].values
reg = LinearRegression().fit(X, y)
print("10.", f"{reg.coef_[0]:.3f}")

# 11. Correlation between projects and wiki enabled
repos_df['has_projects'] = repos_df['has_projects'].astype(int)
repos_df['has_wiki'] = repos_df['has_wiki'].astype(int)
corr_proj_wiki = repos_df['has_projects'].corr(repos_df['has_wiki'])
print("11.", f"{corr_proj_wiki:.3f}")

# 12. Hireable users following difference
hireable_following = users_df[users_df['hireable'] == True]['following'].mean()
non_hireable_following = users_df[users_df['hireable'] != True]['following'].mean()
following_diff = hireable_following - non_hireable_following
print("12.", f"{following_diff:.3f}")

# 13. Bio length correlation with followers
users_df['bio_length'] = users_df['bio'].fillna('').str.len()
bio_users = users_df[users_df['bio_length'] > 0]
X = bio_users['bio_length'].values.reshape(-1, 1)
y = bio_users['followers'].values
reg = LinearRegression().fit(X, y)
print("13.", f"{reg.coef_[0]:.3f}")

# 14. Most weekend repositories
repos_df['is_weekend'] = repos_df['created_at'].dt.dayofweek >= 5
weekend_counts = repos_df[repos_df['is_weekend']].groupby('login').size()
top_weekend = weekend_counts.nlargest(5).index.tolist()
print("14.", ",".join(top_weekend))

# 15. Hireable email sharing difference
hireable_email = users_df[users_df['hireable'] == True]['email'].notna().mean()
non_hireable_email = users_df[users_df['hireable'] != True]['email'].notna().mean()
email_diff = hireable_email - non_hireable_email
print("15.", f"{email_diff:.3f}")

# 16. Most common surname
def get_surname(name):
    if pd.isna(name):
        return None
    parts = str(name).strip().split()
    return parts[-1] if parts else None

users_df['surname'] = users_df['name'].apply(get_surname)
surname_counts = users_df['surname'].value_counts()
max_count = surname_counts.max()
most_common_surnames = surname_counts[surname_counts == max_count].index.tolist()
most_common_surnames.sort()
print("16.", max_count)

print("Most common surnames:", ", ".join(most_common_surnames))