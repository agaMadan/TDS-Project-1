import requests
import pandas as pd
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry

CALLS_PER_MINUTE = 80  # Staying slightly under the limit to be safe
ONE_MINUTE = 60
load_dotenv()

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=ONE_MINUTE)
def api_call(url: str, headers: dict, params: dict = None) -> requests.Response:
    """Make a rate-limited API call"""
    return requests.get(url, headers=headers, params=params)

class GitHubScraper:
    def __init__(self, token: str):
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'

    def get_berlin_users(self, min_followers: int = 200) -> List[Dict[str, Any]]:
        """Fetch GitHub users from Berlin with minimum number of followers."""
        users = []
        page = 1

        while True:
            response = api_call(
                f'{self.base_url}/search/users',
                headers=self.headers,
                params={
                    'q': f'location:Berlin followers:>={min_followers}',
                    'page': page,
                    'per_page': 100
                }
            )

            if response.status_code != 200:
                print(f"Error fetching users: {response.status_code}")
                break

            data = response.json()
            if not data['items']:
                break

            users.extend(data['items'])
            page += 1

        return users

    def get_user_details(self, username: str) -> Dict[str, Any]:
        """Fetch detailed information for a specific user."""
        response = api_call(
            f'{self.base_url}/users/{username}',
            headers=self.headers
        )

        if response.status_code != 200:
            print(f"Error fetching user details for {username}: {response.status_code}")
            return None

        return response.json()

    def get_user_repos(self, username: str, max_repos: int = 500) -> List[Dict[str, Any]]:
        """Fetch repositories for a specific user."""
        repos = []
        page = 1

        while len(repos) < max_repos:
            response = api_call(
                f'{self.base_url}/users/{username}/repos',
                headers=self.headers,
                params={
                    'sort': 'pushed',
                    'direction': 'desc',
                    'per_page': 100,
                    'page': page
                }
            )

            if response.status_code != 200:
                print(f"Error fetching repos for {username}: {response.status_code}")
                break

            data = response.json()
            if not data:
                break

            repos.extend(data)
            if len(data) < 100:
                break

            page += 1

        return repos[:max_repos]

    def process_user(self, user: Dict[str, Any]) -> tuple:
        """Process a single user and their repositories."""
        username = user['login']
        user_details = self.get_user_details(username)

        if not user_details:
            return None, None

        repos = self.get_user_repos(username)
        repos_df = self.create_repos_dataframe(repos, username) if repos else pd.DataFrame()

        return user_details, repos_df

    def clean_company_name(self, company: str) -> str:
        """Clean up company names according to specifications."""
        if not company:
            return ""

        company = company.strip()
        if company.startswith('@'):
            company = company[1:]
        return company.upper()

    def create_users_dataframe(self, users_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create users DataFrame with specified columns."""
        users_df = pd.DataFrame(users_data)
        columns = [
            'login', 'name', 'company', 'location', 'email', 'hireable',
            'bio', 'public_repos', 'followers', 'following', 'created_at'
        ]

        users_df['company'] = users_df['company'].apply(self.clean_company_name)
        return users_df[columns]

    def create_repos_dataframe(self, repos_data: List[Dict[str, Any]], username: str) -> pd.DataFrame:
        """Create repositories DataFrame with specified columns."""
        if not repos_data:
            return pd.DataFrame()

        repos_df = pd.DataFrame(repos_data)
        repos_df['login'] = username
        repos_df['license_name'] = repos_df['license'].apply(
            lambda x: x['key'] if isinstance(x, dict) and 'key' in x else None
        )

        columns = [
            'login', 'full_name', 'created_at', 'stargazers_count',
            'watchers_count', 'language', 'has_projects', 'has_wiki', 'license_name'
        ]

        return repos_df[columns]

def main():
    token = os.getenv('GITHUB_TOKEN')

    try:
        scraper = GitHubScraper(token)
        print("Fetching Berlin users...")
        berlin_users = scraper.get_berlin_users(min_followers=200)
        print(f"Found {len(berlin_users)} users in Berlin with >200 followers")

        users_data = []
        all_repos_data = []

        # parallel processing users
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_user = {
                executor.submit(scraper.process_user, user): user
                for user in berlin_users
            }

            completed = 0
            for future in as_completed(future_to_user):
                user = future_to_user[future]
                completed += 1
                print(f"Processing {completed}/{len(berlin_users)}: {user['login']}")

                try:
                    user_details, repos_df = future.result()
                    if user_details:
                        users_data.append(user_details)
                    if not repos_df.empty:
                        all_repos_data.append(repos_df)
                except Exception as e:
                    print(f"Error processing {user['login']}: {str(e)}")

        print("Creating final datasets...")
        users_df = scraper.create_users_dataframe(users_data)
        repos_df = pd.concat(all_repos_data, ignore_index=True) if all_repos_data else pd.DataFrame()

        print("Saving results...")
        users_df.to_csv('users.csv', index=False)
        repos_df.to_csv('repositories.csv', index=False)

        print("Done!")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()