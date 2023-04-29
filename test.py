import os
import requests
import pandas as pd

GITHUB_CLIENT_ID = os.environ["GITHUB_CLIENT_ID"]
GITHUB_CLIENT_SECRET = os.environ["GITHUB_CLIENT_SECRET"]
GITHUB_ACCESS_TOKEN = os.environ["GITHUB_ACCESS_TOKEN"]

def get_access_token():
    url = "https://github.com/login/oauth/access_token"
    payload = {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "scope": "repo"
    }
    headers = {"Accept": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["access_token"]

def get_trending_repos(access_token, language=None, sort="stars", order="desc", per_page=10):
    url = "https://api.github.com/search/repositories"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "q": f"language:{language}" if language else "is:public",
        "sort": sort,
        "order": order,
        "per_page": per_page
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["items"]

def main():
    trending_repos = get_trending_repos(GITHUB_ACCESS_TOKEN, language="python", per_page=10)

    # Process the results using pandas (optional)
    repos_df = pd.DataFrame(trending_repos)
    repos_df = repos_df[["id", "name", "html_url", "description", "stargazers_count", "language"]]

    print(repos_df)

if __name__ == "__main__":
    main()
