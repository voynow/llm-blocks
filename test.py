
import base64
from datetime import datetime, timedelta
import os
import pandas as pd
import requests


GITHUB_CLIENT_ID = os.environ["GITHUB_CLIENT_ID"]
GITHUB_CLIENT_SECRET = os.environ["GITHUB_CLIENT_SECRET"]
GITHUB_ACCESS_TOKEN = os.environ["GITHUB_ACCESS_TOKEN"]


def get_trending_repos(access_token, language=None, sort="stars", order="desc", per_page=10, last_n_days=30):
    """ Get trending repositories from GitHub
    """
    url = "https://api.github.com/search/repositories"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {access_token}"
    }

    query = f"language:{language}" if language else "is:public"
    date = (datetime.now() - timedelta(days=last_n_days)).strftime("%Y-%m-%d")
    query += f" created:>{date}"

    params = {
        "q": query,
        "sort": sort,
        "order": order,
        "per_page": per_page
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["items"]

import base64


def get_file_contents(access_token, download_url):
    """ Get file contents from GitHub
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(download_url, headers=headers)
    response.raise_for_status()
    return response.text


def get_repo_contents_recursive(access_token, repo_owner, repo_name, path="", all_files=None):
    """ Get all files from a GitHub repository recursively
    """
    if all_files is None:
        all_files = []

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers).json()

    for item in response:
        if item["type"] == "file":
            file_content = get_file_contents(access_token, item["download_url"])
            all_files.append({"name": item["name"], "path": item["path"], "content": file_content})
        elif item["type"] == "dir":
            get_repo_contents_recursive(access_token, repo_owner, repo_name, item["path"], all_files)

    return all_files


def main():

    trending_repos = get_trending_repos(
        GITHUB_ACCESS_TOKEN,
        language="python",
        per_page=1,
        last_n_days=14
    )

    for repo in trending_repos:
        repo_owner = repo["owner"]["login"]
        repo_name = repo["name"]

        contents = get_repo_contents_recursive(GITHUB_ACCESS_TOKEN, repo_owner, repo_name)

        for item in contents:
            print(f"{item['name']} {item['path']}:\n{item['content']}\n\n")


if __name__ == "__main__":
    main()
