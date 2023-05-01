from datetime import datetime, timedelta
import requests
import pathlib
from typing import Dict, List, Optional


BASE_URL = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json"
}
SEARCH_REPOS_ENDPOINT = f"{BASE_URL}/search/repositories"
REPO_CONTENTS_ENDPOINT = f"{BASE_URL}/repos/{{repo_owner}}/{{repo_name}}/contents/{{path}}"


def get_authorized_headers(access_token: str) -> Dict[str, str]:
    """ Get headers with authorization token
    """
    return {**HEADERS, "Authorization": f"Bearer {access_token}"}


def get_trending_repos(
    access_token: str, 
    language: Optional[str] = None, 
    sort: str = "stars", 
    order: str = "desc", 
    per_page: int = 25, 
    last_n_days: int = 30
) -> List[Dict]:
    """ Get trending repos from the last n days.
    """
    headers = get_authorized_headers(access_token)

    query = f"language:{language}" if language else "is:public"
    date = (datetime.now() - timedelta(days=last_n_days)).strftime("%Y-%m-%d")
    query += f" created:>{date}"

    params = {
        "q": query,
        "sort": sort,
        "order": order,
        "per_page": per_page
    }
    response = requests.get(SEARCH_REPOS_ENDPOINT, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["items"]


def get_file_contents(access_token: str, download_url: str) -> str:
    """ Get file contents from download URL
    """
    headers = get_authorized_headers(access_token)
    response = requests.get(download_url, headers=headers)
    response.raise_for_status()
    return response.text


def get_repo_contents_recursive(
    access_token: str, repo_owner: str, repo_name: str, path: str = "", suffixes=None
) -> List[Dict]:
    """ Recursively get all files in a repo
    """
    url = REPO_CONTENTS_ENDPOINT.format(repo_owner=repo_owner, repo_name=repo_name, path=path)
    headers = get_authorized_headers(access_token)
    response = requests.get(url, headers=headers).json()

    def process_item(item: Dict) -> List[Dict]:
        """ Process item in response
        """
        if item["type"] == "file":
            if item["download_url"] is None:
                return []
            if suffixes is not None:
                if pathlib.Path(item["name"]).suffix not in suffixes:
                    return []
            file_content = get_file_contents(access_token, item["download_url"])
            return [{"item": item, "content": file_content}]
        elif item["type"] == "dir":
            return get_repo_contents_recursive(access_token, repo_owner, repo_name, item["path"])
        else:
            return []

    return [file for item in response for file in process_item(item)]
