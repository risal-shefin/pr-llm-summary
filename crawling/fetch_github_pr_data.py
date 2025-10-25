#!/usr/bin/env python3
"""
Script to fetch PR data from top 100 starred Python projects on GitHub.
Exports comprehensive PR information including diffs and commit details.
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import csv
import argparse
from pathlib import Path
import re


class GitHubPRDataCollector:
    """Collector for GitHub Pull Request data from top Python repositories."""
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize the collector with GitHub API credentials.
        
        Args:
            github_token: GitHub Personal Access Token for API authentication
        """
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        if not self.github_token:
            print("WARNING: No GitHub token provided. API rate limits will be very restrictive.")
            print("Set GITHUB_TOKEN environment variable or pass token as argument.")
        
        self.base_url = "https://api.github.com"
        self.graphql_url = "https://api.github.com/graphql"
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
        }
        if self.github_token:
            self.headers['Authorization'] = f'token {self.github_token}'
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def graphql_query(self, query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL query against GitHub API.
        
        Args:
            query: GraphQL query string
            variables: Query variables dictionary
            
        Returns:
            GraphQL response data
        """
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
        
        try:
            response = self.session.post(self.graphql_url, json=payload)
            
            # Handle rate limit exceeded
            if response.status_code == 403:
                print("GraphQL rate limit exceeded. Checking reset time...")
                rate_limit = self.check_rate_limit()
                if rate_limit:
                    graphql_info = rate_limit.get('resources', {}).get('graphql', {})
                    reset_time = graphql_info.get('reset', 0)
                    wait_time = max(0, reset_time - time.time()) + 5
                    
                    if wait_time > 0:
                        print(f"Waiting {wait_time:.0f} seconds for GraphQL rate limit to reset...")
                        time.sleep(wait_time)
                        # Retry the request after waiting
                        response = self.session.post(self.graphql_url, json=payload)
                    else:
                        print("Rate limit should be reset, but still getting 403. Waiting 60 seconds...")
                        time.sleep(60)
                        response = self.session.post(self.graphql_url, json=payload)
            
            response.raise_for_status()
            result = response.json()
            
            if 'errors' in result:
                print(f"GraphQL errors: {result['errors']}")
                return {}
            
            return result.get('data', {})
        except requests.exceptions.RequestException as e:
            print(f"GraphQL query error: {e}")
            return {}
    
    def check_rate_limit(self) -> Dict[str, Any]:
        """Check current GitHub API rate limit status."""
        try:
            response = self.session.get(f"{self.base_url}/rate_limit")
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not check rate limit: {e}")
        return {}
    
    def wait_for_rate_limit(self):
        """Wait if rate limit is exceeded."""
        rate_limit = self.check_rate_limit()
        if rate_limit:
            # Check both REST and GraphQL rate limits
            core = rate_limit.get('resources', {}).get('core', {})
            graphql = rate_limit.get('resources', {}).get('graphql', {})
            
            # Use the lower of the two
            core_remaining = core.get('remaining', 0)
            graphql_remaining = graphql.get('remaining', 0)
            
            # Check if either is low
            if core_remaining < 10:
                reset_time = core.get('reset', 0)
                wait_time = max(0, reset_time - time.time()) + 5
                print(f"REST API rate limit low ({core_remaining} remaining). Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
            
            if graphql_remaining < 10:
                reset_time = graphql.get('reset', 0)
                wait_time = max(0, reset_time - time.time()) + 5
                print(f"GraphQL rate limit low ({graphql_remaining} remaining). Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
    
    def get_top_python_repos(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch top starred Python repositories from GitHub using GraphQL.
        
        Args:
            count: Number of top repositories to fetch
            
        Returns:
            List of repository data dictionaries
        """
        repos = []
        per_page = min(100, count)
        
        print(f"Fetching top {count} Python repositories...")
        
        query = """
        query($searchQuery: String!, $count: Int!, $cursor: String) {
          search(query: $searchQuery, type: REPOSITORY, first: $count, after: $cursor) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              ... on Repository {
                nameWithOwner
                name
                owner {
                  login
                }
                stargazerCount
              }
            }
          }
        }
        """
        
        cursor = None
        while len(repos) < count:
            self.wait_for_rate_limit()
            
            variables = {
                'searchQuery': 'language:python sort:stars',
                'count': per_page,
                'cursor': cursor
            }
            
            data = self.graphql_query(query, variables)
            
            if not data or 'search' not in data:
                break
            
            search_result = data['search']
            nodes = search_result.get('nodes', [])
            
            for node in nodes:
                if node and len(repos) < count:
                    repos.append({
                        'name': node.get('name', ''),
                        'owner': {'login': node.get('owner', {}).get('login', '')},
                        'stargazers_count': node.get('stargazerCount', 0)
                    })
            
            print(f"  Fetched {len(repos)}/{count} repos")
            
            page_info = search_result.get('pageInfo', {})
            if not page_info.get('hasNextPage') or len(repos) >= count:
                break
            
            cursor = page_info.get('endCursor')
        
        return repos[:count]
    
    def get_pull_requests_graphql(self, owner: str, repo: str, 
                                  max_prs: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch pull requests with all details using GraphQL (single query per batch).
        This replaces multiple REST API calls per PR with one GraphQL query.
        
        Args:
            owner: Repository owner
            repo: Repository name
            max_prs: Maximum number of PRs to fetch
            
        Returns:
            List of complete PR data dictionaries with files and commits
        """
        query = """
        query($owner: String!, $repo: String!, $count: Int!, $cursor: String) {
          repository(owner: $owner, name: $repo) {
            pullRequests(first: $count, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                number
                title
                body
                state
                merged
                author {
                  login
                  ... on User {
                    id
                  }
                }
                labels(first: 20) {
                  nodes {
                    name
                  }
                }
                createdAt
                updatedAt
                closedAt
                mergedAt
                additions
                deletions
                changedFiles
                commitsCount: commits(first: 1) {
                  totalCount
                }
                baseRefName
                headRefName
                files(first: 100) {
                  nodes {
                    path
                    additions
                    deletions
                    changeType
                  }
                }
                commitsList: commits(first: 100) {
                  nodes {
                    commit {
                      oid
                      message
                      author {
                        name
                        date
                      }
                      additions
                      deletions
                      changedFiles
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        prs = []
        cursor = None
        per_page = min(25, max_prs)  # GraphQL has lower limits per query
        
        while len(prs) < max_prs:
            self.wait_for_rate_limit()
            
            variables = {
                'owner': owner,
                'repo': repo,
                'count': per_page,
                'cursor': cursor
            }
            
            data = self.graphql_query(query, variables)
            
            if not data or 'repository' not in data:
                break
            
            repository = data['repository']
            if not repository or 'pullRequests' not in repository:
                break
            
            pull_requests = repository['pullRequests']
            nodes = pull_requests.get('nodes', [])
            
            if not nodes:
                break
            
            prs.extend(nodes)
            
            page_info = pull_requests.get('pageInfo', {})
            if not page_info.get('hasNextPage') or len(prs) >= max_prs:
                break
            
            cursor = page_info.get('endCursor')
        
        return prs[:max_prs]
    
    def get_pr_file_patches(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """
        Fetch file patches for a PR using REST API (GraphQL doesn't provide patches).
        This is the only remaining REST call needed per PR.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            
        Returns:
            List of files with patches
        """
        self.wait_for_rate_limit()
        
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PR #{pr_number} file patches: {e}")
            return []
    
    def get_pull_requests(self, owner: str, repo: str, 
                         state: str = 'all', 
                         max_prs: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch pull requests for a specific repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: PR state filter ('open', 'closed', 'all')
            max_prs: Maximum number of PRs to fetch
            
        Returns:
            List of PR data dictionaries
        """
        prs = []
        page = 1
        per_page = 100
        
        while len(prs) < max_prs:
            self.wait_for_rate_limit()
            
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
            params = {
                'state': state,
                'per_page': per_page,
                'page': page,
                'sort': 'created',
                'direction': 'desc'
            }
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                prs.extend(data)
                
                if len(data) < per_page:
                    break
                
                page += 1
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching PRs for {owner}/{repo}: {e}")
                break
        
        return prs[:max_prs]
    
    def get_pr_details(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        """
        Fetch detailed information for a specific PR.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            
        Returns:
            Detailed PR data dictionary
        """
        self.wait_for_rate_limit()
        
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PR #{pr_number} details: {e}")
            return {}
    
    def get_pr_commits(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """
        Fetch commits for a specific PR.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            
        Returns:
            List of commit data dictionaries
        """
        self.wait_for_rate_limit()
        
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/commits"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PR #{pr_number} commits: {e}")
            return []
    
    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """
        Fetch changed files for a specific PR.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            
        Returns:
            List of changed file data dictionaries
        """
        self.wait_for_rate_limit()
        
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PR #{pr_number} files: {e}")
            return []
    
    def get_commit_diff(self, owner: str, repo: str, commit_sha: str) -> str:
        """
        Fetch diff for a specific commit.
        
        Args:
            owner: Repository owner
            repo: Repository name
            commit_sha: Commit SHA
            
        Returns:
            Diff text
        """
        self.wait_for_rate_limit()
        
        url = f"{self.base_url}/repos/{owner}/{repo}/commits/{commit_sha}"
        headers = self.headers.copy()
        headers['Accept'] = 'application/vnd.github.v3.diff'
        
        try:
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching commit {commit_sha} diff: {e}")
            return ""
    
    def get_commit_files(self, owner: str, repo: str, commit_sha: str) -> List[Dict[str, Any]]:
        """
        Fetch file changes for a specific commit.
        
        Args:
            owner: Repository owner
            repo: Repository name
            commit_sha: Commit SHA
            
        Returns:
            List of file changes with patches
        """
        self.wait_for_rate_limit()
        
        url = f"{self.base_url}/repos/{owner}/{repo}/commits/{commit_sha}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            commit_data = response.json()
            
            # Extract file information
            files = []
            for f in commit_data.get('files', []):
                files.append({
                    'filename': f.get('filename', ''),
                    'status': f.get('status', ''),
                    'additions': f.get('additions', 0),
                    'deletions': f.get('deletions', 0),
                    'changes': f.get('changes', 0),
                    'patch': f.get('patch', ''),
                    'blob_url': f.get('blob_url', ''),
                    'raw_url': f.get('raw_url', ''),
                    'previous_filename': f.get('previous_filename', '')
                })
            
            return files
        except requests.exceptions.RequestException as e:
            print(f"Error fetching commit {commit_sha} files: {e}")
            return []
    
    def collect_pr_data(self, owner: str, repo: str, pr_number: int, pr_basic: Dict[str, Any] = None, 
                       fetch_commit_files: bool = False) -> Dict[str, Any]:
        """
        Collect comprehensive data for a single PR from GraphQL data.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            pr_basic: PR data from GraphQL query (required for GraphQL mode)
            fetch_commit_files: Whether to fetch file changes for each commit (WARNING: uses many API calls)
            
        Returns:
            Complete PR data dictionary
        """
        print(f"  Processing PR #{pr_number} from {owner}/{repo}")
        
        if not pr_basic:
            print(f"  Warning: No GraphQL data provided for PR #{pr_number}")
            return {}
        
        # Get file patches (only REST call needed - GraphQL doesn't provide patches)
        files_with_patches = self.get_pr_file_patches(owner, repo, pr_number)
        
        # Create patch lookup by filename
        patch_lookup = {f.get('filename', ''): f.get('patch', '') for f in files_with_patches}
        
        # Clean body text
        body_text = (pr_basic.get('body') or '').strip()
        body_text = re.sub(r'<!--.*?-->', '', body_text, flags=re.DOTALL)
        body_text = re.sub(r'\n{3,}', '\n\n', body_text).strip()
        
        # Extract labels
        labels = []
        if pr_basic.get('labels') and pr_basic['labels'].get('nodes'):
            labels = [label.get('name', '') for label in pr_basic['labels']['nodes']]
        
        # Extract files
        files = []
        if pr_basic.get('files') and pr_basic['files'].get('nodes'):
            for f in pr_basic['files']['nodes']:
                filename = f.get('path', '')
                # Map GraphQL changeType to REST status
                change_type = f.get('changeType', 'modified').lower()
                status_map = {'added': 'added', 'deleted': 'removed', 'modified': 'modified', 
                             'renamed': 'renamed', 'copied': 'copied', 'changed': 'modified'}
                status = status_map.get(change_type, 'modified')
                
                files.append({
                    'filename': filename,
                    'status': status,
                    'additions': f.get('additions', 0),
                    'deletions': f.get('deletions', 0),
                    'changes': f.get('additions', 0) + f.get('deletions', 0),
                    'patch': patch_lookup.get(filename, ''),
                    'blob_url': f"https://github.com/{owner}/{repo}/blob/{pr_basic.get('headRefName', '')}/{filename}"
                })
        
        # Extract commits with optional file changes using the aliased field
        commit_list = []
        if pr_basic.get('commitsList') and pr_basic['commitsList'].get('nodes'):
            for c in pr_basic['commitsList']['nodes']:
                commit_data = c.get('commit', {})
                author_data = commit_data.get('author', {})
                commit_sha = commit_data.get('oid', '')
                
                commit_entry = {
                    'sha': commit_sha,
                    'message': commit_data.get('message', ''),
                    'author': author_data.get('name', ''),
                    'date': author_data.get('date', ''),
                    # Basic stats from GraphQL (no extra API calls!)
                    'stats': {
                        'additions': commit_data.get('additions', 0),
                        'deletions': commit_data.get('deletions', 0),
                        'total': commit_data.get('additions', 0) + commit_data.get('deletions', 0)
                    },
                    'changed_files_count': commit_data.get('changedFiles', 0)
                }
                
                # Optionally fetch detailed file changes with patches (uses 1 API call per commit)
                if fetch_commit_files and commit_sha:
                    try:
                        commit_files = self.get_commit_files(owner, repo, commit_sha)
                        commit_entry['files'] = commit_files
                        # Update stats with actual values if available
                        if commit_files:
                            commit_entry['stats'] = {
                                'additions': sum(f.get('additions', 0) for f in commit_files),
                                'deletions': sum(f.get('deletions', 0) for f in commit_files),
                                'total': sum(f.get('changes', 0) for f in commit_files)
                            }
                    except Exception as e:
                        print(f"    Error fetching files for commit {commit_sha[:7]}: {e}")
                
                commit_list.append(commit_entry)
        
        # Get total commit count using the aliased field
        commits_count = 0
        if pr_basic.get('commitsCount') and pr_basic['commitsCount'].get('totalCount'):
            commits_count = pr_basic['commitsCount']['totalCount']
        elif commit_list:
            commits_count = len(commit_list)
        
        # Extract author info
        author_info = pr_basic.get('author', {})
        author = {
            'login': author_info.get('login', ''),
            'id': author_info.get('id', 0),
            'type': 'User'  # GraphQL query only returns User type
        }
        
        # Compile PR data
        return {
            'repository': f"{owner}/{repo}",
            'title': pr_basic.get('title', ''),
            'body': body_text,
            'number': pr_basic.get('number', 0),
            'state': pr_basic.get('state', '').lower(),
            'merged': pr_basic.get('merged', False),
            'author': author,
            'labels': labels,
            'created_at': pr_basic.get('createdAt', ''),
            'updated_at': pr_basic.get('updatedAt', ''),
            'closed_at': pr_basic.get('closedAt', ''),
            'merged_at': pr_basic.get('mergedAt', ''),
            'additions': pr_basic.get('additions', 0),
            'deletions': pr_basic.get('deletions', 0),
            'changed_files': pr_basic.get('changedFiles', 0),
            'commits': commits_count,
            'base_branch': pr_basic.get('baseRefName', ''),
            'head_branch': pr_basic.get('headRefName', ''),
            'files': files,
            'commit_list': commit_list
        }
    
    def collect_dataset(self, num_repos: int = 100, 
                       prs_per_repo: int = 10,
                       output_dir: str = 'github_pr_dataset',
                       fetch_commit_files: bool = False) -> List[Dict[str, Any]]:
        """
        Collect PR dataset from top GitHub Python repositories using GraphQL.
        
        Args:
            num_repos: Number of top repositories to process
            prs_per_repo: Number of PRs to collect per repository
            output_dir: Output directory for dataset files
            fetch_commit_files: Whether to fetch file changes for each commit 
                               (WARNING: uses many API calls - ~N commits per PR)
            
        Returns:
            List of all collected PR data
        """
        print(f"Starting dataset collection with GraphQL optimization...")
        print(f"Target: {num_repos} repositories, {prs_per_repo} PRs per repo")
        print(f"Fetch commit-level files: {fetch_commit_files}")
        
        # Estimate API calls
        estimated_graphql = num_repos + (num_repos * ((prs_per_repo + 24) // 25))  # Repos + PR batches
        estimated_rest = num_repos * prs_per_repo  # 1 per PR for file patches
        if fetch_commit_files:
            avg_commits_per_pr = 20  # Conservative estimate
            estimated_rest += num_repos * prs_per_repo * avg_commits_per_pr
        
        print(f"\nEstimated API calls:")
        print(f"  GraphQL: ~{estimated_graphql} calls")
        print(f"  REST: ~{estimated_rest} calls")
        if fetch_commit_files:
            print(f"  WARNING: Commit file fetching is ENABLED - will use many REST API calls!")
            print(f"  Consider running without --fetch-commit-files first.")
        print()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get top repositories using GraphQL
        repos = self.get_top_python_repos(num_repos)
        print(f"\nFound {len(repos)} repositories")
        
        all_pr_data = []
        
        # Process each repository
        for idx, repo in enumerate(repos, 1):
            owner = repo.get('owner', {}).get('login', '')
            repo_name = repo.get('name', '')
            stars = repo.get('stargazers_count', 0)

            print(f"\n[{idx}/{len(repos)}] Processing {owner}/{repo_name} ({stars:,} stars)")

            # Get PRs with all details using GraphQL (1 query instead of 3+ per PR)
            prs = self.get_pull_requests_graphql(owner, repo_name, max_prs=prs_per_repo)
            print(f"  Found {len(prs)} PRs", flush=True)
            
            # Process each PR (1 REST call for patches + optional N calls for commit files)
            for pr in prs[:prs_per_repo]:
                pr_number = pr.get('number', 0)
                try:
                    # GraphQL already fetched most data, just need patches (and optionally commit files)
                    pr_data = self.collect_pr_data(owner, repo_name, pr_number, pr_basic=pr, 
                                                   fetch_commit_files=fetch_commit_files)
                    if pr_data:
                        all_pr_data.append(pr_data)
                except Exception as e:
                    print(f"  Error processing PR #{pr_number}: {e}")
                    continue
        
        print(f"\n\nDataset collection complete!")
        print(f"Total PRs collected: {len(all_pr_data)}")
        
        # Save final dataset
        self._save_dataset(all_pr_data, output_path)
        
        return all_pr_data
    
    def _save_dataset(self, pr_data: List[Dict[str, Any]], output_path: Path):
        """Save the complete dataset in multiple formats."""
        
        # Save as JSON
        json_file = output_path / 'pr_dataset_complete.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(pr_data, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {json_file}")
        
        # Save as JSONL (one PR per line)
        jsonl_file = output_path / 'pr_dataset_complete.jsonl'
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for pr in pr_data:
                f.write(json.dumps(pr, ensure_ascii=False) + '\n')
        print(f"Saved JSONL: {jsonl_file}")
        
        # Save summary as CSV
        if not pr_data:
            return
            
        csv_file = output_path / 'pr_dataset_summary.csv'
        fieldnames = [
            'repository', 'number', 'title', 'state', 'merged',
            'author', 'created_at', 'additions', 'deletions',
            'changed_files', 'commits', 'labels'
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for pr in pr_data:
                writer.writerow({
                    'repository': pr.get('repository', ''),
                    'number': pr.get('number', 0),
                    'title': pr.get('title', ''),
                    'state': pr.get('state', ''),
                    'merged': pr.get('merged', False),
                    'author': pr.get('author', {}).get('login', ''),
                    'created_at': pr.get('created_at', ''),
                    'additions': pr.get('additions', 0),
                    'deletions': pr.get('deletions', 0),
                    'changed_files': pr.get('changed_files', 0),
                    'commits': pr.get('commits', 0),
                    'labels': ', '.join(pr.get('labels', []))
                })
        print(f"Saved CSV summary: {csv_file}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Collect PR data from top GitHub Python repositories'
    )
    parser.add_argument(
        '--token',
        help='GitHub Personal Access Token (or set GITHUB_TOKEN env var)',
        default=None
    )
    parser.add_argument(
        '--repos',
        type=int,
        default=100,
        help='Number of top repositories to process (default: 100)'
    )
    parser.add_argument(
        '--prs-per-repo',
        type=int,
        default=10,
        help='Number of PRs to collect per repository (default: 10)'
    )
    parser.add_argument(
        '--output-dir',
        default='github_pr_dataset_exp',
        help='Output directory for dataset files (default: github_pr_dataset)'
    )
    parser.add_argument(
        '--fetch-commit-files',
        action='store_true',
        help='Fetch file changes for each commit (WARNING: uses ~N API calls per PR where N=number of commits)'
    )
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = GitHubPRDataCollector(github_token=args.token)
    
    # Check rate limit
    rate_limit = collector.check_rate_limit()
    if rate_limit:
        core = rate_limit.get('resources', {}).get('core', {})
        print(f"API Rate Limit: {core.get('remaining', 0)}/{core.get('limit', 0)}")
        if core.get('remaining', 0) < 100:
            print("WARNING: Low rate limit. Consider waiting or using a token with higher limits.")
    
    # Collect dataset
    try:
        collector.collect_dataset(
            num_repos=args.repos,
            prs_per_repo=args.prs_per_repo,
            output_dir=args.output_dir,
            fetch_commit_files=args.fetch_commit_files
        )
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
    except Exception as e:
        print(f"\n\nError during collection: {e}")
        raise


if __name__ == '__main__':
    main()
