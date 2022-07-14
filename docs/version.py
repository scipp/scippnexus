import sys
from typing import List
from packaging.version import parse, Version
import requests
import argparse


def get_releases(repo, organization: str = 'scipp') -> List[Version]:
    """Return reversed sorted list of release tag names."""
    r = requests.get(f'https://api.github.com/repos/{organization}/{repo}/releases')
    data = r.json()
    return sorted([parse(e['tag_name']) for e in data if not e['draft']], reverse=True)


def is_latest(releases, version):
    if isinstance(version, str):
        version = parse(version)
    latest = releases[0]
    return (latest.major, latest.minor) <= (version.major, version.minor)


# TODO refactor entirly to return struct or similar, for use from Python
def main(repo: str, action: str, version: str) -> int:
    releases = get_releases(repo=repo)
    version = parse(version)
    if version in releases:
        releases.remove(version)
    latest = releases[0]
    if action == 'is-latest':  # This major or minor release exists
        print(is_latest(releases, version))
    elif action == 'is-new':  # New major or minor release
        print((latest.major, latest.minor) < (version.major, version.minor))
    elif action == 'get-replaced':
        # If we release 1.2 we want to find 1.1
        for release in releases:
            if (release.major, release.minor) != (version.major, version.minor):
                print(release)
                break
    elif action == 'get-target':
        # TODO change to non-test folders
        if is_latest(releases, version):
            print('release-test/stable')
        else:
            print(f'release-test/{version.major}.{version.minor}')

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--repo', dest='repo', required=True, help='Repository name')
    parser.add_argument('--action',
                        choices=['is-latest', 'is-new', 'get-replaced', 'get-target'],
                        required=True,
                        help='Action to perform: Check whether this major or minor '
                        'release exists (is-latest), check whether this is a new major '
                        'or minor release (is-new), get the version this is replacing '
                        '(get-replaced). In all cases the patch/micro version is '
                        'ignored.')
    parser.add_argument('--version',
                        dest='version',
                        required=True,
                        help='Version the action refers to')

    args = parser.parse_args()
    sys.exit(main(**vars(args)))
