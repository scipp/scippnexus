# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import argparse
import sys
from typing import List

import git
from packaging.version import InvalidVersion, Version, parse


def _get_releases() -> List[Version]:
    """Return reversed sorted list of release tag names."""
    tags = git.Repo(search_parent_directories=True).tags
    versions = []
    for t in tags:
        try:
            versions.append(parse(t.name))
        except InvalidVersion:
            pass
    return sorted(versions, reverse=True)


class VersionInfo:
    def __init__(self):
        self._releases = _get_releases()

    def _to_version(self, version) -> Version:
        if isinstance(version, str):
            try:
                return parse(version)
            except InvalidVersion:
                # When not building for a tagged release we may get, e.g., 'main'.
                # Pretend this means the current latest release.
                return self._releases[0]
        return version

    def minor_releases(self, first: str = '0.1') -> List[str]:
        """Return set of minor releases in the form '1.2'.

        `first` gives the first release to be included. By default '0.0' releases are
        not included.
        """
        first = parse(first)
        releases = [r for r in self._releases if r >= first]
        releases = sorted(set((r.major, r.minor) for r in releases), reverse=True)
        return [f'{major}.{minor}' for major, minor in releases]

    def is_latest(self, version: str) -> bool:
        """Return True if `version` has the same or larger major and minor as the
        latest release.
        """
        version = self._to_version(version)
        latest = self._releases[0]
        return (latest.major, latest.minor) <= (version.major, version.minor)

    def is_new(self, version: str) -> bool:
        """Return True if `version` is a new major or minor release."""
        version = self._to_version(version)
        releases = [r for r in self._releases if r != version]
        latest = releases[0]
        return (latest.major, latest.minor) < (version.major, version.minor)

    def target(self, version: str) -> str:
        version = self._to_version(version)
        if self.is_latest(version):
            return ''
        else:
            return f'release/{version.major}.{version.minor}'

    def replaced(self, version: str) -> str:
        version = self._to_version(version)
        # If we release 1.2 we want to find 1.1
        for release in self._releases:
            if (release.major, release.minor) < (version.major, version.minor):
                return release


def main(action: str, version: str) -> int:
    info = VersionInfo()
    if action == 'is-latest':
        print(info.is_latest(version))
    elif action == 'is-new':
        print(info.is_new(version))
    elif action == 'get-replaced':
        print(info.replaced(version))
    elif action == 'get-target':
        print(info.target(version))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--action',
        choices=['is-latest', 'is-new', 'get-replaced', 'get-target'],
        required=True,
        help='Action to perform: Check whether this major or minor '
        'release exists or is new (is-latest), check whether this is a '
        'new major or minor release (is-new), get the version this is '
        'replacing (get-replaced), get the target folder for '
        'publishing the docs (get-target). In all cases the '
        'patch/micro version is ignored.',
    )
    parser.add_argument(
        '--version', dest='version', required=True, help='Version the action refers to'
    )

    args = parser.parse_args()
    sys.exit(main(**vars(args)))
