#!/usr/bin/env bash
# Build the write-up site and publish it to the gh-pages branch (served at
# https://foogunlana.github.io/emotion-concepts/). The branch holds only the built output, so each publish
# replaces it with a single fresh commit.
#
#   bash site/publish.sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --with markdown python site/build.py

src_commit=$(git rev-parse --short HEAD)
remote=$(git remote get-url origin)
tmp=$(mktemp -d)
cp -R site/_build/. "$tmp"
(
  cd "$tmp"
  git init -q -b gh-pages
  git add -A
  git commit -q -m "Publish write-up (built from ${src_commit})"
  git push -q -f "$remote" gh-pages
)
rm -rf "$tmp"
echo "published → https://foogunlana.github.io/emotion-concepts/ (from ${src_commit})"
