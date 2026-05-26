#!/bin/bash
# sync-upstream.sh — 将上游最新内容同步到本地

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "🔄 Fetching upstream..."
git fetch upstream

echo "🔀 Merging upstream/main into main..."
git checkout main
git merge upstream/main --no-edit

echo "📦 Building site..."
node site/build.js

echo "🚀 Pushing to origin..."
git push origin main

echo "✅ Done! Upstream synced and site built."