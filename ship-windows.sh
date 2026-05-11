#!/bin/bash
set -e

# LingCode CLI Windows Shipping Script
# Automates the tagging and pushing to trigger a GitHub Action build.

# 1. Get current version from package.json
VERSION=$(node -p "require('./package.json').version")
TAG="v$VERSION"

echo "🚀 Preparing to ship LingCode CLI Windows $TAG..."

# 2. Ensure we are in a git repo
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "initial commit: lingcode cli windows foundation"
fi

# 3. Ask for remote URL if not set
if ! git remote | grep -q 'origin'; then
    echo "❌ No git remote 'origin' found."
    echo "Please create a new repository on GitHub and enter the URL below:"
    read -p "Remote URL (e.g., https://github.com/user/repo.git): " REMOTE_URL
    git remote add origin "$REMOTE_URL"
fi

# 4. Push code and tag
echo "Pushing code to origin..."
git push origin main || git push origin master

echo "Creating tag $TAG..."
git tag -a "$TAG" -m "Release $TAG" || true
git push origin "$TAG"

echo ""
echo "✅ Done! GitHub Actions is now building your Windows binary."
echo "Check progress here: $(git remote get-url origin | sed 's/\.git$//')/actions"
echo "Your files will appear here when finished: $(git remote get-url origin | sed 's/\.git$//')/releases"
