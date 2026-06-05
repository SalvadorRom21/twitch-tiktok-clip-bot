#!/usr/bin/env bash
# Recreate the GitHub repo and push this project.
# Run from the repo root after GitHub CLI is authenticated: gh auth login

set -euo pipefail

OWNER="${GITHUB_OWNER:-SalvadorRom21}"
REPO="${GITHUB_REPO:-twitch-tiktok-clip-bot}"
VISIBILITY="${GITHUB_VISIBILITY:-public}"

echo "Creating GitHub repo: ${OWNER}/${REPO}"
gh repo create "${OWNER}/${REPO}" \
  --"${VISIBILITY}" \
  --source=. \
  --remote=origin \
  --description "Automated Twitch clip to TikTok short editor with AI analysis" \
  --push

echo "Done. Repo: https://github.com/${OWNER}/${REPO}"
