#!/usr/bin/env bash
# Set up and deploy the arcade's worldwide leaderboard on your Cloudflare
# account (free plan). Safe to re-run: later runs just redeploy.
#
#   ./leaderboard/deploy.sh
#
# Needs Node.js. The first run opens a browser to log in to Cloudflare and,
# on a brand-new account, asks you to pick a workers.dev subdomain.
set -euo pipefail
cd "$(dirname "$0")"
WRANGLER=(npx --yes wrangler@4.138.0)

echo "== 1/4 Cloudflare login"
if ! "${WRANGLER[@]}" whoami 2>&1 | grep -q "associated with"; then
  "${WRANGLER[@]}" login
fi

echo "== 2/4 Database"
# Look for the database's id, not just the word: a comment mentions d1_databases.
if ! grep -q '"database_id"' wrangler.jsonc; then
  "${WRANGLER[@]}" d1 create kp-leaderboard --binding DB --update-config
fi

echo "== 3/4 Tables"
"${WRANGLER[@]}" d1 execute kp-leaderboard --remote --file schema.sql --yes

echo "== 4/4 Deploy"
log=$(mktemp)
"${WRANGLER[@]}" deploy 2>&1 | tee "$log" || true
url=$(grep -oE 'https://[A-Za-z0-9.-]+\.workers\.dev' "$log" | head -1 || true)
if grep -q "register a workers.dev subdomain" "$log"; then
  # A brand-new account has no workers.dev subdomain yet, and wrangler can't
  # ask for one here (its output is piped). One click in the dashboard fixes it.
  rm -f "$log"
  echo
  echo "One more step: this Cloudflare account needs a workers.dev subdomain."
  echo "  1. Open https://dash.cloudflare.com -> Workers & Pages (left sidebar)."
  echo "  2. Pick a subdomain when asked (e.g. your name) and confirm."
  echo "  3. Run ./leaderboard/deploy.sh again."
  exit 1
fi
rm -f "$log"
if [ -z "$url" ]; then
  echo "Deployed, but couldn't read the URL above. Put it in web/js/config.js by hand." >&2
  exit 1
fi

# Point the site at it.
node -e '
  const fs = require("fs"), f = process.argv[1];
  const s = fs.readFileSync(f, "utf8").replace(/url: ".*?"/, `url: "${process.argv[2]}"`);
  fs.writeFileSync(f, s);
' ../web/js/config.js "$url"

echo
echo "Leaderboard live at $url"
echo "web/js/config.js now points at it. Commit and push to put it on the site:"
echo "  git add leaderboard/wrangler.jsonc web/js/config.js && git commit -m 'Turn on the worldwide leaderboard' && git push"
