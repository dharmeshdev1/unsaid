#!/bin/sh
# Assemble the static demo site: the web interface plus a built index, marked so that the page
# runs the server in a browser worker instead of looking for a real one.
#   scripts/build_site.sh path/to/index path/to/site
set -eu
index="$1"
site="$2"
root="$(dirname "$0")/.."
rm -rf "$site"
mkdir -p "$site/demo"
cp -R "$root"/web/* "$site"/
cp "$index"/* "$site"/demo/
sed -i.orig 's/<body data-state="loading">/<body data-state="loading" data-source="demo">/' "$site/index.html"
rm "$site/index.html.orig"
touch "$site/.nojekyll"
du -sh "$site"
