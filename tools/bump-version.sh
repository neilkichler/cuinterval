#!/bin/sh
if [ $# -eq 0 ]
  then
    echo "Usage: ./bump-version.sh <version>"
    exit
fi

VERSION="$1"

# --- 1. Check format: vMAJOR.MINOR.PATCH ---
if ! echo "$VERSION" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: Version '$VERSION' is not in format vMAJOR.MINOR.PATCH (example: v1.2.3)."
    exit 1
fi

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "$0")")
BASE_DIR=${SCRIPT_DIR}/..
ENV_FILE=${BASE_DIR}/docs/.env.production
HEADER="# This is the version tag of the code used when running pnpm run build\n"
VERSION_TXT="VERSION=\"${VERSION}\""
TXT="${HEADER}${VERSION_TXT}"
echo "$TXT" > "${ENV_FILE}"

# Update CMakeLists.txt version (without 'v' prefix)
CMAKE_FILE=${BASE_DIR}/CMakeLists.txt
# Remove 'v' prefix if present
CLEAN_VERSION=${VERSION#v}
# Replace the VERSION line inside the project() block (indented line with project VERSION)
sed -i "s/^    VERSION [0-9]\+\.[0-9]\+\.[0-9]\+$/    VERSION ${CLEAN_VERSION}/" "${CMAKE_FILE}"

# Update README.md versions
README_FILE=${BASE_DIR}/README.md
# Replace version with 'v' prefix (for git tags/releases)
sed -i "s/v[0-9]\+\.[0-9]\+\.[0-9]\+/${VERSION}/g" "${README_FILE}"
# Replace version without 'v' prefix (for package managers like CPMAddPackage)
sed -i "s/@[0-9]\+\.[0-9]\+\.[0-9]\+/@${CLEAN_VERSION}/g" "${README_FILE}"

# Add changes to git and commit
git -C "$BASE_DIR" add "${ENV_FILE}" "${CMAKE_FILE}" "${README_FILE}"
git -C "$BASE_DIR" commit -m "bump version to ${VERSION}"
git tag -a "${VERSION}" -m "Release version ${VERSION}"