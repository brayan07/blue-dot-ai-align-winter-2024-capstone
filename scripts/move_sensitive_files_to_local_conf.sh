# Script to be used by CI to move sensitive files to local_conf directory

# Set up this file to fail if any command fails
set -e

# If a file called credentials.yml, move it to debate-for-ai-alignment/conf/local/credentials.yml,
# otherwise throw an error
if [ -f credentials.yml ]; then
    mv credentials.yml debate-for-ai-alignment/conf/local/credentials.yml
else
    echo "credentials.yml not found"
    exit 1
fi

# If a file called parameters.yml, move it to debate-for-ai-alignment/conf/local/parameters.yml,
# otherwise throw an error
if [ -f parameters.yml ]; then
    mv parameters.yml debate-for-ai-alignment/conf/local/parameters.yml
else
    echo "parameters.yml not found"
    exit 1
fi