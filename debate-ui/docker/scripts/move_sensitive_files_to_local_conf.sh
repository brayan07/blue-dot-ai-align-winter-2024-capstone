# Script to be used by CI to move sensitive files to local_conf directory

# Set up this file to fail if any command fails
set -e

# If a file called credentials.yml, move it to debate-for-ai-alignment/conf/local/credentials.yml
if [ -f /etc/secrets/credentials.yml ]; then
    echo "Moving production credentials.yml to conf/local..."
    cp /etc/secrets/credentials.yml /app/debate-for-ai-alignment/conf/local/credentials.yml
else
    echo "No production credentials.yml found..."
fi

# If a file called parameters.yml, move it to debate-for-ai-alignment/conf/local/parameters.yml
if [ -f /etc/secrets/parameters.yml ]; then
    echo "Moving production parameters.yml to conf/local..."
    cp /etc/secrets/parameters.yml /app/debate-for-ai-alignment/conf/local/parameters.yml
else
    echo "No production parameters.yml found..."
fi
