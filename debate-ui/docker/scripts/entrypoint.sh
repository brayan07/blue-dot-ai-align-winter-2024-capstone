# Entrypoint for the debate-ui container in production

# Set up this file to fail if any command fails
set -e

# Try to find production configuration files in /etc/secrets
sh /app/debate-ui/docker/scripts/move_sensitive_files_to_local_conf.sh

# If a command is passed to the entrypoint, execute it, otherwise start the server
if [ $# -eq 0 ]; then
    # Start the server
    exec gunicorn -b "0.0.0.0:8050" "src.app:server"
else
    # Execute the command
    exec "$@"
fi






