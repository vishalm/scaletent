#!/bin/bash
echo "Stopping ScaleTent components..."

# Kill the processes in the terminal windows
osascript -e 'tell application "Terminal" to close (every window whose name contains "scaletent")'

# Stop Docker containers
docker-compose -f docker-compose.services.yml down

echo "ScaleTent stopped"
