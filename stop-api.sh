# Stop and remove the containers
docker-compose -f docker-compose.services.yml down

# Remove any existing volume data if needed
docker volume rm scaletent_mongodb_data