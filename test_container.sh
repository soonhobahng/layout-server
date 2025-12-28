#!/bin/bash

echo "Testing Docker container startup..."
docker run --rm -d --name gslayout-test -p 8000:8000 gslayout-parser &
CONTAINER_PID=$!

echo "Waiting for container to start..."
sleep 10

echo "Testing API response..."
curl -f http://localhost:8000/ || echo "API test failed"

echo "Checking container logs..."
docker logs gslayout-test

echo "Stopping test container..."
docker stop gslayout-test

echo "Test complete!"