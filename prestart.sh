# This is a shell script to be run before starting the app
#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y --no-install-recommends \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  libx11-6

# Clean up
apt-get clean && rm -rf /var/lib/apt/lists/*

# Continue with normal startup
exec "$@"