#!/usr/bin/env bash

echo "========================================"
echo "RunPod Photobooth Worker Starting..."
echo "========================================"

# Environment info
echo "Model ID: ${MODEL_ID:-stablediffusionapi/turbovision_xl}"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create necessary directories
mkdir -p /tmp/photobooth
mkdir -p /runpod-volume

# Symlink RunPod volume if it exists
if [ -d "/runpod-volume" ]; then
    echo "Setting up RunPod volume symlinks..."

    # Remove existing cache directories
    rm -rf /root/.cache
    rm -rf /root/.insightface
    rm -rf /root/.local

    # Create symlinks to volume
    ln -sf /runpod-volume/.cache /root/.cache
    ln -sf /runpod-volume/.insightface /root/.insightface
    ln -sf /runpod-volume/.local /root/.local

    # Symlink model cache if exists
    if [ -d "/runpod-volume/models" ]; then
        ln -sf /runpod-volume/models /models
    fi

    echo "Volume symlinks created"
fi

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Health check endpoint setup
if [ ! -f "/workspace/health_check.py" ]; then
    cat > /workspace/health_check.py << 'EOF'
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "service": "photobooth-worker"}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), HealthHandler)
    server.serve_forever()
EOF
fi

# Start health check server in background
python /workspace/health_check.py &
HEALTH_PID=$!

echo "Health check server started (PID: $HEALTH_PID)"

# Cleanup function
cleanup() {
    echo "Shutting down worker..."
    kill $HEALTH_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Clear GPU memory before starting
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Start the main handler
echo "========================================"
echo "Starting RunPod Handler..."
echo "========================================"

cd /workspace
export PYTHONUNBUFFERED=1
python -u handler.py