#!/usr/bin/env bash
# ActiveLearningAI — Cloud server bootstrap script
#
# Run on a fresh Ubuntu 22.04+ server:
#   curl -sL https://raw.githubusercontent.com/.../deploy/cloud-init.sh | bash
# Or:
#   scp deploy/cloud-init.sh user@server: && ssh user@server bash cloud-init.sh
#
# What it does:
#   1. Install Docker + Docker Compose
#   2. Configure firewall (SSH + dashboard only)
#   3. Create data directories
#   4. Install monitoring tools
#   5. Set up swap (for safety on high-RAM machines)

set -euo pipefail

echo "=== ActiveLearningAI Cloud Setup ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "RAM:  $(free -g | awk '/^Mem:/{print $2}')G"
echo "CPU:  $(nproc) cores"

# 1. System updates
echo "--- Installing system packages ---"
sudo apt-get update -q
sudo apt-get install -y -q \
    ca-certificates curl gnupg lsb-release \
    git htop tmux jq ffmpeg sqlite3 \
    python3 python3-pip python3-venv

# 2. Docker
if ! command -v docker &>/dev/null; then
    echo "--- Installing Docker ---"
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
fi

# Docker Compose (v2 plugin)
if ! docker compose version &>/dev/null; then
    echo "--- Installing Docker Compose v2 ---"
    sudo apt-get install -y docker-compose-plugin
fi

echo "Docker: $(docker --version)"
echo "Compose: $(docker compose version)"

# 2b. Python tools (yt-dlp for video downloads, uv for sensory-gateway deps)
echo "--- Installing Python tools ---"
pip3 install --break-system-packages yt-dlp 2>/dev/null || pip3 install yt-dlp
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Firewall — allow SSH + dashboard only
echo "--- Configuring firewall ---"
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8080/tcp comment "Dashboard"
sudo ufw --force enable
echo "Firewall: SSH + 8080 (dashboard) open"

# 4. Data directories
echo "--- Creating data directories ---"
sudo mkdir -p /data/sqlite /data/videos /data/benchmarks /data/backups
sudo chown -R "$USER:$USER" /data

# 5. Swap (4GB safety net — STDP can spike memory temporarily)
if [ ! -f /swapfile ]; then
    echo "--- Creating 4GB swap ---"
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# 6. Kernel tuning for large memory workloads
echo "--- Kernel tuning ---"
cat <<'SYSCTL' | sudo tee /etc/sysctl.d/99-neuromorphic.conf
# Allow more memory to be committed (STDP peak allocation)
vm.overcommit_memory = 1
# Reduce swappiness (prefer RAM, swap is safety only)
vm.swappiness = 10
# Increase max open files (NATS connections)
fs.file-max = 1000000
SYSCTL
sudo sysctl -p /etc/sysctl.d/99-neuromorphic.conf

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Upload codebase:  ./deploy/sync.sh user@this-server"
echo "  2. Configure:        cp deploy/.env.cloud .env"
echo "  3. Build & start:    docker compose -f docker-compose.yml -f deploy/docker-compose.cloud.yml up --build -d"
echo "  4. Download videos:  ./deploy/download-videos.sh"
echo "  5. Monitor:          ./deploy/monitor.sh"
