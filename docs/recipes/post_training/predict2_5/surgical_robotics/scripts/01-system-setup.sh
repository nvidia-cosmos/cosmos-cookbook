#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -eo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}System Setup for Cosmos-H-Surgical-Simulator${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "This script will:"
echo "  1. Configure Docker & containerd to use largest drive"
echo "  2. Clean up logs to free disk space"
echo "  3. Configure DNS for Docker"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
   exit 1
fi

# Find largest mounted drive
echo -e "\n${YELLOW}Finding largest mounted drive...${NC}"
LARGEST_MOUNT=$(df -h --output=size,target -x tmpfs -x devtmpfs -x squashfs -x overlay | \
    tail -n +2 | \
    awk '{print $1 " " $2}' | \
    sort -hr | \
    head -1)

LARGEST_SIZE=$(echo "$LARGEST_MOUNT" | awk '{print $1}')
LARGEST_PATH=$(echo "$LARGEST_MOUNT" | awk '{print $2}')

echo "Largest drive: $LARGEST_PATH ($LARGEST_SIZE)"

# Determine target directories
if [ "$LARGEST_PATH" = "/" ]; then
    NEW_DOCKER_DIR="/var/lib/docker"
    NEW_CONTAINERD_DIR="/var/lib/containerd"
    echo -e "${YELLOW}Largest drive is root - using default locations${NC}"
else
    NEW_DOCKER_DIR="${LARGEST_PATH}/docker"
    NEW_CONTAINERD_DIR="${LARGEST_PATH}/containerd"
fi

echo "Target Docker directory: $NEW_DOCKER_DIR"
echo "Target containerd directory: $NEW_CONTAINERD_DIR"

# Show disk usage
echo -e "\n${YELLOW}Current disk usage:${NC}"
df -h / "$LARGEST_PATH" 2>/dev/null | grep -v Filesystem

read -p "Continue with system setup? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# =============================================================================
# STEP 1: Configure Docker
# =============================================================================
echo -e "\n${GREEN}=== Step 1: Configuring Docker ===${NC}"

CURRENT_DOCKER_DIR=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $4}' || echo "/var/lib/docker")
echo "Current Docker directory: $CURRENT_DOCKER_DIR"

if [ "$CURRENT_DOCKER_DIR" != "$NEW_DOCKER_DIR" ]; then
    echo -e "${YELLOW}Moving Docker to $NEW_DOCKER_DIR...${NC}"

    # Stop services
    systemctl stop docker 2>/dev/null || true
    systemctl stop docker.socket 2>/dev/null || true
    systemctl stop containerd 2>/dev/null || true
    sleep 2

    # Create new directory
    mkdir -p "$NEW_DOCKER_DIR"

    # Move existing data if it exists
    if [ -d "$CURRENT_DOCKER_DIR" ] && [ "$(ls -A "$CURRENT_DOCKER_DIR" 2>/dev/null)" ]; then
        echo -e "${YELLOW}Moving Docker data (this may take a while)...${NC}"
        rsync -aP "$CURRENT_DOCKER_DIR/" "$NEW_DOCKER_DIR/"
        rm -rf "$CURRENT_DOCKER_DIR"
    fi

    # Update daemon.json with DNS fix
    DAEMON_JSON="/etc/docker/daemon.json"

    if [ -f "$DAEMON_JSON" ]; then
        cp "$DAEMON_JSON" "${DAEMON_JSON}.backup.$(date +%Y%m%d_%H%M%S)"
        python3 -c "
import json
with open('$DAEMON_JSON', 'r') as f:
    config = json.load(f)
config['data-root'] = '$NEW_DOCKER_DIR'
config['dns'] = ['8.8.8.8', '1.1.1.1']
with open('$DAEMON_JSON', 'w') as f:
    json.dump(config, f, indent=4)
"
    else
        cat > "$DAEMON_JSON" <<EOF
{
    "data-root": "$NEW_DOCKER_DIR",
    "dns": ["8.8.8.8", "1.1.1.1"]
}
EOF
    fi

    echo "Updated Docker configuration:"
    cat "$DAEMON_JSON"
    echo -e "${GREEN}✓ Docker configuration updated${NC}"
else
    echo -e "${GREEN}✓ Docker already using correct location${NC}"

    # Still add DNS fix if missing
    DAEMON_JSON="/etc/docker/daemon.json"
    if [ -f "$DAEMON_JSON" ]; then
        if ! grep -q '"dns"' "$DAEMON_JSON"; then
            echo -e "${YELLOW}Adding DNS configuration...${NC}"
            cp "$DAEMON_JSON" "${DAEMON_JSON}.backup.$(date +%Y%m%d_%H%M%S)"
            python3 -c "
import json
with open('$DAEMON_JSON', 'r') as f:
    config = json.load(f)
config['dns'] = ['8.8.8.8', '1.1.1.1']
with open('$DAEMON_JSON', 'w') as f:
    json.dump(config, f, indent=4)
"
        fi
    else
        cat > "$DAEMON_JSON" <<EOF
{
    "data-root": "$NEW_DOCKER_DIR",
    "dns": ["8.8.8.8", "1.1.1.1"]
}
EOF
    fi
fi

# =============================================================================
# STEP 2: Configure Containerd
# =============================================================================
echo -e "\n${GREEN}=== Step 2: Configuring Containerd ===${NC}"

CONTAINERD_CONFIG="/etc/containerd/config.toml"
CURRENT_CONTAINERD_DIR="/var/lib/containerd"
if [ -f "$CONTAINERD_CONFIG" ]; then
    CONFIG_ROOT=$(grep "^root = " "$CONTAINERD_CONFIG" | sed 's/root = "\(.*\)"/\1/' | tr -d '"' || echo "")
    if [ -n "$CONFIG_ROOT" ]; then
        CURRENT_CONTAINERD_DIR="$CONFIG_ROOT"
    fi
fi

echo "Current containerd directory: $CURRENT_CONTAINERD_DIR"

if [ "$CURRENT_CONTAINERD_DIR" != "$NEW_CONTAINERD_DIR" ]; then
    echo -e "${YELLOW}Moving containerd to $NEW_CONTAINERD_DIR...${NC}"

    # Create new directory
    mkdir -p "$NEW_CONTAINERD_DIR"

    # Move existing data if it exists
    if [ -d "$CURRENT_CONTAINERD_DIR" ] && [ "$(ls -A "$CURRENT_CONTAINERD_DIR" 2>/dev/null)" ]; then
        echo -e "${YELLOW}Moving containerd data (this may take a while)...${NC}"
        rsync -aP "$CURRENT_CONTAINERD_DIR/" "$NEW_CONTAINERD_DIR/"
        rm -rf "$CURRENT_CONTAINERD_DIR"
    fi

    # Update containerd config
    if [ -f "$CONTAINERD_CONFIG" ]; then
        cp "$CONTAINERD_CONFIG" "${CONTAINERD_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"

        if grep -q "^root = " "$CONTAINERD_CONFIG"; then
            sed -i "s|^root = .*|root = \"$NEW_CONTAINERD_DIR\"|" "$CONTAINERD_CONFIG"
        elif grep -q "^#root = " "$CONTAINERD_CONFIG"; then
            sed -i "s|^#root = .*|root = \"$NEW_CONTAINERD_DIR\"|" "$CONTAINERD_CONFIG"
        else
            sed -i "/^disabled_plugins/a root = \"$NEW_CONTAINERD_DIR\"" "$CONTAINERD_CONFIG"
        fi
    else
        mkdir -p /etc/containerd
        cat > "$CONTAINERD_CONFIG" <<EOF
disabled_plugins = ["cri"]
root = "$NEW_CONTAINERD_DIR"
EOF
    fi

    echo "Updated containerd configuration"
    echo -e "${GREEN}✓ Containerd configuration updated${NC}"
else
    echo -e "${GREEN}✓ Containerd already using correct location${NC}"
fi

# =============================================================================
# STEP 3: Clean up logs
# =============================================================================
echo -e "\n${GREEN}=== Step 3: Cleaning up logs ===${NC}"

echo -e "${YELLOW}Current journal size:${NC}"
journalctl --disk-usage

echo -e "${YELLOW}Vacuuming journal logs to 1GB...${NC}"
journalctl --vacuum-size=1G

echo -e "${YELLOW}Truncating syslog files...${NC}"
truncate -s 0 /var/log/syslog 2>/dev/null || true
truncate -s 0 /var/log/syslog.1 2>/dev/null || true

echo -e "${YELLOW}Cleaning apt cache...${NC}"
apt clean

# Configure journal max size permanently
JOURNALD_CONF="/etc/systemd/journald.conf"
if [ ! -f "${JOURNALD_CONF}.backup" ]; then
    cp "$JOURNALD_CONF" "${JOURNALD_CONF}.backup"
fi

if grep -q "^SystemMaxUse=" "$JOURNALD_CONF"; then
    sed -i 's/^SystemMaxUse=.*/SystemMaxUse=1G/' "$JOURNALD_CONF"
else
    if grep -q "^\[Journal\]" "$JOURNALD_CONF"; then
        sed -i '/^\[Journal\]/a SystemMaxUse=1G' "$JOURNALD_CONF"
    else
        echo -e "\n[Journal]\nSystemMaxUse=1G" >> "$JOURNALD_CONF"
    fi
fi

systemctl restart systemd-journald

echo -e "${GREEN}✓ Logs cleaned and configured${NC}"

# =============================================================================
# Start services
# =============================================================================
echo -e "\n${GREEN}=== Starting Docker services ===${NC}"
systemctl start containerd
sleep 2
systemctl start docker
sleep 2

# =============================================================================
# Verification
# =============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Verification${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Docker configuration:${NC}"
docker info 2>/dev/null | grep -E "Docker Root Dir|Storage Driver"

echo -e "\n${YELLOW}Final disk usage:${NC}"
df -h / "$LARGEST_PATH" 2>/dev/null | grep -v Filesystem

echo -e "\n${YELLOW}Journal size:${NC}"
journalctl --disk-usage

# =============================================================================
# Configure pip cache on largest drive
# =============================================================================
echo -e "\n${GREEN}=== Configuring pip cache ===${NC}"

# Determine pip cache location based on largest drive
if [ "$LARGEST_PATH" = "/" ]; then
    PIP_CACHE_DIR="/root/.cache/pip"
else
    PIP_CACHE_DIR="${LARGEST_PATH}/.cache/pip"
fi

echo "Pip cache directory: $PIP_CACHE_DIR"

# Create and set permissions
mkdir -p "$PIP_CACHE_DIR"
chmod -R 755 "$PIP_CACHE_DIR"

# Create symlinks from ~/.cache/pip for root and the invoking user
if [ "$LARGEST_PATH" != "/" ]; then
    mkdir -p /root/.cache
    rm -rf /root/.cache/pip
    ln -sf "$PIP_CACHE_DIR" /root/.cache/pip
    echo "Created symlink: /root/.cache/pip -> $PIP_CACHE_DIR"

    # Also configure for the sudo-invoking user if different from root
    SUDO_USER_HOME=$(getent passwd "${SUDO_USER:-}" | cut -d: -f6)
    if [ -n "$SUDO_USER_HOME" ] && [ "$SUDO_USER_HOME" != "/root" ]; then
        mkdir -p "$SUDO_USER_HOME/.cache"
        rm -rf "$SUDO_USER_HOME/.cache/pip"
        ln -sf "$PIP_CACHE_DIR" "$SUDO_USER_HOME/.cache/pip"
        chown -h "${SUDO_USER}:${SUDO_USER}" "$SUDO_USER_HOME/.cache/pip"
        echo "Created symlink: $SUDO_USER_HOME/.cache/pip -> $PIP_CACHE_DIR"
    fi
fi

echo -e "${GREEN}✓ Pip cache configured${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ System Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}Note: This script only moves Docker/containerd data (data-root), not the docker CLI.${NC}"
echo -e "${CYAN}The 'docker' command stays in place (e.g. /usr/bin/docker).${NC}"
echo ""
echo -e "${CYAN}If 'docker' fails after this: run 'sudo systemctl start containerd && sudo systemctl start docker', then 'docker info'.${NC}"
echo ""
echo -e "${CYAN}Next step: Run 02-cosmos-setup.sh as your regular user${NC}"
echo ""
