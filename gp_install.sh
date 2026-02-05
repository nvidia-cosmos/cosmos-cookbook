#!/bin/bash
# GlobalProtect installer/remover (DEB/RPM; x86_64, aarch64, armv7l)
# - DEB: exact names (globalprotect, globalprotect-ui)
# - RPM: any GlobalProtect* (incl. *_rpm variants), choose highest installed
# - Downloads to secure temp directory (mktemp); clean version parsing (stdout only)
# - SHA256 checksum verification for all packages
# - Robust version check: DEB via dpkg; RPM via sort -V max rule (no cross-tool confusion)
set -euo pipefail

YELLOW="\033[1;33m"; RED="\033[0;31m"; GREEN="\033[0;32m"; ENDCOLOR="\033[0m"
[[ "${USER:-}" == "root" ]] && { echo -e "${RED}Run without sudo.${ENDCOLOR}"; exit 1; }

# Global temp directory (cleaned up on exit)
SECURE_TEMP_DIR=""
trap '[[ -n "${SECURE_TEMP_DIR:-}" && -d "$SECURE_TEMP_DIR" ]] && rm -rf "$SECURE_TEMP_DIR"' EXIT

is_deb() { [[ -f /etc/debian_version ]]; }
is_rpm() { [[ -f /etc/redhat-release ]]; }

# SHA256 checksums for package verification (more secure than MD5)
declare -A PACKAGE_CHECKSUMS
PACKAGE_CHECKSUMS["GlobalProtect_deb-6.3.3.1-616.deb"]="992e076901fb26848a3e60b37312ee787deadce6e7a266c51dbf6b9ef1c1b6fc"
PACKAGE_CHECKSUMS["GlobalProtect_deb_aarch64-6.3.3.1-616.deb"]="7d91a9c7343c9cc1b370fc805de84a512938d9683a331455060be93caf6aa4a1"
PACKAGE_CHECKSUMS["GlobalProtect_deb_arm-6.3.3.1-616.deb"]="d48ffe800ddd3dc98ff679c586f0447e09b3e25e117e2a6868ed7f4a7e69ac78"
PACKAGE_CHECKSUMS["GlobalProtect_UI_deb-6.3.3.1-616.deb"]="07dc124bc2ed90b2b04b487b009ad996a465ba8124861549c2c107f0039a6585"
PACKAGE_CHECKSUMS["GlobalProtect_rpm-6.3.3.1-616.rpm"]="26fddae89ef40f356c24a7bf6019adf845e1072663edf0069eb9a3c59ed3c3de"
PACKAGE_CHECKSUMS["GlobalProtect_rpm_aarch64-6.3.3.1-616.rpm"]="9862bedc4ead6de0854bbdfaa2f592773d8b742ed1c136dac86f7b8d601ad25c"
PACKAGE_CHECKSUMS["GlobalProtect_rpm_arm-6.3.3.1-616.rpm"]="4c3a2abe9c24d074be2dcb88dd2bd6ee1f0ca2750896059dafffa585904c4471"
PACKAGE_CHECKSUMS["GlobalProtect_UI_rpm-6.3.3.1-616.rpm"]="3b31a4ec44a63d1ad835437a9b0f5f7af7b76631860e751ca804eea312adb1f8"


# Verify SHA256 checksum of downloaded package
verify_package_checksum() {
  local file_path="$1"
  local filename=$(basename "$file_path")
  local expected_checksum="${PACKAGE_CHECKSUMS[$filename]}"
  
  if [[ -z "$expected_checksum" ]]; then
    echo -e "${YELLOW}No checksum defined for: $filename (skipping verification)${ENDCOLOR}" >&2
    return 0
  fi
  
  echo -e "${YELLOW}Verifying SHA256 checksum for $filename...${ENDCOLOR}" >&2
  
  local calculated_checksum
  calculated_checksum=$(sha256sum "$file_path" | awk '{print $1}')
  
  if [[ "$calculated_checksum" == "$expected_checksum" ]]; then
    echo -e "${GREEN}SHA256 checksum verification successful${ENDCOLOR}" >&2
    return 0
  else
    echo -e "${RED}SHA256 checksum verification failed!${ENDCOLOR}" >&2
    echo -e "${RED}Expected: $expected_checksum${ENDCOLOR}" >&2
    echo -e "${RED}Calculated: $calculated_checksum${ENDCOLOR}" >&2
    return 1
  fi
}

install_dependencies() {
  if is_deb; then
    local need_update=0
    for pkg in curl jq; do
      dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed" || need_update=1
    done
    ((need_update)) && { echo -e "${YELLOW}Updating APT cache...${ENDCOLOR}"; sudo apt-get update -qq; }
    for pkg in curl jq; do
      dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed" || {
        echo -e "${YELLOW}Installing $pkg...${ENDCOLOR}"
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg"
      }
    done
  elif is_rpm; then
    for pkg in curl jq; do
      rpm -q "$pkg" >/dev/null 2>&1 || { echo -e "${YELLOW}Installing $pkg...${ENDCOLOR}"; sudo dnf -y install "$pkg"; }
    done
  fi
}

# -------- installed package detection --------
get_installed_name_and_version() {
  if is_deb; then
    dpkg-query -W -f='${Package} ${Version}\n' 'globalprotect*' 2>/dev/null \
      | awk '$1=="globalprotect" || $1=="globalprotect-ui" { print; exit }'
    return 0
  elif is_rpm; then
    mapfile -t lines < <(rpm -qa --qf '%{NAME} %{VERSION}-%{RELEASE}\n' 'GlobalProtect*' 2>/dev/null || true)
    (( ${#lines[@]} )) || return 1
    local best_line="${lines[0]}" best_ver="${lines[0]#* }"
    for line in "${lines[@]:1}"; do
      ver="${line#* }"
      local first
      first=$(printf '%s\n%s\n' "$best_ver" "$ver" | LC_ALL=C sort -V | head -n1)
      if [[ "$first" == "$best_ver" && "$best_ver" != "$ver" ]]; then
        best_ver="$ver"; best_line="$line"
      fi
    done
    echo "$best_line"
    return 0
  fi
}

get_installed_version() {
  local nv
  if nv=$(get_installed_name_and_version); then
    echo "${nv#* }"
  fi
}

download_pkg_and_get_version() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if ! curl -fsSL --retry 3 --retry-connrefused --retry-delay 2 -o "$dest" "$url"; then
    echo -e "${RED}Failed to download: $url${ENDCOLOR}" >&2
    return 1
  fi
  
  # Verify SHA256 checksum after download
  if ! verify_package_checksum "$dest"; then
    rm -f "$dest"
    echo -e "${RED}Downloaded file failed checksum verification. Removed file.${ENDCOLOR}" >&2
    return 1
  fi
  
  local version=""
  if is_rpm; then
    version=$(rpm -qp --queryformat '%{VERSION}-%{RELEASE}\n' "$dest" 2>/dev/null || true)
  elif is_deb; then
    version=$(dpkg-deb -f "$dest" Version 2>/dev/null || true)
  fi
  [[ -n "$version" ]] || { echo -e "${RED}Could not read package version from $dest${ENDCOLOR}" >&2; return 1; }
  printf '%s\n' "$version"
}

is_update_needed() {
  local i="${1//[[:space:]]/}" a="${2//[[:space:]]/}"
  [[ -z "$i" || -z "$a" ]] && return 0
  [[ "$i" == "$a" ]] && return 1
  if is_deb; then
    dpkg --compare-versions "$i" lt "$a"
    return $?
  elif is_rpm; then
    local max
    max=$(printf '%s\n%s\n' "$i" "$a" | LC_ALL=C sort -V | tail -n1)
    [[ "$max" == "$a" && "$i" != "$a" ]] && return 0 || return 1
  else
    local max
    max=$(printf '%s\n%s\n' "$i" "$a" | LC_ALL=C sort -V | tail -n1)
    [[ "$max" == "$a" && "$i" != "$a" ]] && return 0 || return 1
  fi
}

remove_existing_globalprotect() {
  echo -e "${YELLOW}Removing existing GlobalProtect installation...${ENDCOLOR}"
  if is_deb; then
    mapfile -t pkgs < <(dpkg-query -W -f='${Package}\n' 'globalprotect*' 2>/dev/null \
                        | awk '$1=="globalprotect" || $1=="globalprotect-ui" {print}')
    if ((${#pkgs[@]})); then
      sudo apt -y purge "${pkgs[@]}" || true
      sudo systemctl daemon-reload
    fi
  elif is_rpm; then
    mapfile -t pkgs < <(rpm -qa 'GlobalProtect*' 2>/dev/null || true)
    if ((${#pkgs[@]})); then
      sudo rpm -e "${pkgs[@]}" || true
      sudo systemctl daemon-reload
    fi
  fi
  echo -e "${GREEN}Removed.${ENDCOLOR}"
}

detect_portal() {
  local country
  country=$(curl -fsSL -k https://ipinfo.io/ | jq -r .country 2>/dev/null || echo "")
  case "$country" in
    CN) echo "gp-shz-cn.vpn.nvidia.cn" ;;
    HK) echo "gp-dc3-hk.vpn.nvidia.cn" ;;
    *)  echo "nvidia.gpcloudservice.com" ;;
  esac
}

# --- helper to patch pangps.xml ---
enable_default_browser() {
  local xml="/opt/paloaltonetworks/globalprotect/pangps.xml"
  sudo mkdir -p /opt/paloaltonetworks/globalprotect
  sudo touch "$xml"

  if ! sudo grep -q "<Settings>" "$xml"; then
    echo -e "<Settings>\n</Settings>" | sudo tee "$xml" >/dev/null
  fi

  # insert <default-browser>yes</default-browser> only if not already present
  if ! sudo grep -q "<default-browser>" "$xml"; then
    sudo sed -i '/<Settings>/a \  <default-browser>yes<\/default-browser>' "$xml"
    echo -e "${GREEN}Added <default-browser>yes</default-browser> to pangps.xml${ENDCOLOR}"
  else
    echo -e "${YELLOW}pangps.xml already contains default-browser setting${ENDCOLOR}"
  fi
}

install_globalprotect() {
  local ARCH PORTAL
  ARCH=$(uname -m)
  PORTAL=$(detect_portal)
  
  # Create temporary directory (honors TMPDIR if set)
  SECURE_TEMP_DIR=$(mktemp -d -t globalprotect.XXXXXXXXXX)
  if [[ ! -d "$SECURE_TEMP_DIR" ]]; then
    echo -e "${RED}Failed to create temporary directory${ENDCOLOR}"
    return 1
  fi
  
  # Set permissions to allow _apt user access (needed for apt install)
  chmod 755 "$SECURE_TEMP_DIR"
  
  if [[ "$ARCH" == "x86_64" ]]; then
    echo -e "Select the version to install:"
    echo -e "1) GlobalProtect UI  (x86_64)"
    echo -e "2) GlobalProtect CLI (x86_64)"
    read -r -p "Choose an option: " version_option
    local package_url="" pkg_file="" pkg_name=""
    case "$version_option" in
      1)
        if is_rpm; then
          package_url='https://d2hvyxt0t758wb.cloudfront.net/gp_install_files/GlobalProtect_UI_rpm-6.3.3.1-616.rpm'
          pkg_name="GlobalProtect_UI_rpm-6.3.3.1-616.rpm"
        elif is_deb; then
          package_url='https://d2hvyxt0t758wb.cloudfront.net/gp_install_files/GlobalProtect_UI_deb-6.3.3.1-616.deb'
          pkg_name="GlobalProtect_UI_deb-6.3.3.1-616.deb"
        fi ;;
      2)
        if is_rpm; then
          package_url='https://d2hvyxt0t758wb.cloudfront.net/gp_install_files/GlobalProtect_rpm-6.3.3.1-616.rpm'
          pkg_name="GlobalProtect_rpm-6.3.3.1-616.rpm"
        elif is_deb; then
          package_url='https://d2hvyxt0t758wb.cloudfront.net/gp_install_files/GlobalProtect_deb-6.3.3.1-616.deb'
          pkg_name="GlobalProtect_deb-6.3.3.1-616.deb"
        fi ;;
      *) echo -e "${RED}Invalid option.${ENDCOLOR}"; return 1 ;;
    esac
    
    pkg_file="$SECURE_TEMP_DIR/$pkg_name"

    if [[ -z "$package_url" || -z "$pkg_file" ]]; then
      echo -e "${RED}Unsupported package manager or architecture.${ENDCOLOR}"
      return 1
    fi

    local installed available
    installed="$(get_installed_version || true)"
    available="$(download_pkg_and_get_version "$package_url" "$pkg_file")" || return 1

    if [[ -n "${installed:-}" ]]; then
      if is_update_needed "$installed" "$available"; then
        echo -e "${YELLOW}Updating...${ENDCOLOR}"
        remove_existing_globalprotect
      else
        echo -e "${GREEN}Already up to date.${ENDCOLOR}"; return 0
      fi
    fi

    if is_rpm; then sudo dnf -y install "$pkg_file" || sudo dnf -y install "$package_url"
    elif is_deb; then sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg_file"; fi

    if [[ "$version_option" == "1" ]]; then
      echo -e "${GREEN}GlobalProtect UI installed.${ENDCOLOR}"
      echo -e "Portal: $PORTAL"
    else
      echo -e "${GREEN}GlobalProtect CLI installed.${ENDCOLOR}"
      echo -e "Connect: globalprotect connect --portal $PORTAL"
      enable_default_browser
    fi
  elif [[ "$ARCH" == "aarch64" ]]; then
    echo -e "Installing GlobalProtect CLI (ARM 64-bit)"
    local package_url pkg_file pkg_name
    if is_rpm; then
      package_url='https://d2hvyxt0t758wb.cloudfront.net/gp_install_files/GlobalProtect_rpm_aarch64-6.3.3.1-616.rpm'
      pkg_name="GlobalProtect_rpm_aarch64-6.3.3.1-616.rpm"
    elif is_deb; then
      package_url='https://d2hvyxt0t758wb.cloudfront.net/gp_install_files/GlobalProtect_deb_aarch64-6.3.3.1-616.deb'
      pkg_name="GlobalProtect_deb_aarch64-6.3.3.1-616.deb"
    fi
    
    pkg_file="$SECURE_TEMP_DIR/$pkg_name"
    
    if [[ -z "$package_url" || -z "$pkg_file" ]]; then
      echo -e "${RED}Unsupported package manager for aarch64.${ENDCOLOR}"
      return 1
    fi
    
    installed="$(get_installed_version || true)"
    available="$(download_pkg_and_get_version "$package_url" "$pkg_file")" || return 1
    if [[ -n "${installed:-}" ]]; then
      if is_update_needed "$installed" "$available"; then remove_existing_globalprotect; fi
    fi
    if is_rpm; then sudo dnf -y install "$pkg_file"; elif is_deb; then sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg_file"; fi
    echo -e "${GREEN}GlobalProtect CLI installed.${ENDCOLOR}"
    echo -e "Connect: globalprotect connect --portal $PORTAL"
    enable_default_browser
  elif [[ "$ARCH" == "armv7l" || "$ARCH" == "armv7" ]]; then
    echo -e "Installing GlobalProtect CLI (ARM 32-bit)"
    local package_url pkg_file pkg_name
    if is_rpm; then
      package_url='https://d2hvyxt0t758wb.cloudfront.net/gp_install_files/GlobalProtect_rpm_arm-6.3.3.1-616.rpm'
      pkg_name="GlobalProtect_rpm_arm-6.3.3.1-616.rpm"
    elif is_deb; then
      package_url='https://d2hvyxt0t758wb.cloudfront.net/gp_install_files/GlobalProtect_deb_arm-6.3.3.1-616.deb'
      pkg_name="GlobalProtect_deb_arm-6.3.3.1-616.deb"
    fi
    
    pkg_file="$SECURE_TEMP_DIR/$pkg_name"
    
    local installed available
    installed="$(get_installed_version || true)"
    available="$(download_pkg_and_get_version "$package_url" "$pkg_file")" || return 1
    if [[ -n "${installed:-}" ]]; then
      if is_update_needed "$installed" "$available"; then remove_existing_globalprotect; fi
    fi
    if is_rpm; then sudo dnf -y install "$pkg_file"; elif is_deb; then sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg_file"; fi
    echo -e "${GREEN}GlobalProtect CLI installed.${ENDCOLOR}"
    echo -e "Connect: globalprotect connect --portal $PORTAL"
    enable_default_browser
  else
    echo -e "${RED}Unsupported architecture: $(uname -m)${ENDCOLOR}"
    echo -e "${YELLOW}Supported: x86_64, aarch64 (ARM 64-bit), armv7l (ARM 32-bit)${ENDCOLOR}"
    exit 1
  fi

  sudo systemctl try-restart gpd.service || true
  systemctl --user try-restart gpa.service || true
}

remove_globalprotect() {
  local installed_info
  installed_info=$(get_installed_name_and_version 2>/dev/null || true)
  if [[ -z "$installed_info" ]]; then
    echo -e "${GREEN}No GlobalProtect installation found.${ENDCOLOR}"
    return 0
  fi
  echo -e "${YELLOW}Removing GlobalProtect...${ENDCOLOR}"
  remove_existing_globalprotect
}

# ---- Menu ----
while true; do
  echo -e "${GREEN}GlobalProtect Installer/Remover${ENDCOLOR}"
  echo -e "1) Install GlobalProtect"
  echo -e "2) Remove GlobalProtect"
  echo -e "3) Exit"
  echo
  read -r -p "Choose an option: " option
  echo
  case "$option" in
    1) install_dependencies; install_globalprotect ;;
    2) remove_globalprotect ;;
    3) exit 0 ;;
    *) echo -e "${RED}Invalid option. Please try again.${ENDCOLOR}" ;;
  esac
done