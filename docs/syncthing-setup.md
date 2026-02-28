# Syncthing: Phone Camera to WSL2 Inbox

Sync Pokemon card photos from your phone to `~/cardprice/data/inbox/` automatically.

## 1. Install Syncthing on WSL2

```bash
# Add the official repo (Debian/Ubuntu)
sudo mkdir -p /etc/apt/keyrings
sudo curl -L -o /etc/apt/keyrings/syncthing-archive-keyring.gpg \
  https://syncthing.net/release-key.gpg
echo "deb [signed-by=/etc/apt/keyrings/syncthing-archive-keyring.gpg] \
  https://apt.syncthing.net/ syncthing stable" \
  | sudo tee /etc/apt/sources.list.d/syncthing.list
sudo apt update && sudo apt install -y syncthing

# First run (generates config, then Ctrl-C)
syncthing --no-browser
```

## 2. Install on Phone

- **Android**: Install "Syncthing" from F-Droid or Google Play.
- **iOS**: Install "Mobius Sync" from the App Store (~$5, the only maintained iOS client).

## 3. Configure the Shared Folder

**On WSL2** (edit `~/.local/state/syncthing/config.xml` or use the web UI at `http://127.0.0.1:8384`):
- Add a folder with path `/home/godli/cardprice/data/inbox/`
- Set folder type to **Receive Only** (phone sends, WSL receives)
- Set file versioning to **None** (we process and move files ourselves)

**On Phone**:
- Open Syncthing/Mobius Sync, add the WSL2 device (exchange device IDs via QR or manual paste)
- Share your camera folder (e.g. `DCIM/CardPhotos`) with the WSL2 device
- Set folder type to **Send Only**

## 4. Auto-start Syncthing on WSL2 Boot

Create a systemd user service (WSL2 with systemd enabled):

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/syncthing.service << 'EOF'
[Unit]
Description=Syncthing File Synchronization
After=network.target

[Service]
ExecStart=/usr/bin/syncthing serve --no-browser --no-restart
Restart=on-failure

[Install]
WantedBy=default.target
EOF

systemctl --user enable syncthing.service
systemctl --user start syncthing.service
```

If systemd is not enabled in your WSL2, add to `~/.bashrc` instead:

```bash
pgrep -x syncthing > /dev/null || syncthing serve --no-browser &>/dev/null &
```

## 5. WSL2 Networking (Non-Mirrored Mode)

This is the hard part. WSL2 NAT mode means your phone cannot reach WSL2 directly.

**Option A -- Relay (easiest, no config needed)**:
Syncthing's global relay servers broker the connection automatically. Works out of the
box but is slower (~1-5 MB/s). Fine for a few card photos at a time.

**Option B -- Port forward from Windows to WSL2**:
```powershell
# Run in PowerShell as Admin. Get WSL IP first:
#   wsl hostname -I
netsh interface portproxy add v4tov4 listenport=22000 listenaddress=0.0.0.0 `
  connectport=22000 connectaddress=<WSL_IP>
netsh interface portproxy add v4tov4 listenport=21027 listenaddress=0.0.0.0 `
  connectport=21027 connectaddress=<WSL_IP>

# Open Windows Firewall
New-NetFirewallRule -DisplayName "Syncthing TCP" -Direction Inbound `
  -LocalPort 22000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Syncthing UDP" -Direction Inbound `
  -LocalPort 21027,22000 -Protocol UDP -Action Allow
```

Note: WSL2's IP changes on reboot. Script the portproxy update or switch to
mirrored mode (`networkingMode=mirrored` in `%USERPROFILE%/.wslconfig`).

**Option C -- Mirrored networking (best)**:
```ini
# %USERPROFILE%/.wslconfig
[wsl2]
networkingMode=mirrored
```
Then `wsl --shutdown` and reopen. WSL2 shares the host IP; phone connects directly.
This project uses non-mirrored, but mirrored is simpler for Syncthing.

## Ports Reference

| Port  | Protocol | Purpose              |
|-------|----------|----------------------|
| 22000 | TCP/UDP  | Sync protocol (BEP)  |
| 21027 | UDP      | Local discovery       |
| 8384  | TCP      | Web UI (local only)   |
