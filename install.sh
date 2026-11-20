#!/bin/sh
# Installs the bitscryer and artificer commands for everyone on this machine (Linux):
#   sudo sh install.sh
# It makes a small Python of its own with numpy and PyTorch (/opt/thaumaturge/venv, or THAUMATURGE_VENV) and puts
# /usr/local/bin/bitscryer and /usr/local/bin/artificer, which run them from this folder, so a `git pull` here
# updates them. Run it again only if this script changes. Machine settings go in /etc/thaumaturge/ (see examples/).
#
# PyTorch is for Artificer. It is pinned to 2.10.0 from the CUDA 12.8 index: the last release whose CUDA 12.8 build
# still has kernels for Volta cards (V100, sm_70); from 2.11 that build starts at Turing (sm_75). On a machine with
# newer cards only, THAUMATURGE_TORCH="torch" THAUMATURGE_TORCH_INDEX=... picks another; THAUMATURGE_TORCH=none
# leaves PyTorch out (BitScryer doesn't need it).
set -eu
REPO=$(cd "$(dirname "$0")" && pwd)
VENV=${THAUMATURGE_VENV:-/opt/thaumaturge/venv}
TORCH=${THAUMATURGE_TORCH:-torch==2.10.0}
TORCH_INDEX=${THAUMATURGE_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade numpy
if [ "$TORCH" != "none" ]; then
    "$VENV/bin/python" -m pip install --quiet "$TORCH" --index-url "$TORCH_INDEX" \
        --extra-index-url https://pypi.org/simple
fi
install -d -m 755 /etc/thaumaturge
for tool in bitscryer artificer; do
    cat > /usr/local/bin/$tool <<WRAP
#!/bin/sh
# $tool (from $REPO); run it as yourself, never with sudo.
PYTHONPATH="$REPO\${PYTHONPATH:+:\$PYTHONPATH}" exec "$VENV/bin/python" -m thaumaturge.$tool "\$@"
WRAP
    chmod 755 /usr/local/bin/$tool
done
echo "Done: bitscryer and artificer (run them as yourself, not with sudo)"
