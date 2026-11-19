#!/bin/sh
# Installs the bitscryer command for everyone on this machine (Linux):
#   sudo sh install.sh
# It makes a small Python of its own with numpy (/opt/thaumaturge/venv, or THAUMATURGE_VENV) and puts
# /usr/local/bin/bitscryer, which runs BitScryer from this folder, so a `git pull` here updates it. Run it again
# only if this script changes. Machine settings go in /etc/thaumaturge/bitscryer.json (see examples/).
set -eu
REPO=$(cd "$(dirname "$0")" && pwd)
VENV=${THAUMATURGE_VENV:-/opt/thaumaturge/venv}
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade numpy
install -d -m 755 /etc/thaumaturge
cat > /usr/local/bin/bitscryer <<WRAP
#!/bin/sh
# BitScryer (from $REPO); run it as yourself, never with sudo.
PYTHONPATH="$REPO\${PYTHONPATH:+:\$PYTHONPATH}" exec "$VENV/bin/python" -m thaumaturge.bitscryer "\$@"
WRAP
chmod 755 /usr/local/bin/bitscryer
echo "Done: bitscryer (run it as yourself, not with sudo)"
