"""Network interface auto-detection for robot communication."""

from pathlib import Path

_SKIP_PREFIXES = ("lo", "wl", "docker", "br-", "veth", "virbr", "vnet", "tun", "tap")


def detect_robot_interface() -> str:
    """Return the name of the single wired NIC that is operationally UP."""
    net_root = Path("/sys/class/net")
    for iface in sorted(net_root.iterdir()):
        ifname = iface.name
        if any(ifname.startswith(p) for p in _SKIP_PREFIXES):
            continue
        try:
            state = (iface / "operstate").read_text().strip().lower()
        except OSError:
            continue
        if state != "up":
            continue
        print(f"[network] auto-detected interface: {ifname}")
        return ifname
    print("[network] no wired NIC found, falling back to loopback (lo)")
    return "lo"
