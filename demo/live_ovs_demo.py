#!/usr/bin/env python3
"""
demo/live_ovs_demo.py
-----------------------
A REAL, live, end-to-end version of Objective 4 that talks to Open
vSwitch directly instead of through Ryu.

Why this exists alongside controller/ryu_mitigation_app.py:
Ryu is the standard, taught way to build this (and ryu_mitigation_app.py
is the project's primary deliverable for that reason) but it needs a
Ryu install matched to your Python version. This script needs nothing
but `openvswitch-switch` (so it also works as a quick sanity check, or a
backup live demo, on a machine where Ryu isn't set up yet) and drives
the EXACT SAME trained model and decision logic from
shared/mitigation_logic.py -- it is not a separate, simplified pipeline.

How it gets per-flow stats without a controller:
Even with no controller installing explicit OpenFlow rules (the switch
just running its default NORMAL L2-learning action), Open vSwitch's own
datapath keeps a live cache of per-microflow packet/byte counters --
`ovs-appctl dpctl/dump-flows` exposes exactly that, keyed by source/dest
MAC (and IP/ports when the datapath's wildcarding keeps them). That is
what this script polls every POLL_INTERVAL seconds; each flow's duration
is tracked locally from when this script first saw it. The same
(duration, packet_count, byte_count) triple is then handed to
MitigationEngine exactly as controller/ryu_mitigation_app.py does with
an OFPFlowStatsReply.

Detected flows are blocked with a real, higher-priority OpenFlow rule
(`ovs-ofctl add-flow ... actions=drop`), which the datapath honours
immediately regardless of the NORMAL baseline action.

SETUP (run as root; needs Mininet + Open vSwitch, NOT Ryu):

    sudo python3 mininet_topo/topology.py --userspace --no-controller
    sudo python3 demo/live_ovs_demo.py s1

Inside the Mininet CLI (or a terminal attached to the same netns):

    mininet> atk1 hping3 -S --flood -p 80 iot2

and watch this script's output block the flow in real time.
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shared.mitigation_logic import MitigationEngine

POLL_INTERVAL = 2
OF_VERSION = "OpenFlow13"
BLOCK_PRIORITY = 100

FLOW_RE = re.compile(
    r"eth\(src=(?P<src>[0-9a-f:]+),dst=(?P<dst>[0-9a-f:]+)\).*?"
    r"packets:(?P<pkts>\d+), ?bytes:(?P<bytes>\d+)"
)


def dump_microflows():
    """Per-flow counters straight from the OVS datapath flow cache -- works
    with or without a controller, since it reflects what's actually been
    forwarded, not what any controller installed."""
    out = subprocess.run(
        ["ovs-appctl", "dpctl/dump-flows"], capture_output=True, text=True, check=True,
    ).stdout
    flows = []
    for line in out.splitlines():
        m = FLOW_RE.search(line)
        if not m:
            continue
        flows.append({
            "key": (m.group("src"), m.group("dst")),
            "packets": int(m.group("pkts")),
            "bytes": int(m.group("bytes")),
        })
    return flows


def block_flow(bridge, src_mac, dst_mac):
    rule = f"priority={BLOCK_PRIORITY},dl_src={src_mac},dl_dst={dst_mac},actions=drop"
    subprocess.run(["ovs-ofctl", "add-flow", bridge, "-O", OF_VERSION, rule], check=True)


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <ovs-bridge-name>  (e.g. s1)")
        sys.exit(1)
    bridge = sys.argv[1]

    print("Loading trained anomaly-detection model ...")
    engine = MitigationEngine()
    print(f"Polling live OVS datapath flow stats every {POLL_INTERVAL}s "
          f"(blocking via bridge '{bridge}'). Ctrl+C to stop.\n")

    first_seen = {}   # key -> timestamp first observed
    blocked = set()

    try:
        while True:
            now = time.time()
            for fl in dump_microflows():
                key = fl["key"]
                if fl["packets"] == 0 or key in blocked:
                    continue
                first_seen.setdefault(key, now)
                duration = max(now - first_seen[key], 0.5)

                verdict = engine.predict(duration, fl["packets"], fl["bytes"])
                tag = "ATTACK" if verdict["is_attack"] else "benign "
                print(f"[{time.strftime('%H:%M:%S')}] {key[0]} -> {key[1]:17s} "
                      f"dur={duration:6.1f}s pkts={fl['packets']:8d} bytes={fl['bytes']:10d} "
                      f"-> {tag} (confidence={verdict['confidence']:.2f})")

                if verdict["is_attack"]:
                    block_flow(bridge, *key)
                    blocked.add(key)
                    print(f"           >>> [MITIGATION] installed DROP rule {key[0]} -> {key[1]} <<<")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
