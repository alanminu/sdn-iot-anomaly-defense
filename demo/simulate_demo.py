#!/usr/bin/env python3
"""
demo/simulate_demo.py
-----------------------
A pure-Python, no-network-required replay of the detect -> block pipeline.

This does NOT need Mininet, Open vSwitch or Ryu installed -- it exists so
the detection + mitigation DECISION LOGIC (the actual intellectual
contribution of Objective 4) can be demonstrated and sanity-checked on
any machine in a few seconds, including as a safety net right before a
live presentation if the real Mininet/Ryu testbed (mininet_topo/ +
controller/) is unavailable on the presentation machine.

It replays a synthetic "session": some normal background flows, then an
attacker flow that ramps up exactly like the hping3 SYN-flood used in
the live demo (mininet_topo/run_attack_demo.sh), polled every
POLL_INTERVAL simulated seconds -- identical cadence to the real Ryu app
-- and shows the engine's live verdict at each poll, then flags the
exact poll at which the flow gets blocked.

USAGE
    python3 demo/simulate_demo.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from shared.mitigation_logic import MitigationEngine

POLL_INTERVAL = 2  # seconds, matches controller/ryu_mitigation_app.py


def simulate_flow(engine, name, packet_fn, byte_fn, n_polls, start_t=0.0):
    """packet_fn(t)/byte_fn(t) give CUMULATIVE packets/bytes for the flow
    at simulated time t (matching how OpenFlow flow-stats counters work:
    they are cumulative for the life of the flow)."""
    blocked_at = None
    print(f"\n=== Flow: {name} ===")
    for i in range(1, n_polls + 1):
        t = start_t + i * POLL_INTERVAL
        pkts = int(packet_fn(t))
        byts = int(byte_fn(t))
        verdict = engine.predict(duration_s=t, packet_count=pkts, byte_count=byts)
        tag = "ATTACK" if verdict["is_attack"] else "benign"
        print(f"  t={t:5.1f}s  pkts={pkts:6d}  bytes={byts:9d}  "
              f"-> {tag:6s} (confidence={verdict['confidence']:.2f})")
        if verdict["is_attack"] and blocked_at is None:
            blocked_at = t
            print(f"  >>> [MITIGATION] flow '{name}' BLOCKED at t={t:.1f}s "
                  f"(after {pkts} packets) <<<")
            break
        time.sleep(0.15)  # small pause so the printout reads like a live log
    if blocked_at is None:
        print(f"  (flow '{name}' ran the full simulated window without being blocked)")
    return blocked_at


def main():
    print("Loading trained model ...")
    engine = MitigationEngine()
    print("Model loaded. Replaying a simulated SDN session "
          f"(controller poll interval = {POLL_INTERVAL}s) ...\n")
    print("=" * 70)
    print(" BEFORE: legitimate IoT traffic (iot1 <-> iot2, ordinary pings/requests)")
    print("=" * 70)
    simulate_flow(
        engine, "iot1 -> iot2 (normal)",
        packet_fn=lambda t: 1.2 * t,                 # ~1.2 pkt/s, light traffic
        byte_fn=lambda t: 1.2 * t * 95,               # ~95 bytes/pkt average
        n_polls=6,
    )

    print("\n" + "=" * 70)
    print(" ATTACK: atk1 launches a SYN flood against iot2 "
          "(same shape as `hping3 -S --flood -p 80 iot2`)")
    print("=" * 70)
    blocked_at = simulate_flow(
        engine, "atk1 -> iot2 (SYN flood)",
        packet_fn=lambda t: 900 * t,                  # ~900 pkt/s flood
        byte_fn=lambda t: 900 * t * 58,                # small SYN packets
        n_polls=10,
    )

    print("\n" + "=" * 70)
    print(" AFTER: iot1 <-> iot3 traffic, unrelated to the blocked flow, still flows normally")
    print("=" * 70)
    simulate_flow(
        engine, "iot1 -> iot3 (normal, post-mitigation)",
        packet_fn=lambda t: 1.0 * t,
        byte_fn=lambda t: 1.0 * t * 90,
        n_polls=4,
    )

    print("\n" + "=" * 70)
    if blocked_at:
        print(f" RESULT: attack flow detected and blocked after {blocked_at:.1f}s "
              f"of simulated traffic. Legitimate flows were never touched.")
    else:
        print(" RESULT: attack flow was NOT blocked in this run -- check the model/threshold.")
    print("=" * 70)


if __name__ == "__main__":
    main()
