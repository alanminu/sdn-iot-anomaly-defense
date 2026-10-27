#!/usr/bin/env bash
# mininet_topo/run_attack_demo.sh
# --------------------------------
# Scripted before/after demo for the live presentation (Objective 3 + 4).
# Run this FROM INSIDE the Mininet CLI's xterm, or adapt into the `mininet>`
# prompt directly. It assumes the topology in topology.py is already up
# and ryu_mitigation_app.py is already running.
#
#   mininet> iot2 xterm &            # open a terminal on the victim first,
#                                     # inside it run: tcpdump -ni iot2-eth0
#   mininet> source mininet_topo/run_attack_demo.sh
#
# Or simply paste these commands one at a time into the `mininet>` prompt.

echo "[1/4] BEFORE: normal ping between IoT devices (should succeed)"
iot1 ping -c 4 iot2

echo "[2/4] Launching attacker traffic: atk1 SYN-floods iot2 on port 80"
echo "      (left running in the background for ~15s)"
atk1 hping3 -S --flood -p 80 iot2 &
ATTACK_PID=$!
sleep 15

echo "[3/4] AFTER ~15s: controller should have polled flow stats, scored"
echo "      the flood as an attack, and installed a drop rule. Checking..."
s1 ovs-ofctl dump-flows s1 -O OpenFlow13 | grep -i "priority=100" || \
  echo "      (no block rule yet -- wait a couple more seconds and re-check)"

kill $ATTACK_PID 2>/dev/null

echo "[4/4] Confirming legitimate traffic is unaffected: iot1 -> iot3"
iot1 ping -c 4 iot3
