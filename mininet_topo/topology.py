#!/usr/bin/env python3
"""
mininet_topo/topology.py
-------------------------
Objective 3 from the project proposal:
    "Set up a small Mininet topology (4-6 hosts) with a Ryu controller."

Topology (one switch, five IoT-style hosts -- enough to show one
attacker, one victim and legitimate background traffic, Objective 3):

    iot1 --\
    iot2 ---\
    iot3 ----+--- s1 ---- (remote controller: ryu_mitigation_app.py)
    iot4 ---/
    atk1  --/

    atk1  = the "compromised IoT device" that launches the flood
    iot1..iot4 = normal devices exchanging ordinary traffic
    s1    = OVS switch in OpenFlow1.3 mode, controlled remotely by Ryu

RUN (needs root, and Ryu already running separately -- see README.md):

    sudo python3 mininet_topo/topology.py

Once the Mininet CLI appears you can, for example:

    mininet> iot1 ping -c 20 iot2                      # normal traffic
    mininet> atk1 hping3 -S --flood -p 80 iot2          # SYN-flood attack
    mininet> dpctl dump-flows -O OpenFlow13             # watch the block rule land

NO RYU YET? Pass --no-controller to run the switch standalone (plain
L2 learning, no OpenFlow app at all) and drive mitigation with
demo/live_ovs_demo.py instead -- see README.md "No Ryu set up yet".
"""
import sys

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel, info


def build_topology(controller_ip="127.0.0.1", controller_port=6633,
                    datapath="kernel", use_controller=True):
    net = Mininet(switch=OVSSwitch, link=TCLink, controller=None, autoSetMacs=True)

    c0 = None
    if use_controller:
        info("*** Adding remote Ryu controller\n")
        c0 = net.addController(
            "c0", controller=RemoteController, ip=controller_ip, port=controller_port
        )
    else:
        info("*** No controller requested -- switch will run standalone "
             "(use demo/live_ovs_demo.py for mitigation)\n")

    info("*** Adding switch\n")
    fail_mode = "standalone" if not use_controller else "secure"
    s1 = net.addSwitch("s1", protocols="OpenFlow13", datapath=datapath, failMode=fail_mode)

    info("*** Adding hosts (4 normal IoT devices + 1 attacker)\n")
    iot1 = net.addHost("iot1", ip="10.0.0.1/24")
    iot2 = net.addHost("iot2", ip="10.0.0.2/24")
    iot3 = net.addHost("iot3", ip="10.0.0.3/24")
    iot4 = net.addHost("iot4", ip="10.0.0.4/24")
    atk1 = net.addHost("atk1", ip="10.0.0.9/24")

    info("*** Creating links\n")
    for h in (iot1, iot2, iot3, iot4, atk1):
        net.addLink(h, s1, bw=10)  # 10 Mbps links, typical for constrained IoT devices

    info("*** Starting network\n")
    net.build()
    if c0:
        c0.start()
        s1.start([c0])
    else:
        s1.start([])

    return net


if __name__ == "__main__":
    setLogLevel("info")
    # "datapath=user" lets this run on hosts/containers with no Open
    # vSwitch kernel module (e.g. this project's own sandboxed dev/test
    # environment); use the default "kernel" datapath on a normal Linux
    # box with the openvswitch kernel module loaded for best performance.
    dp = "user" if "--userspace" in sys.argv else "kernel"
    use_ctrl = "--no-controller" not in sys.argv
    net = build_topology(datapath=dp, use_controller=use_ctrl)
    CLI(net)
    net.stop()
