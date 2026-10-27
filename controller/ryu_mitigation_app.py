#!/usr/bin/env python3
"""
ryu_mitigation_app.py
----------------------
Objective 4 from the project proposal:
    "Write a Ryu application that polls flow statistics, feeds them to the
     trained model, and installs a flow rule to drop or rate-limit traffic
     from a source flagged as malicious. Demonstrate the before/after
     effect (attack traffic passing, then being blocked) as the project's
     centrepiece demo."

This is a standard OpenFlow 1.3 Ryu app written against the modern,
widely-documented Ryu API (ofp_event handlers, ryu.lib.packet with
human-readable MAC/IP strings, OFPActionOutput, etc.) -- the same style
used in Ryu's own simple_switch_13.py tutorial app -- plus:

  * a periodic (every POLL_INTERVAL seconds) OFPFlowStatsRequest to every
    connected switch
  * on every OFPFlowStatsReply, each flow's (duration, packet_count,
    byte_count) is handed to shared.mitigation_logic.MitigationEngine
  * a flow flagged as an attack gets a high-priority DROP rule installed
    for its exact 5-tuple match, so it is cut off at the switch with no
    further controller involvement (closing the detect -> block loop)

Run (on a machine with Ryu + Mininet installed, Python matching your
Ryu install -- see README.md for the exact two commands):

    ryu-manager controller/ryu_mitigation_app.py --verbose

Then point Mininet's switch at it (mininet_topo/topology.py does this
automatically): the switch connects to 127.0.0.1:6633.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ether_types

from shared.mitigation_logic import MitigationEngine

POLL_INTERVAL = 2          # seconds between flow-stats polls (Objective 4)
BLOCK_RULE_PRIORITY = 100  # higher than the learning-switch's normal rules
BLOCK_RULE_IDLE_TIMEOUT = 30  # seconds a block rule stays installed with no matching traffic
NORMAL_RULE_PRIORITY = 1


class SDNAnomalyMitigationApp(app_manager.RyuApp):
    """
    IoT SD-Network anomaly detection + mitigation controller.

    - Runs a standard MAC-learning L2 switch (so the topology forwards
      normal traffic exactly like a regular switch would).
    - In parallel, polls per-flow statistics on a timer, scores every flow
      with the trained ML model, and installs a DROP rule for any flow the
      model flags as malicious -- the "detect, then block" loop that is
      this project's contribution over plain detection-only baselines
      (see literature review items 2, 4, 5, 6 in the report).
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}          # {dpid: {mac: port}}
        self.datapaths = {}            # {dpid: datapath}
        self.blocked_flows = set()     # {(dpid, match_tuple)} already dropped
        self.flow_first_seen = {}      # {(dpid, match_tuple): timestamp}

        self.logger.info("Loading trained anomaly-detection model ...")
        self.engine = MitigationEngine()
        self.logger.info("Model loaded. Polling flow stats every %ss.", POLL_INTERVAL)

        self.monitor_thread = hub.spawn(self._monitor)

    # ------------------------------------------------------------------
    # Standard OpenFlow handshake + L2 learning-switch forwarding
    # ------------------------------------------------------------------

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.datapaths[datapath.id] = datapath
        self.mac_to_port.setdefault(datapath.id, {})

        # table-miss flow entry: unmatched packets go to the controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info("Switch connected: dpid=%s", datapath.id)

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                 instructions=inst, idle_timeout=idle_timeout,
                                 hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    def drop_flow(self, datapath, match, priority=BLOCK_RULE_PRIORITY):
        """Install a flow rule with an empty action list -> drop (Objective 4)."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, [])]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                 instructions=inst, idle_timeout=BLOCK_RULE_IDLE_TIMEOUT)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst, src = eth.dst, eth.src
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, NORMAL_RULE_PRIORITY, match, actions, idle_timeout=20)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                   in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    # ------------------------------------------------------------------
    # Flow-stats polling -> ML scoring -> mitigation (Objective 4)
    # ------------------------------------------------------------------

    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_flow_stats(dp)
            hub.sleep(POLL_INTERVAL)

    def _request_flow_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        datapath = ev.msg.datapath
        dpid = datapath.id

        for stat in ev.msg.body:
            # Only score the learned-forwarding flows, not our own
            # table-miss / drop rules.
            if stat.priority not in (NORMAL_RULE_PRIORITY,):
                continue
            if stat.packet_count == 0:
                continue

            match_key = (dpid, tuple(sorted(stat.match.items())))
            duration_s = stat.duration_sec + stat.duration_nsec / 1e9

            verdict = self.engine.predict(duration_s, stat.packet_count, stat.byte_count)

            if verdict["is_attack"] and match_key not in self.blocked_flows:
                self.blocked_flows.add(match_key)
                self.drop_flow(datapath, stat.match)
                self.logger.warning(
                    "[MITIGATION] dpid=%s match=%s pkts=%s bytes=%s dur=%.1fs "
                    "-> ATTACK (confidence=%.2f) -> flow BLOCKED",
                    dpid, dict(stat.match.items()), stat.packet_count,
                    stat.byte_count, duration_s, verdict["confidence"],
                )
            elif not verdict["is_attack"] and verdict["reason"] == "model":
                self.logger.debug(
                    "dpid=%s match=%s pkts=%s bytes=%s dur=%.1fs -> benign (confidence=%.2f)",
                    dpid, dict(stat.match.items()), stat.packet_count,
                    stat.byte_count, duration_s, verdict["confidence"],
                )
