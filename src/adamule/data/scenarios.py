"""Synthetic transaction generation scenarios for payment/mule graphs.

Implements Scenarios A through I:
- Scenario A: Normal individual P2P transactions
- Scenario B: Legitimate merchant aggregation (Hard Negative)
- Scenario C: Normal irregular merchant (Statistically unusual bursts)
- Scenario D: Basic mule ring (Victim -> A -> B -> C -> Cashout)
- Scenario E: Fan-in mule (Many victims -> Mule)
- Scenario F: Fan-out mule (Mule -> Many intermediaries)
- Scenario G: Layered mule network (Multi-hop tree)
- Scenario H: Camouflaged mule network (Fraudulent topology mimicking merchant aggregation)
- Scenario I: Adaptive structuring (Dynamically structured evasive flows)
"""

import math
import random
from typing import Any, Dict, List, Tuple
import numpy as np


class ScenarioGenerator:
    """Generates synthetic transaction batches according to calibrated scenarios."""

    def __init__(self, entities: Dict[str, Any], time_span_days: int = 30, seed: int = 42):
        self.accounts = entities["accounts"]
        self.merchants = entities["merchants"]
        self.devices = entities["devices"]
        self.ips = entities["ips"]
        
        self.account_ids = [a["account_id"] for a in self.accounts]
        self.merchant_ids = [m["merchant_id"] for m in self.merchants]
        self.device_ids = [d["device_id"] for d in self.devices]
        self.ip_ids = [i["ip_id"] for i in self.ips]
        
        self.start_timestamp = 1704067200.0  # e.g. 2024-01-01 00:00:00 UTC
        self.max_time_seconds = time_span_days * 86400.0
        self.rng = np.random.default_rng(seed)
        self.tx_counter = 0

    def _next_tx_id(self) -> str:
        self.tx_counter += 1
        return f"TX_{self.tx_counter:08d}"

    def _sample_device_and_ip(self, account_id: str, is_risky: bool = False) -> Tuple[str, str]:
        if is_risky:
            # Pick from devices/IPs with higher risk indicators
            d = self.rng.choice(self.device_ids[: max(5, len(self.device_ids) // 10)])
            ip = self.rng.choice(self.ip_ids[: max(5, len(self.ip_ids) // 10)])
        else:
            d = self.rng.choice(self.device_ids)
            ip = self.rng.choice(self.ip_ids)
        return str(d), str(ip)

    def generate_scenario_a(self, count: int) -> List[Dict[str, Any]]:
        """Scenario A: Normal individual P2P transactions.
        
        Realistic log-normal amounts, distributed throughout normal active hours.
        """
        txs = []
        for _ in range(count):
            sender = self.rng.choice(self.account_ids)
            receiver = self.rng.choice(self.account_ids)
            while receiver == sender:
                receiver = self.rng.choice(self.account_ids)
                
            # Log-normal distribution centered around 500-1500
            amount = float(np.clip(self.rng.lognormal(mean=6.5, sigma=0.8), 20.0, 20000.0))
            # Timestamps spread over the time span with daylight weighting
            day_fraction = self.rng.beta(a=3.0, b=2.0)  # biases towards daytime/evening
            day = self.rng.integers(0, int(self.max_time_seconds // 86400))
            ts = self.start_timestamp + day * 86400.0 + day_fraction * 86400.0
            
            dev, ip = self._sample_device_and_ip(sender, is_risky=False)
            txs.append({
                "transaction_id": self._next_tx_id(),
                "timestamp": ts,
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": round(amount, 2),
                "transaction_type": "p2p",
                "merchant_id": "None",
                "device_id": dev,
                "ip_id": ip,
                "location": "domestic",
                "channel": self.rng.choice(["upi_app", "web", "qr_code"]),
                "is_fraud": 0,
                "fraud_type": "none",
                "scenario_id": "scenario_a"
            })
        return txs

    def generate_scenario_b(self, count: int) -> List[Dict[str, Any]]:
        """Scenario B: Legitimate merchant aggregation (Hard Negative).
        
        Customers pay merchant (fan-in), merchant makes settlement/supplier payouts (fan-out).
        High degree and high volume, but legitimate business profiles.
        """
        txs = []
        merchants_sample = self.rng.choice(self.merchants, size=min(15, len(self.merchants)), replace=False)
        txs_per_merchant = max(1, count // len(merchants_sample))
        
        for m in merchants_sample:
            m_account = m["merchant_account_id"]
            m_id = m["merchant_id"]
            
            # 85% consumer purchases (customer -> merchant)
            num_in = int(txs_per_merchant * 0.85)
            # 15% settlement/supplier payouts (merchant -> supplier account)
            num_out = txs_per_merchant - num_in
            
            # Incoming transactions
            for _ in range(num_in):
                cust = self.rng.choice(self.account_ids)
                while cust == m_account:
                    cust = self.rng.choice(self.account_ids)
                amount = float(np.clip(self.rng.lognormal(mean=5.0, sigma=0.6), 15.0, 5000.0))
                ts = self.start_timestamp + self.rng.uniform(0, self.max_time_seconds)
                dev, ip = self._sample_device_and_ip(cust, is_risky=False)
                txs.append({
                    "transaction_id": self._next_tx_id(),
                    "timestamp": ts,
                    "sender_id": cust,
                    "receiver_id": m_account,
                    "amount": round(amount, 2),
                    "transaction_type": "p2m",
                    "merchant_id": m_id,
                    "device_id": dev,
                    "ip_id": ip,
                    "location": "domestic",
                    "channel": "qr_code",
                    "is_fraud": 0,
                    "fraud_type": "none",
                    "scenario_id": "scenario_b"
                })
                
            # Outgoing supplier/settlement payouts
            for _ in range(num_out):
                supplier = self.rng.choice(self.account_ids)
                while supplier == m_account:
                    supplier = self.rng.choice(self.account_ids)
                amount = float(np.clip(self.rng.lognormal(mean=8.0, sigma=0.7), 2000.0, 50000.0))
                ts = self.start_timestamp + self.rng.uniform(0, self.max_time_seconds)
                dev, ip = self._sample_device_and_ip(m_account, is_risky=False)
                txs.append({
                    "transaction_id": self._next_tx_id(),
                    "timestamp": ts,
                    "sender_id": m_account,
                    "receiver_id": supplier,
                    "amount": round(amount, 2),
                    "transaction_type": "merchant_settlement",
                    "merchant_id": m_id,
                    "device_id": dev,
                    "ip_id": ip,
                    "location": "domestic",
                    "channel": "web",
                    "is_fraud": 0,
                    "fraud_type": "none",
                    "scenario_id": "scenario_b"
                })
        return txs

    def generate_scenario_c(self, count: int) -> List[Dict[str, Any]]:
        """Scenario C: Normal irregular merchant (Statistically unusual bursts).
        
        Legitimate small merchant experiencing festive, flash-sale, or wholesale bursts.
        Unusual burstiness and high variance, but NOT fraudulent.
        """
        txs = []
        bursty_merchants = self.rng.choice(self.merchants, size=min(5, len(self.merchants)), replace=False)
        txs_per_m = max(1, count // len(bursty_merchants))
        
        for m in bursty_merchants:
            m_account = m["merchant_account_id"]
            m_id = m["merchant_id"]
            # Burst clustered within a 48-hour window
            burst_start = self.start_timestamp + self.rng.uniform(0, max(1, self.max_time_seconds - 172800))
            
            for _ in range(txs_per_m):
                cust = self.rng.choice(self.account_ids)
                while cust == m_account:
                    cust = self.rng.choice(self.account_ids)
                amount = float(np.clip(self.rng.lognormal(mean=7.2, sigma=1.0), 50.0, 40000.0))
                ts = burst_start + self.rng.exponential(scale=3600.0)
                dev, ip = self._sample_device_and_ip(cust, is_risky=False)
                txs.append({
                    "transaction_id": self._next_tx_id(),
                    "timestamp": ts,
                    "sender_id": cust,
                    "receiver_id": m_account,
                    "amount": round(amount, 2),
                    "transaction_type": "p2m",
                    "merchant_id": m_id,
                    "device_id": dev,
                    "ip_id": ip,
                    "location": "domestic",
                    "channel": "pos",
                    "is_fraud": 0,
                    "fraud_type": "none",
                    "scenario_id": "scenario_c"
                })
        return txs

    def generate_scenario_d(self, count: int) -> List[Dict[str, Any]]:
        """Scenario D: Basic mule ring.
        
        Victim -> Mule A -> Mule B -> Mule C -> Cashout node.
        Linear path with rapid fund forwarding (high velocity, short dwell time).
        """
        txs = []
        rings = max(1, count // 4)
        for _ in range(rings):
            # Select 5 accounts: 1 victim, 3 mules, 1 cashout
            chain = self.rng.choice(self.account_ids, size=5, replace=False)
            victim, mule_a, mule_b, mule_c, cashout = chain
            
            base_time = self.start_timestamp + self.rng.uniform(0, max(1, self.max_time_seconds - 36000))
            total_amount = float(np.clip(self.rng.uniform(20000, 80000), 10000, 95000))
            
            hops = [
                (victim, mule_a, total_amount),
                (mule_a, mule_b, total_amount * 0.96),  # cut taken
                (mule_b, mule_c, total_amount * 0.92),
                (mule_c, cashout, total_amount * 0.88),
            ]
            current_time = base_time
            shared_device, shared_ip = self._sample_device_and_ip(mule_a, is_risky=True)
            
            for s, r, amt in hops:
                current_time += self.rng.uniform(300, 2400)  # 5 to 40 mins later
                dev = shared_device if s != victim else self.rng.choice(self.device_ids)
                ip = shared_ip if s != victim else self.rng.choice(self.ip_ids)
                txs.append({
                    "transaction_id": self._next_tx_id(),
                    "timestamp": current_time,
                    "sender_id": s,
                    "receiver_id": r,
                    "amount": round(amt, 2),
                    "transaction_type": "intermediary_transfer" if r != cashout else "atm_cashout",
                    "merchant_id": "None",
                    "device_id": dev,
                    "ip_id": ip,
                    "location": "domestic",
                    "channel": "upi_app",
                    "is_fraud": 1,
                    "fraud_type": "basic_mule",
                    "scenario_id": "scenario_d"
                })
        return txs

    def generate_scenario_e(self, count: int) -> List[Dict[str, Any]]:
        """Scenario E: Fan-in mule.
        
        Multiple victim accounts deposit into a single mule aggregator within a short time window.
        """
        txs = []
        aggregators = max(1, count // 10)
        for _ in range(aggregators):
            mule = self.rng.choice(self.account_ids)
            num_victims = min(10, count - len(txs))
            if num_victims <= 0:
                break
            victims = [v for v in self.rng.choice(self.account_ids, size=num_victims * 2, replace=False) if v != mule][:num_victims]
            base_time = self.start_timestamp + self.rng.uniform(0, max(1, self.max_time_seconds - 7200))
            
            for v in victims:
                amount = float(self.rng.uniform(3000, 15000))
                ts = base_time + self.rng.uniform(60, 3600)
                dev, ip = self._sample_device_and_ip(v, is_risky=False)
                txs.append({
                    "transaction_id": self._next_tx_id(),
                    "timestamp": ts,
                    "sender_id": v,
                    "receiver_id": mule,
                    "amount": round(amount, 2),
                    "transaction_type": "p2p",
                    "merchant_id": "None",
                    "device_id": dev,
                    "ip_id": ip,
                    "location": "domestic",
                    "channel": "upi_app",
                    "is_fraud": 1,
                    "fraud_type": "fan_in_mule",
                    "scenario_id": "scenario_e"
                })
        return txs

    def generate_scenario_f(self, count: int) -> List[Dict[str, Any]]:
        """Scenario F: Fan-out mule.
        
        One compromised/source account rapidly distributes funds to multiple intermediaries.
        """
        txs = []
        distributors = max(1, count // 10)
        for _ in range(distributors):
            source = self.rng.choice(self.account_ids)
            num_recipients = min(10, count - len(txs))
            if num_recipients <= 0:
                break
            recipients = [r for r in self.rng.choice(self.account_ids, size=num_recipients * 2, replace=False) if r != source][:num_recipients]
            base_time = self.start_timestamp + self.rng.uniform(0, max(1, self.max_time_seconds - 7200))
            shared_dev, shared_ip = self._sample_device_and_ip(source, is_risky=True)
            
            for rec in recipients:
                amount = float(self.rng.uniform(2000, 9500))
                ts = base_time + self.rng.uniform(30, 1800)
                txs.append({
                    "transaction_id": self._next_tx_id(),
                    "timestamp": ts,
                    "sender_id": source,
                    "receiver_id": rec,
                    "amount": round(amount, 2),
                    "transaction_type": "intermediary_transfer",
                    "merchant_id": "None",
                    "device_id": shared_dev,
                    "ip_id": shared_ip,
                    "location": "domestic",
                    "channel": "upi_app",
                    "is_fraud": 1,
                    "fraud_type": "fan_out_mule",
                    "scenario_id": "scenario_f"
                })
        return txs

    def generate_scenario_g(self, count: int) -> List[Dict[str, Any]]:
        """Scenario G: Layered mule network.
        
        Multi-hop tree structure (victims -> layer 1 mules -> layer 2 mules -> cashout).
        """
        txs = []
        networks = max(1, count // 8)
        for _ in range(networks):
            nodes = self.rng.choice(self.account_ids, size=7, replace=False)
            v1, v2 = nodes[0], nodes[1]
            l1_a, l1_b = nodes[2], nodes[3]
            l2_a = nodes[4]
            cash_a, cash_b = nodes[5], nodes[6]
            
            t0 = self.start_timestamp + self.rng.uniform(0, max(1, self.max_time_seconds - 86400))
            dev_layer, ip_layer = self._sample_device_and_ip(l1_a, is_risky=True)
            
            edges = [
                (v1, l1_a, 25000.0, t0, False),
                (v2, l1_b, 30000.0, t0 + 300, False),
                (l1_a, l2_a, 23500.0, t0 + 1200, True),
                (l1_b, l2_a, 28000.0, t0 + 1500, True),
                (l2_a, cash_a, 25000.0, t0 + 3600, True),
                (l2_a, cash_b, 24000.0, t0 + 4000, True),
            ]
            for s, r, amt, ts, is_mule_dev in edges:
                dev = dev_layer if is_mule_dev else self.rng.choice(self.device_ids)
                ip = ip_layer if is_mule_dev else self.rng.choice(self.ip_ids)
                txs.append({
                    "transaction_id": self._next_tx_id(),
                    "timestamp": ts,
                    "sender_id": s,
                    "receiver_id": r,
                    "amount": round(amt, 2),
                    "transaction_type": "intermediary_transfer",
                    "merchant_id": "None",
                    "device_id": dev,
                    "ip_id": ip,
                    "location": "domestic",
                    "channel": "upi_app",
                    "is_fraud": 1,
                    "fraud_type": "layered_mule",
                    "scenario_id": "scenario_g"
                })
        return txs

    def generate_scenario_h(self, count: int) -> List[Dict[str, Any]]:
        """Scenario H: Camouflaged mule network.
        
        Fraud ring deliberately mimics legitimate merchant aggregation.
        Intersperses small legitimate purchases and routes through pseudo-merchants to confuse GNNs.
        """
        txs = []
        rings = max(1, count // 8)
        for _ in range(rings):
            actors = self.rng.choice(self.account_ids, size=4, replace=False)
            v, mule, cash = actors[0], actors[1], actors[2]
            target_merchant = self.rng.choice(self.merchants)
            m_account = target_merchant["merchant_account_id"]
            m_id = target_merchant["merchant_id"]
            
            t0 = self.start_timestamp + self.rng.uniform(0, max(1, self.max_time_seconds - 50000))
            # 1. Victim to mule
            txs.append({
                "transaction_id": self._next_tx_id(),
                "timestamp": t0,
                "sender_id": v,
                "receiver_id": mule,
                "amount": 40000.0,
                "transaction_type": "p2p",
                "merchant_id": "None",
                "device_id": self.rng.choice(self.device_ids),
                "ip_id": self.rng.choice(self.ip_ids),
                "location": "domestic",
                "channel": "upi_app",
                "is_fraud": 1,
                "fraud_type": "camouflage_mule",
                "scenario_id": "scenario_h"
            })
            
            # 2. Mule conducts 2-3 genuine-looking micro-purchases at merchant
            for i in range(2):
                txs.append({
                    "transaction_id": self._next_tx_id(),
                    "timestamp": t0 + (i + 1) * 300,
                    "sender_id": mule,
                    "receiver_id": m_account,
                    "amount": round(float(self.rng.uniform(100, 450)), 2),
                    "transaction_type": "p2m",
                    "merchant_id": m_id,
                    "device_id": self.rng.choice(self.device_ids),
                    "ip_id": self.rng.choice(self.ip_ids),
                    "location": "domestic",
                    "channel": "qr_code",
                    "is_fraud": 1,  # Camouflage transaction part of fraud operation
                    "fraud_type": "camouflage_mule",
                    "scenario_id": "scenario_h"
                })
                
            # 3. Mule transfers rest to cashout
            txs.append({
                "transaction_id": self._next_tx_id(),
                "timestamp": t0 + 1800,
                "sender_id": mule,
                "receiver_id": cash,
                "amount": 38500.0,
                "transaction_type": "intermediary_transfer",
                "merchant_id": "None",
                "device_id": self.rng.choice(self.device_ids),
                "ip_id": self.rng.choice(self.ip_ids),
                "location": "domestic",
                "channel": "upi_app",
                "is_fraud": 1,
                "fraud_type": "camouflage_mule",
                "scenario_id": "scenario_h"
            })
        return txs

    def generate_scenario_i(self, count: int) -> List[Dict[str, Any]]:
        """Scenario I: Adaptive structuring.
        
        Amounts are structured (smurfed) below typical detection thresholds (e.g. splitting into 4-6 transfers).
        """
        txs = []
        instances = max(1, count // 6)
        for _ in range(instances):
            src = self.rng.choice(self.account_ids)
            dest = self.rng.choice(self.account_ids)
            while dest == src:
                dest = self.rng.choice(self.account_ids)
                
            total_amount = float(self.rng.uniform(35000, 90000))
            num_splits = self.rng.integers(3, 7)
            split_amounts = self.rng.dirichlet(np.ones(num_splits)) * total_amount
            base_t = self.start_timestamp + self.rng.uniform(0, max(1, self.max_time_seconds - 20000))
            
            for i, amt in enumerate(split_amounts):
                txs.append({
                    "transaction_id": self._next_tx_id(),
                    "timestamp": base_t + i * self.rng.uniform(400, 1800),
                    "sender_id": src,
                    "receiver_id": dest,
                    "amount": round(float(amt), 2),
                    "transaction_type": "intermediary_transfer",
                    "merchant_id": "None",
                    "device_id": self.rng.choice(self.device_ids),
                    "ip_id": self.rng.choice(self.ip_ids),
                    "location": "domestic",
                    "channel": "upi_app",
                    "is_fraud": 1,
                    "fraud_type": "adaptive_structuring",
                    "scenario_id": "scenario_i"
                })
        return txs
