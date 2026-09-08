"""Heterogeneous entity and transaction graph data generator.

Calibrated to simulate UPI-scale payment networks with:
- Accounts (Retail, Small Business, Student, Corporate)
- Merchants (Aggregators, Irregular, Corner-stores)
- Devices and IP identities
- Calibrated transaction scenarios (A through I)
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from adamule.data.scenarios import ScenarioGenerator
from adamule.utils.config import get_profile_config, get_project_root
from adamule.utils.logging import get_logger
from adamule.utils.seed import set_seed

logger = get_logger("adamule.data.generator")


class SyntheticDataGenerator:
    """Generates synthetic entities and calibrated transaction graphs."""

    def __init__(self, config: Dict[str, Any] | None = None, seed: int = 42):
        self.seed = seed
        set_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.config = config or get_profile_config("development")

    def generate_entities(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate heterogeneous entities: Accounts, Merchants, Devices, IPs."""
        num_accounts = self.config.get("num_accounts", 500)
        num_merchants = self.config.get("num_merchants", 50)
        num_devices = self.config.get("num_devices", 400)
        num_ips = self.config.get("num_ips", 300)

        # 1. Devices
        devices = []
        for i in range(num_devices):
            dtype = self.rng.choice(["android", "ios", "web", "pos"], p=[0.55, 0.25, 0.15, 0.05])
            is_risky = self.rng.random() < 0.08
            devices.append({
                "device_id": f"DEV_{i:06d}",
                "device_type": dtype,
                "age_days": int(self.rng.integers(10, 1800)),
                "risk_indicator": round(float(self.rng.uniform(0.6, 1.0) if is_risky else self.rng.uniform(0.0, 0.2)), 3)
            })

        # 2. IP Identities
        ips = []
        for i in range(num_ips):
            net_type = self.rng.choice(["residential", "mobile", "vpn", "hosting"], p=[0.45, 0.45, 0.08, 0.02])
            is_risky = net_type in ["vpn", "hosting"] or (self.rng.random() < 0.05)
            ips.append({
                "ip_id": f"IP_{i:06d}",
                "region": f"REG_{self.rng.integers(1, 12):02d}",
                "network_type": net_type,
                "risk_indicator": round(float(self.rng.uniform(0.5, 0.95) if is_risky else self.rng.uniform(0.0, 0.15)), 3)
            })

        # 3. Accounts
        accounts = []
        for i in range(num_accounts):
            acc_id = f"ACC_{i:06d}"
            # ~12% small businesses, ~65% retail consumers, ~15% students, ~8% corporate
            segment = self.rng.choice(
                ["retail", "small_business", "student", "corporate"],
                p=[0.65, 0.12, 0.15, 0.08]
            )
            has_business = 1 if segment in ["small_business", "corporate"] else 0
            b_profile = "none"
            if has_business:
                b_profile = self.rng.choice(["retail_store", "wholesaler", "online_merchant", "professional_service"])

            accounts.append({
                "account_id": acc_id,
                "account_type": "business" if has_business else "individual",
                "age_days": int(self.rng.integers(30, 3650)),
                "customer_segment": segment,
                "risk_prior": round(float(self.rng.beta(1.5, 8.0)), 3),
                "location": f"LOC_{self.rng.integers(1, 20):02d}",
                "business_profile": b_profile,
                "has_business_profile": has_business,
            })

        # 4. Merchants (Subset of accounts act as merchant aggregators)
        merchants = []
        business_accs = [a for a in accounts if a["has_business_profile"] == 1]
        if len(business_accs) < num_merchants:
            # Supplement if needed
            business_accs = accounts[:num_merchants]

        selected_m_accs = self.rng.choice(business_accs, size=min(num_merchants, len(business_accs)), replace=False)
        for i, m_acc in enumerate(selected_m_accs):
            m_id = f"MERCH_{i:05d}"
            merchants.append({
                "merchant_id": m_id,
                "merchant_account_id": m_acc["account_id"],
                "merchant_type": self.rng.choice(["grocery", "electronics", "fashion", "dining", "wholesale", "services"]),
                "business_age": m_acc["age_days"],
                "transaction_volume": round(float(self.rng.uniform(50000, 1500000)), 2),
                "average_transaction": round(float(self.rng.uniform(100, 4500)), 2),
                "expected_frequency": round(float(self.rng.uniform(10, 200)), 1),
                "business_category": f"MCC_{self.rng.integers(5000, 6000)}"
            })

        return {
            "accounts": accounts,
            "merchants": merchants,
            "devices": devices,
            "ips": ips
        }

    def generate_transactions(self, entities: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Generate transactions across all calibrated scenarios according to config weights."""
        total_txs = self.config.get("num_transactions", 3000)
        time_span = self.config.get("time_span_days", 30)
        weights = self.config.get("scenarios", {})

        scen_gen = ScenarioGenerator(entities, time_span_days=time_span, seed=self.seed)

        w_a = weights.get("scenario_a_weight", 0.35)
        w_b = weights.get("scenario_b_weight", 0.25)
        w_c = weights.get("scenario_c_weight", 0.10)
        w_d = weights.get("scenario_d_weight", 0.06)
        w_e = weights.get("scenario_e_weight", 0.06)
        w_f = weights.get("scenario_f_weight", 0.06)
        w_g = weights.get("scenario_g_weight", 0.06)
        w_h = weights.get("scenario_h_weight", 0.06)

        total_weight = w_a + w_b + w_c + w_d + w_e + w_f + w_g + w_h

        n_a = int(total_txs * (w_a / total_weight))
        n_b = int(total_txs * (w_b / total_weight))
        n_c = int(total_txs * (w_c / total_weight))
        n_d = int(total_txs * (w_d / total_weight))
        n_e = int(total_txs * (w_e / total_weight))
        n_f = int(total_txs * (w_f / total_weight))
        n_g = int(total_txs * (w_g / total_weight))
        n_h = max(0, total_txs - (n_a + n_b + n_c + n_d + n_e + n_f + n_g))

        logger.info(
            f"Generating {total_txs} transactions: Scen A={n_a}, B={n_b}, C={n_c}, "
            f"D={n_d}, E={n_e}, F={n_f}, G={n_g}, H={n_h}"
        )

        all_txs = []
        if n_a > 0:
            all_txs.extend(scen_gen.generate_scenario_a(n_a))
        if n_b > 0:
            all_txs.extend(scen_gen.generate_scenario_b(n_b))
        if n_c > 0:
            all_txs.extend(scen_gen.generate_scenario_c(n_c))
        if n_d > 0:
            all_txs.extend(scen_gen.generate_scenario_d(n_d))
        if n_e > 0:
            all_txs.extend(scen_gen.generate_scenario_e(n_e))
        if n_f > 0:
            all_txs.extend(scen_gen.generate_scenario_f(n_f))
        if n_g > 0:
            all_txs.extend(scen_gen.generate_scenario_g(n_g))
        if n_h > 0:
            all_txs.extend(scen_gen.generate_scenario_h(n_h))

        df_txs = pd.DataFrame(all_txs)
        # Sort chronologically
        df_txs = df_txs.sort_values(by="timestamp").reset_index(drop=True)
        return df_txs

    def generate_full_dataset(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Generate complete dataset with entity dataframes and transactions dataframe."""
        entities = self.generate_entities()
        df_txs = self.generate_transactions(entities)
        
        entity_dfs = {
            "accounts": pd.DataFrame(entities["accounts"]),
            "merchants": pd.DataFrame(entities["merchants"]),
            "devices": pd.DataFrame(entities["devices"]),
            "ips": pd.DataFrame(entities["ips"])
        }
        return df_txs, entity_dfs

    def save(self, output_dir: str | Path | None = None) -> Path:
        """Generate and save entities and transactions to CSV in raw directory."""
        if output_dir is None:
            output_dir = get_project_root() / "data" / "raw"
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        df_txs, entity_dfs = self.generate_full_dataset()
        df_txs.to_csv(out_path / "transactions.csv", index=False)
        for name, df in entity_dfs.items():
            df.to_csv(out_path / f"{name}.csv", index=False)

        logger.info(
            f"Saved {len(df_txs)} transactions and {len(entity_dfs['accounts'])} accounts to {out_path}"
        )
        return out_path
