"""Feature engineering for transaction graphs.

Extracts node-level, structural, temporal, and business legitimacy features
strictly respecting temporal causality (no forward-looking leakage).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class TemporalFeatureExtractor:
    """Computes features for accounts and transactions with temporal cutoff constraints."""

    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim

    def compute_account_features(
        self,
        accounts_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        cutoff_timestamp: Optional[float] = None
    ) -> pd.DataFrame:
        """Extract node features for accounts using only transactions strictly before cutoff_timestamp.
        
        Args:
            accounts_df: DataFrame of accounts.
            transactions_df: DataFrame of transactions.
            cutoff_timestamp: If provided, only transactions with timestamp <= cutoff are considered.
            
        Returns:
            DataFrame with account_id and engineered numerical feature columns.
        """
        # Filter transactions by temporal cutoff to prevent data leakage
        if cutoff_timestamp is not None:
            valid_txs = transactions_df[transactions_df["timestamp"] <= cutoff_timestamp].copy()
        else:
            valid_txs = transactions_df.copy()

        # Group by sender (outflow) and receiver (inflow)
        out_agg = valid_txs.groupby("sender_id").agg(
            out_degree=("transaction_id", "count"),
            out_volume=("amount", "sum"),
            out_mean_amt=("amount", "mean"),
            out_std_amt=("amount", "std"),
            unique_receivers=("receiver_id", "nunique"),
            unique_devices=("device_id", "nunique"),
            unique_ips=("ip_id", "nunique")
        ).reset_index()

        in_agg = valid_txs.groupby("receiver_id").agg(
            in_degree=("transaction_id", "count"),
            in_volume=("amount", "sum"),
            in_mean_amt=("amount", "mean"),
            in_std_amt=("amount", "std"),
            unique_senders=("sender_id", "nunique")
        ).reset_index()

        # Merge with all accounts
        df_feat = accounts_df[["account_id", "age_days", "risk_prior", "has_business_profile"]].copy()
        df_feat = df_feat.merge(out_agg, left_on="account_id", right_on="sender_id", how="left").drop(columns=["sender_id"], errors="ignore")
        df_feat = df_feat.merge(in_agg, left_on="account_id", right_on="receiver_id", how="left").drop(columns=["receiver_id"], errors="ignore")

        # Fill NaNs for accounts with no transactions in this window
        df_feat["out_degree"] = df_feat["out_degree"].fillna(0.0)
        df_feat["in_degree"] = df_feat["in_degree"].fillna(0.0)
        df_feat["out_volume"] = df_feat["out_volume"].fillna(0.0)
        df_feat["in_volume"] = df_feat["in_volume"].fillna(0.0)
        df_feat["out_mean_amt"] = df_feat["out_mean_amt"].fillna(0.0)
        df_feat["in_mean_amt"] = df_feat["in_mean_amt"].fillna(0.0)
        df_feat["out_std_amt"] = df_feat["out_std_amt"].fillna(0.0)
        df_feat["in_std_amt"] = df_feat["in_std_amt"].fillna(0.0)
        df_feat["unique_receivers"] = df_feat["unique_receivers"].fillna(0.0)
        df_feat["unique_senders"] = df_feat["unique_senders"].fillna(0.0)
        df_feat["unique_devices"] = df_feat["unique_devices"].fillna(1.0)
        df_feat["unique_ips"] = df_feat["unique_ips"].fillna(1.0)

        # Structural & Ratio features
        total_deg = df_feat["in_degree"] + df_feat["out_degree"]
        df_feat["total_degree"] = total_deg
        df_feat["fan_in_ratio"] = np.where(total_deg > 0, df_feat["in_degree"] / (total_deg + 1e-5), 0.0)
        df_feat["fan_out_ratio"] = np.where(total_deg > 0, df_feat["out_degree"] / (total_deg + 1e-5), 0.0)

        total_vol = df_feat["in_volume"] + df_feat["out_volume"]
        df_feat["net_flow"] = df_feat["in_volume"] - df_feat["out_volume"]
        df_feat["volume_balance_ratio"] = np.where(total_vol > 0, df_feat["net_flow"] / (total_vol + 1e-5), 0.0)

        # Inter-arrival & burstiness metrics
        if len(valid_txs) > 0:
            valid_txs["ts_diff"] = valid_txs.groupby("sender_id")["timestamp"].diff().fillna(3600.0)
            burst_stats = valid_txs.groupby("sender_id")["ts_diff"].agg(
                mean_interarrival=("mean"),
                min_interarrival=("min")
            ).reset_index()
            df_feat = df_feat.merge(burst_stats, left_on="account_id", right_on="sender_id", how="left").drop(columns=["sender_id"], errors="ignore")
        else:
            df_feat["mean_interarrival"] = 3600.0
            df_feat["min_interarrival"] = 3600.0

        df_feat["mean_interarrival"] = df_feat["mean_interarrival"].fillna(3600.0)
        df_feat["min_interarrival"] = df_feat["min_interarrival"].fillna(3600.0)

        # Log transform large numbers for numerical stability
        df_feat["log_out_volume"] = np.log1p(df_feat["out_volume"])
        df_feat["log_in_volume"] = np.log1p(df_feat["in_volume"])
        df_feat["log_age_days"] = np.log1p(df_feat["age_days"])

        return df_feat

    def compute_edge_features(self, transactions_df: pd.DataFrame) -> np.ndarray:
        """Extract edge features for each transaction.
        
        Features:
        - log(amount)
        - normalized hour of day
        - day of week
        - one-hot channel (upi_app, web, qr_code, pos)
        - one-hot transaction_type (p2p, p2m, merchant_settlement, intermediary_transfer, atm_cashout)
        """
        feats = []
        timestamps = pd.to_datetime(transactions_df["timestamp"], unit="s")
        hours = (timestamps.dt.hour / 24.0).values
        days = (timestamps.dt.dayofweek / 7.0).values
        log_amounts = np.log1p(transactions_df["amount"].values)

        # Channels
        channels = transactions_df["channel"].fillna("upi_app").values
        c_upi = (channels == "upi_app").astype(float)
        c_web = (channels == "web").astype(float)
        c_qr = (channels == "qr_code").astype(float)
        c_pos = (channels == "pos").astype(float)

        # Types
        ttypes = transactions_df["transaction_type"].fillna("p2p").values
        t_p2p = (ttypes == "p2p").astype(float)
        t_p2m = (ttypes == "p2m").astype(float)
        t_settle = (ttypes == "merchant_settlement").astype(float)
        t_inter = (ttypes == "intermediary_transfer").astype(float)
        t_atm = (ttypes == "atm_cashout").astype(float)

        feature_matrix = np.column_stack([
            log_amounts,
            hours,
            days,
            c_upi,
            c_web,
            c_qr,
            c_pos,
            t_p2p,
            t_p2m,
            t_settle,
            t_inter,
            t_atm
        ])
        return feature_matrix.astype(np.float32)
