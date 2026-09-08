// Crypto & Web3 Peel-Chain / Mixer Detection Layer

function openCryptoModal() {
    audio.playClick();
    document.getElementById('cryptoModal').classList.add('active');
}

function closeCryptoModal() {
    audio.playClick();
    document.getElementById('cryptoModal').classList.remove('active');
}

function tracePeelChain() {
    audio.playLaserPulse();
    const resultDiv = document.getElementById('cryptoTraceResult');
    resultDiv.innerHTML = '<div style="color: #38bdf8; text-align: center; padding: 20px;">Scanning Ethereum & Tron Mempools for Peel Chains...</div>';

    setTimeout(() => {
        audio.playAlarm();
        resultDiv.innerHTML = `
            <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 12px; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <b style="color: #f87171;">⛓️ PEEL CHAIN ANOMALY UNMASKED</b>
                    <span style="font-size: 0.7rem; background: rgba(239,68,68,0.2); color:#fca5a5; padding: 2px 6px; border-radius: 4px;">HIGH CONFIDENCE</span>
                </div>
                <div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.5;">
                    • <b>Origin Ingestion:</b> <code>0x71C...a829</code> ($100,000 USDT)<br>
                    • <b>Peel Hop #1:</b> $8,500 sent to Dispersal Wallet <code>0x38B...f812</code>; $91,500 peeled to Change Address.<br>
                    • <b>Peel Hop #2:</b> $9,200 sent to OTC P2P Desk; $82,300 peeled.<br>
                    • <b>Mixer Interaction:</b> 3 hops interacting with Railgun / Privacy Pool Contract.<br>
                    • <b>Destination:</b> High-volume decentralized exchange liquidity pool (Uniswap v3).
                </div>
                <button class="btn-pay" style="width: 100%; margin-top: 10px; background: #ef4444;" onclick="flagCryptoWallet()">
                    🚨 Broadcast Blacklist Flag to Blockchain Oracles
                </button>
            </div>
        `;
        addChatMessage('⛓️ <b>WEB3 AUDIT:</b> USDT Peel Chain detected! 5 peeling hops below $10,000 reporting threshold unmasked.', 'alert');
        if (typeof speakAI === 'function') speakAI('Crypto peel chain identified. 100,000 USDT structured across five hops.');
    }, 1200);
}

function flagCryptoWallet() {
    audio.playFreeze();
    alert('Wallet 0x71C...a829 flagged across Chainalysis / OFAC Compliance screening feed.');
    closeCryptoModal();
}
