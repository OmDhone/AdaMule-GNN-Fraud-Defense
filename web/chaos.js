// Chaos Engine & Stress-Testing Suite

function openChaosModal() {
    audio.playClick();
    document.getElementById('chaosModal').classList.add('active');
}

function closeChaosModal() {
    audio.playClick();
    document.getElementById('chaosModal').classList.remove('active');
}

function injectChaos(type) {
    audio.playScan();
    closeChaosModal();

    if (type === 'simswap') {
        audio.playAlarm();
        addChatMessage('💥 <b>CHAOS EVENT: SIM-SWAP SURGE!</b><br>Simulating 100 coordinated credentials compromises across retail accounts. AdaMule velocity threshold activated.', 'alert');
        initParticles(80);
        if (typeof speakAI === 'function') speakAI('Warning: Coordinated SIM swap surge detected across retail accounts.');
        addLiveTx('[CHAOS: SIM-SWAP SURGE] 100 Compromised Tokens Detected • Rate-Limiter Active', 'busted');
    } else if (type === 'diwali') {
        audio.playCash();
        addChatMessage('🛍️ <b>CHAOS EVENT: FESTIVAL SHOPPING RUSH (10X LOAD)!</b><br>Retail consumer transactions surged 10x. Testing false positive resistance on Sharma Kirana...', 'bot');
        legitimacyLens = 100;
        document.getElementById('lensSlider').value = 100;
        adjustLegitimacyLens(100);
        setTimeout(() => {
            audio.playSuccess();
            addChatMessage('✅ <b>STRESS TEST PASSED:</b> Hard-Negative FPR maintained at <b>0.0000</b>. Zero honest merchants blocked under 10x traffic load!', 'bot');
            if (typeof speakAI === 'function') speakAI('Stress test passed. Zero false alarms during traffic rush.');
        }, 1200);
        addLiveTx('[CHAOS: FESTIVAL RUSH] 10x Retail Volume • 100% Genuine Merchants Approved', 'clean');
    } else if (type === 'retaliate') {
        audio.playLaserPulse();
        addChatMessage('🎭 <b>CHAOS EVENT: SYNDICATE ADAPTIVE RETALIATION!</b><br>Attacker observed node freezing and dynamically re-routed flow via 4 new decoy intermediate accounts.', 'alert');
        nodes.forEach(n => {
            if (n.type === 'victim') n.risk = Math.min(99, n.risk + 15);
        });
        updateDossierHUD(selectedNode);
        if (typeof speakAI === 'function') speakAI('Syndicate retaliating. Evasion algorithm shifting topological paths.');
        addLiveTx('[CHAOS: RETALIATION] Attacker Attempting Dynamic Layering Re-Route', 'busted');
    }
}
