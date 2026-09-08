// Web Speech API Native Voice-Activated AI Forensic Copilot

let recognition = null;
let isVoiceListening = false;
const synth = window.speechSynthesis;

function initVoice() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
        console.warn('Speech Recognition not supported in this browser.');
        return;
    }

    recognition = new SpeechRec();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        const last = event.results.length - 1;
        const cmd = event.results[last][0].transcript.trim().toLowerCase();
        handleVoiceCommand(cmd);
    };

    recognition.onerror = (e) => {
        console.warn('Voice recognition error:', e.error);
        if (e.error === 'not-allowed') {
            stopVoice();
            addChatMessage('🎙️ <b>Microphone Access Denied:</b> Please allow microphone permissions in browser settings.', 'alert');
        }
    };

    recognition.onend = () => {
        if (isVoiceListening) {
            try { recognition.start(); } catch(e) {}
        }
    };
}

function speakAI(text) {
    if (!synth) return;
    try {
        synth.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 0.95;
        synth.speak(utterance);
    } catch(e) {}
}

function toggleVoiceRecognition() {
    audio.playClick();
    if (!recognition) initVoice();
    if (!recognition) {
        alert('Web Speech API is supported on Google Chrome, Microsoft Edge, and Brave.');
        return;
    }

    const voiceBtn = document.getElementById('voiceBtn');
    if (!isVoiceListening) {
        try {
            recognition.start();
            isVoiceListening = true;
            voiceBtn.classList.add('active-red');
            voiceBtn.innerText = '🎙️ Listening...';
            speakAI('AdaMule Voice Copilot active. Ready for command.');
            addChatMessage('🎙️ <b>Voice Copilot Active:</b> Try saying: <i>"Freeze account"</i>, <i>"Trace ring"</i>, <i>"Execute heist"</i>, or <i>"3D globe"</i>.', 'bot');
        } catch (e) {
            console.error(e);
        }
    } else {
        stopVoice();
    }
}

function stopVoice() {
    if (recognition) {
        try { recognition.stop(); } catch(e) {}
    }
    isVoiceListening = false;
    const voiceBtn = document.getElementById('voiceBtn');
    if (voiceBtn) {
        voiceBtn.classList.remove('active-red');
        voiceBtn.innerText = '🎙️ Voice AI';
    }
    speakAI('Voice copilot standby.');
}

function handleVoiceCommand(cmd) {
    addChatMessage(`🗣️ <b>Voice Command Received:</b> "${cmd}"`, 'bot');

    if (cmd.includes('freeze') || cmd.includes('block') || cmd.includes('lock')) {
        freezeSelectedNode();
        speakAI(`Account ${selectedNode.label} has been frozen.`);
    } else if (cmd.includes('trace') || cmd.includes('investigate') || cmd.includes('ring')) {
        traceRingFromSelected();
        speakAI('Tracing multi-hop structuring ring. Direct destination link to ATM cashout hub.');
    } else if (cmd.includes('heist') || cmd.includes('attack') || cmd.includes('launch')) {
        executeSyndicateHeist();
        speakAI('Executing syndicate heist packets across payment rails.');
    } else if (cmd.includes('globe') || cmd.includes('3d') || cmd.includes('world') || cmd.includes('map')) {
        if (!isGlobeActive) toggleGlobeView();
        speakAI('Switching to 3D WebGL cyber globe.');
    } else if (cmd.includes('graph') || cmd.includes('2d') || cmd.includes('network')) {
        if (isGlobeActive) toggleGlobeView();
        speakAI('Switching to 2D force graph view.');
    } else if (cmd.includes('sar') || cmd.includes('report') || cmd.includes('file')) {
        openSARModal();
        speakAI('Generating official Suspicious Activity Report.');
    } else if (cmd.includes('scan') || cmd.includes('x-ray') || cmd.includes('xray')) {
        runContinuousXRay();
        speakAI('Graph X-Ray complete. All nodes verified.');
    } else {
        speakAI('Command acknowledged. Monitoring network.');
    }
}
