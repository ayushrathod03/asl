/**
 * ASL Word Autocompleter & Sign Predictor Agent
 * Client-Side Application Logic with MediaPipe Hands & Real-Time Smoothing
 */

// ---------------------------------------------------------------------------
// Global Application State
// ---------------------------------------------------------------------------
const state = {
    isCameraRunning: false,
    cameraStream: null,
    isFlipped: true,
    detectionMode: 'client', // 'client' or 'server'
    
    // MediaPipe & Canvas
    handsDetector: null,
    videoElement: null,
    canvasElement: null,
    canvasCtx: null,

    // Smoothing & Consensus Buffer
    bufferSize: 10,
    predictionBuffer: [],
    consensusThreshold: 0.75,
    isCooldown: false,
    cooldownDuration: 850, // ms pause after registering a letter

    // Text Workspace
    typedText: "",
    currentSuggestions: [],
    
    // Performance & FPS
    frameCount: 0,
    lastFpsUpdate: performance.now(),
    fps: 0,
    isPredicting: false
};

// ASL 21 Landmark Connections definition
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],          // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],          // Index
    [9, 10], [10, 11], [11, 12],             // Middle
    [13, 14], [14, 15], [15, 16],            // Ring
    [0, 17], [17, 18], [18, 19], [19, 20],   // Pinky
    [5, 9], [9, 13], [13, 17]                // Palm
];

// ---------------------------------------------------------------------------
// Initialization on DOM Loaded
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initDOMElements();
    initMediaPipe();
    setupEventListeners();
    fetchSystemStatus();
    populateAlphabetReference();
});

function initDOMElements() {
    state.videoElement = document.getElementById('webcam-video');
    state.canvasElement = document.getElementById('skeleton-canvas');
    state.canvasCtx = state.canvasElement.getContext('2d');
}

// ---------------------------------------------------------------------------
// MediaPipe Hands Initialization
// ---------------------------------------------------------------------------
function initMediaPipe() {
    try {
        if (typeof Hands !== 'undefined') {
            state.handsDetector = new Hands({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
            });

            state.handsDetector.setOptions({
                maxNumHands: 2,
                modelComplexity: 1,
                minDetectionConfidence: 0.45,
                minTrackingConfidence: 0.45
            });

            state.handsDetector.onResults(onMediaPipeResults);
            updateTrackingStatus("MediaPipe Ready", "var(--accent-cyan)");
        } else {
            console.warn("MediaPipe CDN Hands not loaded. Defaulting to server fallback mode.");
            setPipelineMode('server');
        }
    } catch (err) {
        console.error("Error setting up client MediaPipe Hands:", err);
        setPipelineMode('server');
    }
}

// ---------------------------------------------------------------------------
// Camera Controls (Start / Stop / Flip)
// ---------------------------------------------------------------------------
async function startCamera() {
    if (state.isCameraRunning) return;

    try {
        updateTrackingStatus("Requesting camera...", "var(--accent-amber)");
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });

        state.cameraStream = stream;
        state.videoElement.srcObject = stream;

        await new Promise((resolve) => {
            state.videoElement.onloadedmetadata = () => {
                state.videoElement.play();
                resolve();
            };
        });

        // Sync canvas dimensions
        state.canvasElement.width = state.videoElement.videoWidth || 640;
        state.canvasElement.height = state.videoElement.videoHeight || 480;

        state.isCameraRunning = true;
        document.getElementById('camera-placeholder').classList.add('hidden');
        updateCameraButtons(true);
        updateTrackingStatus("Tracking Active", "var(--accent-emerald)");

        // Start processing animation frame loop
        requestAnimationFrame(processVideoFrame);
        showToast("Camera started successfully", "success");
    } catch (err) {
        console.error("Camera access error:", err);
        updateTrackingStatus("Camera access denied", "var(--accent-rose)");
        showToast(`Camera error: ${err.message || 'Permission denied'}`, "error");
    }
}

function stopCamera() {
    if (!state.isCameraRunning) return;

    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(track => track.stop());
        state.cameraStream = null;
    }
    state.videoElement.srcObject = null;
    state.isCameraRunning = false;

    // Clear canvas
    state.canvasCtx.clearRect(0, 0, state.canvasElement.width, state.canvasElement.height);

    document.getElementById('camera-placeholder').classList.remove('hidden');
    updateCameraButtons(false);
    updateTrackingStatus("Camera Stopped", "var(--text-secondary)");
    resetHandCards();
}

function updateCameraButtons(isRunning) {
    const btn = document.getElementById('toggle-camera-btn');
    const icon = document.getElementById('toggle-camera-icon');
    const text = document.getElementById('toggle-camera-text');

    if (isRunning) {
        btn.classList.add('stop-state');
        icon.textContent = '⏹';
        text.textContent = 'Stop Camera';
    } else {
        btn.classList.remove('stop-state');
        icon.textContent = '▶';
        text.textContent = 'Start Camera';
    }
}

function toggleCameraFlip() {
    state.isFlipped = !state.isFlipped;
    state.videoElement.style.transform = state.isFlipped ? 'scaleX(-1)' : 'scaleX(1)';
    showToast(`View ${state.isFlipped ? 'Mirrored' : 'Normal'}`);
}

// ---------------------------------------------------------------------------
// Video Frame Processing Loop
// ---------------------------------------------------------------------------
async function processVideoFrame() {
    if (!state.isCameraRunning) return;

    // Calculate FPS
    calculateFPS();

    if (state.detectionMode === 'client' && state.handsDetector) {
        try {
            await state.handsDetector.send({ image: state.videoElement });
        } catch (err) {
            console.error("Frame detection error:", err);
        }
    } else if (state.detectionMode === 'server') {
        processServerFallbackFrame();
    }

    requestAnimationFrame(processVideoFrame);
}

function calculateFPS() {
    state.frameCount++;
    const now = performance.now();
    const elapsed = now - state.lastFpsUpdate;
    if (elapsed >= 1000) {
        state.fps = Math.round((state.frameCount * 1000) / elapsed);
        document.getElementById('fps-counter').textContent = `${state.fps} FPS`;
        state.frameCount = 0;
        state.lastFpsUpdate = now;
    }
}

// ---------------------------------------------------------------------------
// Client-Side MediaPipe Results Handler
// ---------------------------------------------------------------------------
async function onMediaPipeResults(results) {
    const ctx = state.canvasCtx;
    const width = state.canvasElement.width;
    const height = state.canvasElement.height;

    ctx.save();
    ctx.clearRect(0, 0, width, height);

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
        resetHandCards();
        handleNoHandsDetected();
        ctx.restore();
        return;
    }

    const handsPayload = [];

    // Draw and prepare landmarks for each hand
    for (let hIdx = 0; hIdx < results.multiHandLandmarks.length; hIdx++) {
        const rawLandmarks = results.multiHandLandmarks[hIdx];
        const handedness = (results.multiHandedness && results.multiHandedness[hIdx])
            ? results.multiHandedness[hIdx].label
            : `Hand ${hIdx + 1}`;

        // Draw styled neon skeleton
        drawHandSkeleton(ctx, rawLandmarks, width, height, hIdx);

        // Extract normalized 63-dim array
        const normalizedVector = extractNormalizedVector(rawLandmarks);
        handsPayload.push({
            landmarks: normalizedVector,
            normalized: true,
            handedness: handedness
        });
    }

    ctx.restore();

    // Send payload to backend KNN
    if (handsPayload.length > 0 && !state.isPredicting) {
        sendLandmarksForPrediction(handsPayload);
    }
}

// ---------------------------------------------------------------------------
// Exact Landmark Normalization Matching Training Logic
// ---------------------------------------------------------------------------
function extractNormalizedVector(landmarks) {
    // 1. Convert to array of [x, y, z]
    const pts = landmarks.map(lm => [lm.x, lm.y, lm.z]);

    // 2. Translate so wrist (index 0) is at origin (0, 0)
    const wristX = pts[0][0];
    const wristY = pts[0][1];

    for (let i = 0; i < pts.length; i++) {
        pts[i][0] -= wristX;
        pts[i][1] -= wristY;
    }

    // 3. Compute scale = max 2D Euclidean norm
    let maxScale = 0.0;
    for (let i = 0; i < pts.length; i++) {
        const dist = Math.sqrt(pts[i][0] * pts[i][0] + pts[i][1] * pts[i][1]);
        if (dist > maxScale) {
            maxScale = dist;
        }
    }

    // 4. Divide (x, y) by scale if > 1e-6
    if (maxScale > 1e-6) {
        for (let i = 0; i < pts.length; i++) {
            pts[i][0] /= maxScale;
            pts[i][1] /= maxScale;
        }
    } else {
        for (let i = 0; i < pts.length; i++) {
            pts[i][0] = 0;
            pts[i][1] = 0;
        }
    }

    // Flatten into 63 floats array
    const flattened = [];
    for (let i = 0; i < pts.length; i++) {
        flattened.push(pts[i][0], pts[i][1], pts[i][2]);
    }
    return flattened;
}

// ---------------------------------------------------------------------------
// Stylized Skeleton Rendering on Canvas
// ---------------------------------------------------------------------------
function drawHandSkeleton(ctx, landmarks, width, height, handIndex) {
    const isFlipped = state.isFlipped;
    const accentColor = handIndex === 0 ? '#38bdf8' : '#a855f7';
    const jointColor = handIndex === 0 ? '#10b981' : '#f43f5e';

    // Map normalized coordinates to canvas space
    const canvasPoints = landmarks.map(lm => {
        let x = lm.x * width;
        if (isFlipped) {
            x = width - x; // Mirror x coordinate on canvas to match selfie video
        }
        return {
            x: x,
            y: lm.y * height
        };
    });

    // 1. Draw glowing connector bones
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = accentColor;
    ctx.shadowColor = accentColor;
    ctx.shadowBlur = 12;

    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const p1 = canvasPoints[startIdx];
        const p2 = canvasPoints[endIdx];

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
    }

    // 2. Draw glowing joint nodes
    for (let i = 0; i < canvasPoints.length; i++) {
        const pt = canvasPoints[i];
        const radius = (i === 4 || i === 8 || i === 12 || i === 16 || i === 20) ? 6 : 4;

        ctx.beginPath();
        ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = (i === 0) ? '#ffffff' : jointColor;
        ctx.shadowColor = jointColor;
        ctx.shadowBlur = 10;
        ctx.fill();

        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
    }
}

// ---------------------------------------------------------------------------
// Send Normalized Landmarks to Backend for KNN Prediction
// ---------------------------------------------------------------------------
async function sendLandmarksForPrediction(hands) {
    state.isPredicting = true;
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hands: hands })
        });

        if (!response.ok) return;

        const data = await response.json();
        if (data.success && data.predictions) {
            handlePredictionResults(data.predictions);
        }
    } catch (err) {
        console.error("API Predict Error:", err);
    } finally {
        state.isPredicting = false;
    }
}

// ---------------------------------------------------------------------------
// Prediction Handler & Sliding Window Buffer Smoothing
// ---------------------------------------------------------------------------
function handlePredictionResults(predictions) {
    if (!predictions || predictions.length === 0) {
        handleNoHandsDetected();
        return;
    }

    // Update Hand Cards UI
    for (let i = 0; i < 2; i++) {
        const card = document.getElementById(`hand-${i+1}-card`);
        const letterElem = document.getElementById(`hand-${i+1}-letter`);
        const confElem = document.getElementById(`hand-${i+1}-confidence`);
        const barElem = document.getElementById(`hand-${i+1}-bar`);
        const identityElem = document.getElementById(`hand-${i+1}-identity`);

        if (i < predictions.length) {
            const pred = predictions[i];
            const confPct = Math.round(pred.confidence * 100);

            card.classList.add('active');
            letterElem.textContent = pred.label.toUpperCase();
            confElem.textContent = `${confPct}%`;
            barElem.style.width = `${confPct}%`;
            identityElem.textContent = pred.handedness || `Hand ${i+1}`;
        } else {
            card.classList.remove('active');
            letterElem.textContent = '-';
            confElem.textContent = '--%';
            barElem.style.width = '0%';
        }
    }

    // Use primary hand (first hand detected) for typing buffer
    const primaryPred = predictions[0];
    if (primaryPred && primaryPred.label && primaryPred.label !== 'unknown') {
        feedPredictionIntoBuffer(primaryPred.label);
    }
}

function feedPredictionIntoBuffer(label) {
    if (state.isCooldown) return;

    state.predictionBuffer.push(label);
    if (state.predictionBuffer.length > state.bufferSize) {
        state.predictionBuffer.shift();
    }

    // Calculate consensus
    const counts = {};
    let maxCount = 0;
    let mostFrequent = label;

    for (const item of state.predictionBuffer) {
        counts[item] = (counts[item] || 0) + 1;
        if (counts[item] > maxCount) {
            maxCount = counts[item];
            mostFrequent = item;
        }
    }

    const consensus = maxCount / state.bufferSize;
    const consensusPct = Math.round(consensus * 100);

    // Update consensus UI bar
    const progressFill = document.getElementById('consensus-progress-fill');
    const percentText = document.getElementById('consensus-percent');
    progressFill.style.width = `${consensusPct}%`;
    percentText.textContent = `${consensusPct}%`;

    // Lock in character if consensus threshold achieved
    if (state.predictionBuffer.length >= state.bufferSize && consensus >= state.consensusThreshold) {
        commitLetter(mostFrequent);
    }
}

function commitLetter(letter) {
    state.typedText += letter;
    updateTypedDisplay();
    state.predictionBuffer = [];

    // Trigger visual lock and start cooldown
    flashTypedBox();
    state.isCooldown = true;

    const progressFill = document.getElementById('consensus-progress-fill');
    progressFill.style.width = '100%';

    setTimeout(() => {
        state.isCooldown = false;
        progressFill.style.width = '0%';
        document.getElementById('consensus-percent').textContent = '0%';
    }, state.cooldownDuration);

    // Trigger autocomplete query
    queryAutocomplete();
}

function handleNoHandsDetected() {
    state.predictionBuffer = [];
    document.getElementById('consensus-progress-fill').style.width = '0%';
    document.getElementById('consensus-percent').textContent = '0%';
}

function resetHandCards() {
    for (let i = 1; i <= 2; i++) {
        const card = document.getElementById(`hand-${i}-card`);
        if (card) {
            card.classList.remove('active');
            document.getElementById(`hand-${i}-letter`).textContent = '-';
            document.getElementById(`hand-${i}-confidence`).textContent = '--%';
            document.getElementById(`hand-${i}-bar`).style.width = '0%';
        }
    }
}

// ---------------------------------------------------------------------------
// Server-Side Fallback Pipeline
// ---------------------------------------------------------------------------
let lastServerFrameSent = 0;
async function processServerFallbackFrame() {
    const now = performance.now();
    if (now - lastServerFrameSent < 180 || state.isPredicting) return; // throttle ~5 FPS
    lastServerFrameSent = now;

    // Grab frame from video to canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 320;
    tempCanvas.height = 240;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(state.videoElement, 0, 0, 320, 240);
    const base64Data = tempCanvas.toDataURL('image/jpeg', 0.6);

    state.isPredicting = true;
    try {
        const response = await fetch('/api/predict_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Data })
        });

        if (!response.ok) return;

        const data = await response.json();
        if (data.success) {
            handlePredictionResults(data.predictions);
        }
    } catch (err) {
        console.error("Server fallback error:", err);
    } finally {
        state.isPredicting = false;
    }
}

// ---------------------------------------------------------------------------
// Text Workspace & Actions
// ---------------------------------------------------------------------------
function updateTypedDisplay() {
    const contentElem = document.getElementById('typed-text-content');
    contentElem.textContent = state.typedText;

    // Auto-scroll to bottom of typed box
    const displayBox = document.getElementById('typed-output-display');
    displayBox.scrollTop = displayBox.scrollHeight;

    // Update stats
    const chars = state.typedText.length;
    const words = state.typedText.trim() ? state.typedText.trim().split(/\s+/).length : 0;
    document.getElementById('char-count').textContent = `${chars} chars`;
    document.getElementById('word-count').textContent = `${words} words`;
}

function flashTypedBox() {
    const box = document.getElementById('typed-output-display');
    box.style.borderColor = 'var(--accent-emerald)';
    box.style.boxShadow = '0 0 25px rgba(16, 185, 129, 0.4)';
    setTimeout(() => {
        box.style.borderColor = '';
        box.style.boxShadow = '';
    }, 300);
}

function addSpace() {
    if (state.typedText.length > 0 && !state.typedText.endsWith(' ')) {
        state.typedText += ' ';
        updateTypedDisplay();
        queryAutocomplete();
    }
}

function deleteLastChar() {
    if (state.typedText.length > 0) {
        state.typedText = state.typedText.slice(0, -1);
        updateTypedDisplay();
        queryAutocomplete();
    }
}

function clearAllText() {
    if (!state.typedText) return;
    state.typedText = "";
    updateTypedDisplay();
    state.predictionBuffer = [];
    renderSuggestions([]);
    showToast("Cleared text workspace");
}

function copyToClipboard() {
    if (!state.typedText) {
        showToast("Nothing to copy!", "error");
        return;
    }
    navigator.clipboard.writeText(state.typedText).then(() => {
        showToast("Text copied to clipboard! 📋", "success");
    }).catch(err => {
        showToast("Failed to copy text", "error");
    });
}

function speakTextAloud() {
    const textToSpeak = state.typedText.trim();
    if (!textToSpeak) {
        showToast("Please sign or type some text first!", "error");
        return;
    }

    if (!('speechSynthesis' in window)) {
        showToast("Text-to-speech is not supported in this browser.", "error");
        return;
    }

    window.speechSynthesis.cancel(); // Stop any pending speech
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    
    // Choose natural English voice if available
    const voices = window.speechSynthesis.getVoices();
    const naturalVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha')));
    if (naturalVoice) {
        utterance.voice = naturalVoice;
    }

    utterance.onstart = () => {
        document.getElementById('btn-speak').classList.add('pulse');
    };
    utterance.onend = () => {
        document.getElementById('btn-speak').classList.remove('pulse');
    };

    window.speechSynthesis.speak(utterance);
    showToast("Speaking text aloud 🔊");
}

// ---------------------------------------------------------------------------
// Smart Autocomplete Service
// ---------------------------------------------------------------------------
function getCurrentPrefix() {
    if (!state.typedText) return "";
    const words = state.typedText.split(/\s+/);
    return words[words.length - 1].toLowerCase().trim();
}

async function queryAutocomplete() {
    const prefix = getCurrentPrefix();
    if (!prefix || !/^[a-zA-Z]+$/.test(prefix)) {
        renderSuggestions([]);
        return;
    }

    try {
        const response = await fetch(`/api/autocomplete?prefix=${encodeURIComponent(prefix)}&limit=6`);
        if (!response.ok) return;

        const data = await response.json();
        renderSuggestions(data.suggestions || []);
    } catch (err) {
        console.error("Autocomplete error:", err);
    }
}

function renderSuggestions(suggestions) {
    state.currentSuggestions = suggestions;
    const container = document.getElementById('suggestions-container');
    container.innerHTML = "";

    if (!suggestions || suggestions.length === 0) {
        container.innerHTML = `
            <div class="suggestion-placeholder">
                <span>Sign or type letters to see real-time suggestions from 370k+ words</span>
            </div>
        `;
        return;
    }

    suggestions.forEach((word, index) => {
        const chip = document.createElement('div');
        chip.className = 'suggestion-chip';
        chip.setAttribute('role', 'button');
        chip.setAttribute('tabindex', '0');
        chip.innerHTML = `
            <span class="chip-word">${escapeHtml(word)}</span>
            <span class="chip-key">${index + 1}</span>
        `;
        chip.addEventListener('click', () => applySuggestion(word));
        container.appendChild(chip);
    });
}

function applySuggestion(chosenWord) {
    const words = state.typedText.split(/\s+/);
    if (words.length > 0) {
        words[words.length - 1] = chosenWord;
        state.typedText = words.join(' ') + ' ';
    } else {
        state.typedText = chosenWord + ' ';
    }

    updateTypedDisplay();
    queryAutocomplete();
    flashTypedBox();
    showToast(`Autocompleted "${chosenWord}"`, "success");
}

// ---------------------------------------------------------------------------
// UI Helpers, Status & Modal
// ---------------------------------------------------------------------------
async function fetchSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.status === 'ready') {
            document.getElementById('model-status-text').textContent = 'KNN Model Ready';
            document.getElementById('vocab-count-text').textContent = `${(data.vocab_size / 1000).toFixed(0)}k Words`;
        }
    } catch (err) {
        console.warn("Status fetch failed:", err);
    }
}

function setPipelineMode(mode) {
    state.detectionMode = mode;
    const select = document.getElementById('detection-mode-select');
    if (select) select.value = mode;

    const badge = document.getElementById('engine-badge');
    if (badge) {
        badge.textContent = mode === 'client' ? 'Client MediaPipe (Ultra-Fast)' : 'Server OpenCV Fallback';
    }
}

function updateTrackingStatus(text, color) {
    const badge = document.getElementById('tracking-badge');
    if (badge) {
        badge.textContent = text;
        if (color) badge.style.color = color;
    }
}

function populateAlphabetReference() {
    const grid = document.getElementById('asl-alphabet-grid');
    if (!grid) return;

    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    letters.forEach(char => {
        const card = document.createElement('div');
        card.className = 'asl-letter-card';
        card.innerHTML = `
            <div class="asl-letter-char">${char}</div>
            <div class="asl-letter-label">Letter ${char}</div>
        `;
        card.addEventListener('click', () => {
            commitLetter(char.toLowerCase());
            closeModal();
        });
        grid.appendChild(card);
    });
}

function openModal() {
    const modal = document.getElementById('chart-modal-backdrop');
    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
}

function closeModal() {
    const modal = document.getElementById('chart-modal-backdrop');
    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 300);
    }, 2800);
}

function escapeHtml(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

// ---------------------------------------------------------------------------
// Event Listeners Setup
// ---------------------------------------------------------------------------
function setupEventListeners() {
    // Camera toggles
    document.getElementById('toggle-camera-btn').addEventListener('click', () => {
        if (state.isCameraRunning) stopCamera();
        else startCamera();
    });

    document.getElementById('start-cam-placeholder-btn').addEventListener('click', startCamera);
    document.getElementById('flip-camera-btn').addEventListener('click', toggleCameraFlip);

    // Mode selector
    document.getElementById('detection-mode-select').addEventListener('change', (e) => {
        setPipelineMode(e.target.value);
        showToast(`Switched pipeline to ${e.target.value === 'client' ? 'Client MediaPipe' : 'Server Fallback'}`);
    });

    // Toolbar buttons
    document.getElementById('btn-space').addEventListener('click', addSpace);
    document.getElementById('btn-backspace').addEventListener('click', deleteLastChar);
    document.getElementById('btn-clear').addEventListener('click', clearAllText);
    document.getElementById('btn-copy').addEventListener('click', copyToClipboard);
    document.getElementById('btn-speak').addEventListener('click', speakTextAloud);

    // Modal triggers
    document.getElementById('toggle-guide-btn').addEventListener('click', openModal);
    document.getElementById('close-modal-btn').addEventListener('click', closeModal);
    document.getElementById('chart-modal-backdrop').addEventListener('click', (e) => {
        if (e.target.id === 'chart-modal-backdrop') closeModal();
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
        // Prevent key capture if typing inside form input
        if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

        // Number keys 1-6 for autocomplete options
        if (e.key >= '1' && e.key <= '6') {
            const idx = parseInt(e.key) - 1;
            if (state.currentSuggestions && state.currentSuggestions[idx]) {
                e.preventDefault();
                applySuggestion(state.currentSuggestions[idx]);
                return;
            }
        }

        if (e.code === 'Space') {
            e.preventDefault();
            addSpace();
        } else if (e.code === 'Backspace') {
            e.preventDefault();
            deleteLastChar();
        } else if (e.code === 'Escape') {
            closeModal();
        }
    });
}
