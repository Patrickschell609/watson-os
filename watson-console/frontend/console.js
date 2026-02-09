// Watson Console — Main Logic
// Terminal + Claude chat + sprite integration

// ===== SPRITE INIT =====
const spriteCanvas = document.getElementById('claude-sprite-canvas');
const claude = new ClaudeSprite(spriteCanvas);
const spriteContainer = document.getElementById('claude-sprite-container');
const bubble = new SpeechBubble(spriteContainer);

// ===== DRAGGABLE SPRITE =====
let isDragging = false;
let dragStartX, dragStartY;

spriteContainer.addEventListener('mousedown', (e) => {
  isDragging = true;
  const rect = spriteContainer.getBoundingClientRect();
  dragStartX = e.clientX - rect.left;
  dragStartY = e.clientY - rect.top;
  spriteContainer.style.cursor = 'grabbing';
});

document.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  const parent = spriteContainer.parentElement;
  const parentRect = parent.getBoundingClientRect();
  let x = e.clientX - parentRect.left - dragStartX;
  let y = e.clientY - parentRect.top - dragStartY;
  // Clamp within pane
  x = Math.max(0, Math.min(x, parentRect.width - 64));
  y = Math.max(28, Math.min(y, parentRect.height - 120));
  spriteContainer.style.right = 'auto';
  spriteContainer.style.bottom = 'auto';
  spriteContainer.style.left = x + 'px';
  spriteContainer.style.top = y + 'px';
});

document.addEventListener('mouseup', () => {
  isDragging = false;
  spriteContainer.style.cursor = 'grab';
});

// ===== TERMINAL =====
const terminalEl = document.getElementById('terminal');
const terminalInput = document.getElementById('terminal-input');
let commandHistory = [];
let historyIndex = -1;

function termPrint(text, cls = 'output') {
  const div = document.createElement('div');
  div.className = cls;
  div.textContent = text;
  terminalEl.appendChild(div);
  terminalEl.scrollTop = terminalEl.scrollHeight;
}

function termPrompt(cmd) {
  const div = document.createElement('div');
  div.innerHTML = `<span class="prompt">watson ▸</span> ${escapeHtml(cmd)}`;
  terminalEl.appendChild(div);
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

terminalInput.addEventListener('keydown', async (e) => {
  if (e.key === 'Enter') {
    const cmd = terminalInput.value.trim();
    if (!cmd) return;

    commandHistory.push(cmd);
    historyIndex = commandHistory.length;
    termPrompt(cmd);
    terminalInput.value = '';

    // Execute via Tauri if available, otherwise show placeholder
    if (window.__TAURI__) {
      try {
        const result = await window.__TAURI__.invoke('run_command', { command: cmd });
        if (result.stdout) termPrint(result.stdout);
        if (result.stderr) termPrint(result.stderr, 'error');
      } catch (err) {
        termPrint(`Error: ${err}`, 'error');
      }
    } else {
      // Preview mode (no Tauri backend)
      termPrint(`[preview] Would execute: ${cmd}`, 'system');
    }
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (historyIndex > 0) {
      historyIndex--;
      terminalInput.value = commandHistory[historyIndex];
    }
  } else if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (historyIndex < commandHistory.length - 1) {
      historyIndex++;
      terminalInput.value = commandHistory[historyIndex];
    } else {
      historyIndex = commandHistory.length;
      terminalInput.value = '';
    }
  }
});

// ===== CLAUDE CHAT =====
const messagesEl = document.getElementById('claude-messages');
const claudeInput = document.getElementById('claude-input');

// Auto-resize textarea
claudeInput.addEventListener('input', () => {
  claudeInput.style.height = 'auto';
  claudeInput.style.height = Math.min(claudeInput.scrollHeight, 120) + 'px';
});

// Send on Enter (Shift+Enter for newline)
claudeInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendToClaude();
  }
});

async function sendToClaude() {
  const text = claudeInput.value.trim();
  if (!text) return;

  // Show user message
  addMessage(text, 'user');
  claudeInput.value = '';
  claudeInput.style.height = 'auto';

  // Claude thinks
  claude.setState('thinking');
  bubble.show('Hmm, let me think...');

  if (window.__TAURI__) {
    try {
      const response = await window.__TAURI__.invoke('ask_claude', { prompt: text });
      claude.setState('talking');
      bubble.hide();
      addMessage(response, 'claude');

      // Return to idle after a moment
      setTimeout(() => claude.setState('idle'), 3000);
    } catch (err) {
      claude.setState('idle');
      bubble.hide();
      addMessage(`Error: ${err}`, 'claude');
    }
  } else {
    // Preview mode
    setTimeout(() => {
      claude.setState('talking');
      bubble.hide();
      addMessage("I'm here — this is a preview. Connect the API key in ~/.sheild/telescope.toml to go live.", 'claude');
      setTimeout(() => claude.setState('idle'), 3000);
    }, 1500);
  }
}

function addMessage(text, sender) {
  const div = document.createElement('div');
  div.className = `message ${sender}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ===== TOOLBAR ACTIONS =====
async function watsonVault(action) {
  const cmd = `watson-vault ${action}`;
  termPrompt(cmd);

  if (window.__TAURI__) {
    try {
      const result = await window.__TAURI__.invoke('run_command', { command: cmd });
      if (result.stdout) termPrint(result.stdout);
      if (result.stderr) termPrint(result.stderr, 'error');

      // Update vault status dot
      if (action === 'open') {
        document.getElementById('dot-vault').className = 'dot on';
        document.getElementById('stat-vault').innerHTML = '<span class="dot green"></span> Vault open';
      } else if (action === 'close') {
        document.getElementById('dot-vault').className = 'dot off';
        document.getElementById('stat-vault').innerHTML = '<span class="dot red"></span> Vault locked';
      }
    } catch (err) {
      termPrint(`Error: ${err}`, 'error');
    }
  } else {
    termPrint(`[preview] ${cmd}`, 'system');
  }
}

async function startRecon() {
  const url = prompt('Enter URL or search query:');
  if (!url) return;

  const cmd = `watson-recon ${url}`;
  termPrompt(cmd);
  claude.setState('searching');

  if (window.__TAURI__) {
    try {
      const result = await window.__TAURI__.invoke('run_command', { command: cmd });
      if (result.stdout) termPrint(result.stdout);
      claude.setState('idle');
    } catch (err) {
      termPrint(`Error: ${err}`, 'error');
      claude.setState('idle');
    }
  } else {
    termPrint(`[preview] ${cmd}`, 'system');
    setTimeout(() => claude.setState('idle'), 2000);
  }
}

function toggleLog() {
  const btn = document.getElementById('btn-log');
  const active = btn.classList.toggle('active');
  const cmd = active ? 'watson-log &' : 'pkill -f watson-log';
  termPrompt(cmd);
  termPrint(active ? '[+] Session logging started' : '[-] Session logging stopped', 'system');

  document.getElementById('stat-session').innerHTML = active
    ? '<span class="dot amber"></span> Logging'
    : '<span class="dot amber"></span> Session idle';
}

// ===== CLOCK =====
function updateClock() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2, '0');
  const m = String(now.getMinutes()).padStart(2, '0');
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  document.getElementById('stat-clock').textContent =
    `${h}:${m}  ·  ${months[now.getMonth()]} ${now.getDate()}`;
}
updateClock();
setInterval(updateClock, 30000);

// ===== IDLE BEHAVIORS =====
// Claude occasionally does little things when you're not interacting
let idleTimeout;

function resetIdle() {
  clearTimeout(idleTimeout);
  idleTimeout = setTimeout(() => {
    if (claude.state === 'idle') {
      // Occasionally peek at the terminal or adjust glasses
      const actions = [
        () => { claude.setState('searching'); setTimeout(() => claude.setState('idle'), 2000); },
        () => { bubble.show('...'); setTimeout(() => bubble.hide(), 2000); },
      ];
      actions[Math.floor(Math.random() * actions.length)]();
    }
  }, 120000); // Every 2 min of idle
}

document.addEventListener('keydown', resetIdle);
document.addEventListener('mousemove', resetIdle);
resetIdle();

// ===== DIVIDER RESIZE =====
const divider = document.getElementById('divider');
const claudePane = document.getElementById('claude-pane');
let isResizing = false;

divider.addEventListener('mousedown', (e) => {
  isResizing = true;
  document.body.style.cursor = 'col-resize';
  e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
  if (!isResizing) return;
  const mainRect = document.getElementById('main').getBoundingClientRect();
  const newWidth = mainRect.right - e.clientX;
  if (newWidth >= 280 && newWidth <= 600) {
    claudePane.style.width = newWidth + 'px';
  }
});

document.addEventListener('mouseup', () => {
  isResizing = false;
  document.body.style.cursor = '';
});
