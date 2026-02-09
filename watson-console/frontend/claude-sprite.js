// Claude — The Investigator
// 32x32 pixel art sprite, drawn on canvas at 2x scale (64x64 display)
// Warm, friendly detective. Trench coat, round glasses, magnifying glass.
// He lives in the corner of the console and reacts to what's happening.

const SCALE = 2;
const SIZE = 32;
const DISPLAY = SIZE * SCALE;

// Palette — Watson cabin colors
const P = {
  _: null,                    // transparent
  K: '#0f0e0d',              // outline (cabin dark)
  C: '#d4a24e',              // coat (amber/brass)
  c: '#b8882a',              // coat shadow
  S: '#e8dcc8',              // skin (cream)
  s: '#c4b8a0',              // skin shadow
  H: '#5c3a1a',              // hair (dark brown)
  G: '#8ecae6',              // glasses lens (sky blue — friendly)
  g: '#4a8aaa',              // glasses frame
  M: '#aaa294',              // magnifying glass (pewter)
  m: '#7a756a',              // magnifying handle
  W: '#e8dcc8',              // shirt (cream)
  B: '#362b22',              // boots (leather)
  b: '#2a2118',              // boots dark
  E: '#1a1a1a',              // eyes
  L: '#4a7a56',              // lens glow (banker's lamp green)
  T: '#5c2a2a',              // tie (burgundy)
};

// IDLE — standing, slight smile, magnifying glass at side
// 32x32 grid, each char maps to palette
const SPRITE_IDLE = [
  '________________________________',
  '________________________________',
  '___________KKKKKKKK____________',
  '__________KHHHHHHHHK___________',
  '_________KHHHHHHHHHKK__________',
  '_________KHHHHHHHHHHK__________',
  '________KKKKKKKKKKKKKK_________',
  '________KSSSSSSSSSSsK__________',
  '________KSgGGgSgGGgSK__________',
  '________KSgGEgSgEGgSK__________',
  '________KSSSSSSSSSSsK__________',
  '_________KSSSssSSSK____________',
  '_________KSS_TT_SSK____________',
  '__________KSSTTSSKK____________',
  '__________KKKTTKKKK____________',
  '_________KCCCTTCCCcK___________',
  '________KCCCCTTCCCCcK__________',
  '________KCCCCTTCCCCcK__________',
  '________KCCCCWWCCCCcK__________',
  '________KCCCWWWWCCCcK__________',
  '________KCCSWWWWSCCcK__________',
  '_________KCSWWWWSCcK___________',
  '_________KCCWWWWCCcK___________',
  '__________KCCCCCCcK____________',
  '__________KCCccCCcK____________',
  '___________KccccK_KMK__________',
  '___________KBBBBK_KmK__________',
  '___________KBBbBK__KmK_________',
  '___________KBBbBK___KK_________',
  '____________KKKK_______________',
  '________________________________',
  '________________________________',
];

// THINKING — head tilted, hand on chin, magnifying glass raised
const SPRITE_THINK = [
  '________________________________',
  '________________________________',
  '____________KKKKKKKK___________',
  '___________KHHHHHHHHK__________',
  '__________KHHHHHHHHHKK_________',
  '__________KHHHHHHHHHHK_________',
  '_________KKKKKKKKKKKKKK________',
  '_________KSSSSSSSSSSsK_________',
  '_________KSgGGgSgGGgSK_________',
  '_________KSgGEgSgEGgSK_________',
  '_________KSSSSSSSSSSsK_________',
  '__________KSSSssSSSK___________',
  '________SSKSSSsSSsK____________',
  '________SSKSSSSSSKK____________',
  '_________SKKKKKKKK_____________',
  '________KCCCTTCCCcK____________',
  '________KCCCTTCCCcK____KLMK___',
  '_______KCCCCTTCCCCcK__KLLmK___',
  '_______KCCCCWWCCCCcK_KLLmK____',
  '_______KCCCWWWWCCCcKKmmmK_____',
  '_______KCCSWWWWSCCcKKK________',
  '________KCSWWWWSCcKK___________',
  '________KCCWWWWCCcK____________',
  '_________KCCCCCCcK_____________',
  '_________KCCccCCcK_____________',
  '__________KccccK_______________',
  '__________KBBBBK_______________',
  '__________KBBbBK_______________',
  '__________KBBbBK_______________',
  '___________KKKK________________',
  '________________________________',
  '________________________________',
];

// TALKING — mouth open, slight lean forward, hand gesture
const SPRITE_TALK = [
  '________________________________',
  '________________________________',
  '___________KKKKKKKK____________',
  '__________KHHHHHHHHK___________',
  '_________KHHHHHHHHHKK__________',
  '_________KHHHHHHHHHHK__________',
  '________KKKKKKKKKKKKKK_________',
  '________KSSSSSSSSSSsK__________',
  '________KSgGGgSgGGgSK__________',
  '________KSgGEgSgEGgSK__________',
  '________KSSSSSSSSSSsK__________',
  '_________KSSSssSSSK____________',
  '_________KSS_KK_SSK____________',
  '_________KSSKKKSSKK____________',
  '__________KKKTTKKKK____________',
  '_________KCCCTTCCCcK___________',
  '________KCCCCTTCCCCcK__________',
  '______SSKCCCCTTCCCCcK__________',
  '______SSKCCCCWWCCCCSKSS________',
  '_______SKCCCWWWWCCCcKSS________',
  '________KCCSWWWWSCCcKS_________',
  '_________KCSWWWWSCcK___________',
  '_________KCCWWWWCCcK___________',
  '__________KCCCCCCcK____________',
  '__________KCCccCCcK____________',
  '___________KccccK______________',
  '___________KBBBBK______________',
  '___________KBBbBK______________',
  '___________KBBbBK______________',
  '____________KKKK_______________',
  '________________________________',
  '________________________________',
];

// SEARCHING — leaning forward with magnifying glass up to eye
const SPRITE_SEARCH = [
  '________________________________',
  '________________KLLK___________',
  '___________KKKKKLLLK___________',
  '__________KHHHHHHKLK___________',
  '_________KHHHHHHHHKK___________',
  '_________KHHHHHHHHHHK__________',
  '________KKKKKKKKKKKKKK_________',
  '________KSSSSSSSSSSsK__________',
  '________KSgGGgKLLLKSK__________',
  '________KSgGEgKLELKSK__________',
  '________KSSSSsKLLLKsK__________',
  '_________KSSSssKKKK____________',
  '_________KSSSssSmKK____________',
  '__________KSSSSSmmK____________',
  '__________KKKKKKmmK____________',
  '_________KCCCTTCmmcK___________',
  '________KCCCCTTCCCCcK__________',
  '________KCCCCTTCCCCcK__________',
  '________KCCCCWWCCCCcK__________',
  '________KCCCWWWWCCCcK__________',
  '________KCCSWWWWSCCcK__________',
  '_________KCSWWWWSCcK___________',
  '_________KCCWWWWCCcK___________',
  '__________KCCCCCCcK____________',
  '__________KCCccCCcK____________',
  '___________KccccK______________',
  '___________KBBBBK______________',
  '___________KBBbBK______________',
  '___________KBBbBK______________',
  '____________KKKK_______________',
  '________________________________',
  '________________________________',
];

class ClaudeSprite {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.canvas.width = DISPLAY;
    this.canvas.height = DISPLAY;
    this.state = 'idle';
    this.frame = 0;
    this.blinkTimer = 0;
    this.idleTimer = 0;
    this.bobOffset = 0;

    // Speech bubble
    this.speechText = '';
    this.speechVisible = false;

    // Dragging
    this.dragging = false;
    this.dragOffsetX = 0;
    this.dragOffsetY = 0;

    this._animate();
  }

  setState(state) {
    if (this.state !== state) {
      this.state = state;
      this.frame = 0;
    }
  }

  say(text) {
    this.speechText = text;
    this.speechVisible = true;
  }

  hideSpeech() {
    this.speechVisible = false;
  }

  _getSprite() {
    switch (this.state) {
      case 'thinking': return SPRITE_THINK;
      case 'talking':  return SPRITE_TALK;
      case 'searching': return SPRITE_SEARCH;
      default:         return SPRITE_IDLE;
    }
  }

  _render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, DISPLAY, DISPLAY);

    const sprite = this._getSprite();

    // Subtle breathing bob on idle
    let yOff = 0;
    if (this.state === 'idle') {
      this.bobOffset += 0.05;
      yOff = Math.sin(this.bobOffset) * 0.5;
    }

    for (let y = 0; y < SIZE; y++) {
      const row = sprite[y];
      if (!row) continue;
      for (let x = 0; x < SIZE; x++) {
        const ch = row[x];
        if (ch === '_' || ch === ' ' || !ch) continue;
        const color = P[ch];
        if (!color) continue;
        ctx.fillStyle = color;
        ctx.fillRect(
          x * SCALE,
          Math.round((y + yOff) * SCALE),
          SCALE,
          SCALE
        );
      }
    }
  }

  _animate() {
    this.frame++;

    // Idle eye blink every ~120 frames
    if (this.state === 'idle' && this.frame % 120 === 0) {
      // Could modify sprite for 2 frames to show blink
    }

    this._render();
    requestAnimationFrame(() => this._animate());
  }
}

// Speech bubble renderer
class SpeechBubble {
  constructor(container) {
    this.el = document.createElement('div');
    this.el.className = 'speech-bubble';
    this.el.style.cssText = `
      display: none;
      position: absolute;
      bottom: ${DISPLAY + 8}px;
      right: 0;
      max-width: 280px;
      padding: 10px 14px;
      background: #2a2118;
      border: 1px solid #4a3c30;
      border-radius: 8px;
      color: #e8dcc8;
      font-family: 'Liberation Sans', sans-serif;
      font-size: 13px;
      line-height: 1.4;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    `;
    container.appendChild(this.el);

    // Triangle pointer
    const arrow = document.createElement('div');
    arrow.style.cssText = `
      position: absolute;
      bottom: -8px;
      right: 24px;
      width: 0;
      height: 0;
      border-left: 8px solid transparent;
      border-right: 8px solid transparent;
      border-top: 8px solid #2a2118;
    `;
    this.el.appendChild(arrow);
  }

  show(text) {
    this.el.style.display = 'block';
    // Insert text before the arrow
    const textNode = this.el.childNodes[0];
    if (textNode && textNode.nodeType === 3) {
      textNode.textContent = text;
    } else {
      this.el.insertBefore(document.createTextNode(text), this.el.firstChild);
    }
  }

  hide() {
    this.el.style.display = 'none';
  }

  setText(text) {
    // Clear text nodes, keep arrow
    while (this.el.childNodes.length > 1) {
      this.el.removeChild(this.el.firstChild);
    }
    this.el.insertBefore(document.createTextNode(text), this.el.firstChild);
  }
}

// Export for use in console
window.ClaudeSprite = ClaudeSprite;
window.SpeechBubble = SpeechBubble;
window.CLAUDE_STATES = { IDLE: 'idle', THINKING: 'thinking', TALKING: 'talking', SEARCHING: 'searching' };
