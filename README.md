# Watson OS

A 508MB live investigation OS for journalists. Boot from USB, investigate, shut down, leave no trace.

Debian Bookworm. Tor-only networking. No offensive tools. No telemetry. No bloat.

![License](https://img.shields.io/badge/license-MIT-blue)
![Platform](https://img.shields.io/badge/platform-x86__64-lightgrey)
![Size](https://img.shields.io/badge/ISO-508MB-green)

## What This Is

Watson OS is a bootable Linux ISO built for one job: secure open-source investigation. It ships three custom tools on a hardened Debian base with all traffic forced through Tor.

It is not a pentesting distro. There are no exploit frameworks, no password crackers, no packet injection tools. This is a research workstation, not a weapon.

**Who it's for:** Investigative journalists, OSINT researchers, human rights investigators, anyone who needs to research sensitive topics without exposing themselves or their sources.

## The Tools

### SHEILD (Privacy Browser)

Custom browser built with Tauri and Rust on WebKit. It is the only browser on the system. No Firefox. No Chromium.

What it does:
- Routes all traffic through Tor (enforced at the OS level, not optional)
- Fingerprint poisoning: randomizes canvas, WebGL, audio context, and timezone data on every session
- Strips 60+ tracking headers before requests leave the machine
- Ghost Mode: headless Chromium via Playwright for targets that block Tor exit nodes. Rotates identity per request.

What it does not do:
- No JavaScript consent popups (stripped)
- No cookie persistence across sessions
- No WebRTC (disabled at the engine level, prevents IP leaks)

### Telescope (AI Research Proxy)

Built into SHEILD. Lets you point an LLM at a target URL and get a summary without your machine ever touching the target.

How it works:
1. You give Telescope a URL and a question
2. The configured AI provider (Claude, GPT, Ollama, or any OpenAI-compatible API) fetches and reads the page
3. You get the answer. Your IP never appears in the target's access logs.

The security model: the AI provider sees your prompt but not your identity (Tor). The target sees the AI provider's IP but not yours. Neither side gets the full picture.

Supports local models (Ollama, LM Studio) for air-gapped operation. Config lives at `~/.sheild/telescope.toml`.

### Veritas (Deepfake Detector)

9-method video analysis pipeline for verifying media authenticity. Built in Python with a Rust core for the heavy lifting.

Methods include:
- Facial landmark consistency analysis (mediapipe)
- Audio-visual sync drift detection
- Compression artifact analysis
- Temporal coherence checks
- Metadata forensics

Usage:
```bash
watson-verify video.mp4              # Analyze local file
watson-verify https://youtu.be/...   # Download and analyze
watson-verify --gui                  # Graphical interface
```

Reports auto-save to `~/Investigation/evidence/veritas/` when the investigation vault is mounted.

Python dependencies install on first boot (saves 300MB from the ISO).

### Ghost Analyst (OSINT Intelligence Platform)

GPU-accelerated (wgpu/Vulkan) analysis interface. Drop evidence files in, paste URLs, get pattern analysis.

```bash
watson-analyst                       # Open GUI
watson-analyst evidence.pdf          # Open with file loaded
```

Uses the same Telescope AI config as SHEILD. Results save to `~/Investigation/analyst/`.

### Ghost Code (Text Editor)

Custom GPU-accelerated text editor built with wgpu and Rust. This is the default text editor on Watson OS. No Gedit, no Mousepad, no Vim.

What it does:
- Syntax highlighting for Python, Rust, JavaScript, Bash, JSON, HTML, CSS, and more
- Tabbed editing with drag-and-drop file support
- Full undo/redo, search, select all, clipboard integration
- Mountain Cabin theme (consistent with the rest of the OS)
- GPU-rendered text (smooth on any hardware, software Vulkan fallback)

```bash
ghost-code                    # Open empty editor
ghost-code notes.md           # Open a file
ghost-code *.py               # Open multiple files in tabs
```

### CLI Tools

| Command | What it does |
|---------|-------------|
| `watson-recon` | AI-powered OSINT data collection (Telescope integration) |
| `watson-verify` | Veritas deepfake analysis (9 methods) |
| `watson-analyst` | GPU-accelerated AI pattern analysis |
| `watson-exif` | Image metadata extraction, GPS coordinates, EXIF stripping |
| `watson-user` | Username search across 35+ platforms via Tor |
| `watson-domain` | Domain intelligence: DNS, WHOIS, certs, tech detection, subdomains |
| `watson-archive` | Retrieve deleted web content from Wayback Machine and archive.today |
| `watson-timeline` | Build investigation timelines from all vault evidence |
| `watson-log` | Session activity logging |
| `watson-vault` | LUKS2 encrypted evidence management |
| `ghost-code` | Text editor with syntax highlighting |

## Network Security

All network traffic is forced through Tor at the kernel level. This is not a browser setting. It is an iptables firewall that drops everything that is not Tor.

### How it works

```
┌─────────────────────────────────────────┐
│  Watson OS                              │
│                                         │
│  Application (any)                      │
│       │                                 │
│       ▼                                 │
│  iptables nat table                     │
│   ├── DNS (port 53) ──► Tor DNSPort     │
│   │                     (127.0.0.1:5353)│
│   └── TCP (all) ──────► Tor TransPort   │
│                         (127.0.0.1:9040)│
│       │                                 │
│       ▼                                 │
│  Tor process (only process allowed      │
│  outbound access, matched by UID)       │
│       │                                 │
│       ▼                                 │
│  Internet (via Tor circuit)             │
└─────────────────────────────────────────┘

Everything else: DROP + LOG
```

### Firewall rules

- Default policy: DROP on INPUT, FORWARD, and OUTPUT
- Only the Tor process (matched by `debian-tor` UID) can reach the internet
- All DNS redirected to Tor's DNSPort (5353)
- All TCP redirected to Tor's TransPort (9040)
- UDP (except DNS): blocked. Tor does not support UDP. This prevents leaks.
- IPv6: disabled system-wide (Tor does not support it, and it leaks identity)
- Dropped packets are logged with `[WATSON DROP]` prefix for auditing

### What this prevents

- DNS leaks (all DNS goes through Tor, not your ISP)
- WebRTC IP leaks (disabled in SHEILD, blocked by firewall even if re-enabled)
- UDP leaks (dropped at kernel level)
- IPv6 leaks (disabled in sysctl)
- Application-level bypasses (the firewall does not care what app made the request)

## System Hardening

Beyond the firewall, Watson OS applies 30+ kernel and system-level hardening measures:

**Kernel (sysctl):**
- ASLR set to maximum (`randomize_va_space = 2`)
- No IP forwarding
- No ICMP redirects (prevents MITM)
- No ICMP echo responses (invisible to ping sweeps)
- SYN flood protection enabled
- Source routing disabled
- Martian packet logging enabled
- dmesg restricted to root
- Kernel pointer addresses hidden (`kptr_restrict = 2`)
- ptrace restricted (`yama.ptrace_scope = 2`)
- Magic SysRq disabled
- Core dumps disabled system-wide

**Services:**
- avahi (mDNS) disabled and masked. It broadcasts your hostname on the local network.
- CUPS (printing) disabled and masked. Attack surface with no investigative value.
- Bluetooth disabled and masked
- Telemetry and popularity-contest removed
- motd-news disabled

**Login:**
- Root account locked. Use sudo.
- `/tmp` mounted noexec, nosuid, nodev
- Autologin to `watson` user (live session, no password prompt to shoulder-surf)

## Theme

Mountain Cabin. Dark wood, amber lamplight, cream text. Designed for long sessions in low light. Not flashy, not distracting.

- GTK3 + GTK2 + XFWM4 theme included
- XFCE4 Terminal: Liberation Mono, amber cursor, dark background
- LightDM greeter with investigation-themed boot splash
- Bottom panel, thin, dark

## Architecture

Watson OS is built with Debian `live-build`. The ISO is assembled through a series of hooks that run in order during the chroot phase:

| Hook | Purpose |
|------|---------|
| `0050-debloat` | Strip docs, man pages, unused locales, unnecessary firmware. Keeps the ISO under 510MB. |
| `0100-harden` | Kernel hardening, Tor firewall, service lockdown, core dump prevention |
| `0200-sheild` | Install SHEILD as system browser, configure Telescope, set up Ghost Mode |
| `0300-theme` | Mountain Cabin theme: GTK, terminal, LightDM, wallpaper, bash prompt |
| `0400-veritas` | Veritas deepfake detector: first-boot venv setup, CLI wrapper, desktop entry |
| `0500-analyst` | Ghost Analyst: OSINT analysis GUI, Vulkan drivers for GPU rendering |
| `0600-ghost-code` | Ghost Code: text editor, file associations, default editor |
| `9999-final-cleanup` | Wipe apt cache and temp files |

Custom binaries and assets are placed via `config/includes.chroot_after_packages/`:
```
opt/
  sheild/          SHEILD browser binary + frontend + Ghost Mode
  veritas/         Veritas source (Python + Rust)
  analyst/         ghost-analyst binary (13MB, GPU-accelerated)
  ghost-code/      Ghost Code text editor (10MB, GPU-accelerated)
usr/
  local/bin/       watson-recon, watson-exif, watson-user, watson-domain,
                   watson-archive, watson-timeline, watson-log, watson-vault
  share/backgrounds/  Mountain cabin wallpaper
  share/watson/       Boot splash
```

## Building

### Requirements

- Debian Bookworm (or compatible) build host
- `live-build` package installed
- ~4GB free disk space
- Root access (live-build runs chroot operations)
- SHEILD source at `/home/oday/Desktop/SHEILD` (for the full build script)

### Quick build (ISO only, skip SHEILD compile)

```bash
cd watson-os
sudo lb clean          # Clean previous binary output (keeps cached chroot)
sudo lb build          # Build the ISO
```

The cached bootstrap saves about 10 minutes on rebuilds. Do NOT use `lb clean --purge` unless you want a full rebuild from scratch.

### Full build (compile SHEILD + build ISO)

```bash
./build.sh
```

This compiles SHEILD from Rust source, packages it into the ISO includes, then runs `lb build`.

### Output

```
watson-os-amd64.hybrid.iso    # Bootable ISO (USB + CD compatible)
```

### Flash to USB

```bash
sudo dd if=watson-os-amd64.hybrid.iso of=/dev/sdX bs=4M status=progress
```

Replace `/dev/sdX` with your USB device. Double check. `dd` does not ask for confirmation.

### Test in a VM

```bash
qemu-system-x86_64 -m 2G -cdrom watson-os-amd64.hybrid.iso
```

## Package List

Every package is justified. Nothing extra.

**Desktop:** XFCE4 (session, panel, settings, terminal, screenshooter, window manager, file manager, LightDM)

**Display:** xorg core, libinput, fbdev, vesa (covers bare metal and VMs)

**Network:** NetworkManager, Tor, torsocks, iptables, UFW

**Browser runtime:** WebKit2GTK 4.1, GTK3, librsvg, Node.js (Ghost Mode)

**Encryption:** cryptsetup, GnuPG, OpenSSH client, KeePassXC, AppArmor

**Media:** PulseAudio, VLC (investigators watch/listen to source material)

**Filesystem:** gvfs, udisks2, NTFS, exFAT (read evidence drives)

**Python:** python3, pip, venv (for Veritas)

**Veritas deps:** ffmpeg, tesseract-ocr, python3-tk

**Utilities:** git, curl, wget, htop, file, unzip, rsync

**Fonts:** Liberation (one readable set, nothing else)

## Comparison to Alternatives

| | Watson OS | Tails | Whonix |
|---|---|---|---|
| **Size** | 508MB | ~1.3GB | ~2GB+ (two VMs) |
| **Tor enforcement** | iptables, kernel-level | iptables, kernel-level | Separate gateway VM |
| **Browser** | SHEILD (custom, fingerprint poisoning) | Tor Browser | Tor Browser |
| **AI integration** | Telescope (built-in proxy) | None | None |
| **Deepfake detection** | Veritas (9-method) | None | None |
| **OSINT tools** | Ghost Analyst, watson-recon | None (by design) | None |
| **Offensive tools** | None | None | None |
| **Target user** | Investigative journalists | Anyone needing anonymity | Privacy-focused users |
| **Persistence** | Live only (no install) | Optional encrypted persistence | Full install |

Watson OS is not trying to replace Tails or Whonix. Tails is for general anonymity. Whonix is for compartmentalized privacy. Watson OS is specifically for investigation work: researching targets, verifying media, analyzing patterns, all without exposing the journalist.

## What This Is Not

- Not a pentesting distro (no Metasploit, no hashcat, no Burp Suite)
- Not a daily driver (it is a live ISO, not an installed OS)
- Not a general-purpose anonymity tool (use Tails for that)
- Not audited by a third party (yet)

If this OS is seized during a raid, the package list speaks for itself. Research tools. No weapons.

## Project Status

Working. The ISO boots, Tor enforces, SHEILD browses, Veritas analyzes. It has been tested on bare metal and in QEMU.

This is a solo project. Contributions, testing, and feedback are welcome.

## License

MIT
