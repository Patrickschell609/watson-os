#!/usr/bin/env python3
"""
VERITAS GUI - Professional deepfake detection interface
Built for Coffeezilla - drag-and-drop simplicity with forensic power
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import sys
import os
import json
import webbrowser
from pathlib import Path
from datetime import datetime

VERSION = "2.0.0"

class VeritasGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"PROJECT VERITAS v{VERSION}")
        self.root.geometry("700x650")
        self.root.configure(bg='#0a0a0f')
        self.root.resizable(True, True)

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 700) // 2
        y = (self.root.winfo_screenheight() - 650) // 2
        self.root.geometry(f"+{x}+{y}")

        self.selected_file = None
        self.analysis_thread = None
        self.is_analyzing = False

        self.setup_styles()
        self.setup_ui()
        self.check_dependencies()

    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Progress bar style
        style.configure(
            "Cyan.Horizontal.TProgressbar",
            troughcolor='#1a1a2e',
            background='#00ffff',
            darkcolor='#00cccc',
            lightcolor='#00ffff',
            bordercolor='#1a1a2e'
        )

    def setup_ui(self):
        """Build the interface"""
        # Main container
        main = tk.Frame(self.root, bg='#0a0a0f')
        main.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Header
        header = tk.Frame(main, bg='#0a0a0f')
        header.pack(fill=tk.X, pady=(0, 20))

        title = tk.Label(
            header,
            text="PROJECT VERITAS",
            font=("Helvetica", 32, "bold"),
            fg="#00ffff",
            bg="#0a0a0f"
        )
        title.pack()

        subtitle = tk.Label(
            header,
            text="Forensic Deepfake Detection",
            font=("Helvetica", 12),
            fg="#666666",
            bg="#0a0a0f"
        )
        subtitle.pack()

        # Dependency status
        self.dep_frame = tk.Frame(header, bg='#0a0a0f')
        self.dep_frame.pack(pady=(10, 0))

        self.dep_label = tk.Label(
            self.dep_frame,
            text="Checking dependencies...",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#0a0a0f"
        )
        self.dep_label.pack()

        # Drop zone
        drop_container = tk.Frame(main, bg='#0a0a0f')
        drop_container.pack(fill=tk.X, pady=10)

        self.drop_frame = tk.Frame(
            drop_container,
            bg='#1a1a2e',
            highlightbackground='#00ffff',
            highlightthickness=2,
            height=120
        )
        self.drop_frame.pack(fill=tk.X)
        self.drop_frame.pack_propagate(False)

        self.drop_label = tk.Label(
            self.drop_frame,
            text="Click to select video file\nor drag and drop",
            font=("Helvetica", 14),
            fg="#888888",
            bg="#1a1a2e",
            cursor="hand2"
        )
        self.drop_label.pack(expand=True)
        self.drop_label.bind("<Button-1>", self.browse_file)
        self.drop_frame.bind("<Button-1>", self.browse_file)

        # URL input
        url_frame = tk.Frame(main, bg='#0a0a0f')
        url_frame.pack(fill=tk.X, pady=15)

        tk.Label(
            url_frame,
            text="Or paste YouTube/video URL:",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#0a0a0f"
        ).pack(anchor=tk.W)

        self.url_entry = tk.Entry(
            url_frame,
            font=("Helvetica", 12),
            bg="#1a1a2e",
            fg="#ffffff",
            insertbackground="#00ffff",
            relief=tk.FLAT,
            highlightbackground="#333333",
            highlightthickness=1
        )
        self.url_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)

        # Options
        options_frame = tk.Frame(main, bg='#0a0a0f')
        options_frame.pack(fill=tk.X, pady=10)

        self.gen_video_var = tk.BooleanVar(value=True)
        self.gen_html_var = tk.BooleanVar(value=True)

        tk.Checkbutton(
            options_frame,
            text="Generate annotated video",
            variable=self.gen_video_var,
            font=("Helvetica", 10),
            fg="#888888",
            bg="#0a0a0f",
            selectcolor="#1a1a2e",
            activebackground="#0a0a0f",
            activeforeground="#00ffff"
        ).pack(side=tk.LEFT)

        tk.Checkbutton(
            options_frame,
            text="Generate HTML report",
            variable=self.gen_html_var,
            font=("Helvetica", 10),
            fg="#888888",
            bg="#0a0a0f",
            selectcolor="#1a1a2e",
            activebackground="#0a0a0f",
            activeforeground="#00ffff"
        ).pack(side=tk.LEFT, padx=(20, 0))

        # Analyze button
        self.analyze_btn = tk.Button(
            main,
            text="ANALYZE",
            font=("Helvetica", 16, "bold"),
            fg="#0a0a0f",
            bg="#00ffff",
            activebackground="#00cccc",
            activeforeground="#0a0a0f",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_analysis
        )
        self.analyze_btn.pack(fill=tk.X, pady=15, ipady=12)

        # Progress section
        progress_frame = tk.Frame(main, bg='#0a0a0f')
        progress_frame.pack(fill=tk.X, pady=10)

        self.phase_label = tk.Label(
            progress_frame,
            text="Ready",
            font=("Helvetica", 11),
            fg="#888888",
            bg="#0a0a0f"
        )
        self.phase_label.pack(anchor=tk.W)

        self.progress = ttk.Progressbar(
            progress_frame,
            style="Cyan.Horizontal.TProgressbar",
            mode='determinate',
            length=400
        )
        self.progress.pack(fill=tk.X, pady=(5, 0))

        self.detail_label = tk.Label(
            progress_frame,
            text="",
            font=("Helvetica", 9),
            fg="#666666",
            bg="#0a0a0f"
        )
        self.detail_label.pack(anchor=tk.W, pady=(5, 0))

        # Results section
        self.result_frame = tk.Frame(main, bg='#0a0a0f')
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Verdict display (hidden initially)
        self.verdict_frame = tk.Frame(self.result_frame, bg='#1a1a2e')

        self.verdict_label = tk.Label(
            self.verdict_frame,
            text="",
            font=("Helvetica", 14, "bold"),
            bg="#1a1a2e",
            wraplength=600
        )
        self.verdict_label.pack(pady=15, padx=15)

        self.confidence_label = tk.Label(
            self.verdict_frame,
            text="",
            font=("Helvetica", 24, "bold"),
            bg="#1a1a2e"
        )
        self.confidence_label.pack(pady=(0, 15))

        # Action buttons (hidden initially)
        self.action_frame = tk.Frame(self.result_frame, bg='#0a0a0f')

        self.open_html_btn = tk.Button(
            self.action_frame,
            text="Open HTML Report",
            font=("Helvetica", 11),
            fg="#00ffff",
            bg="#1a1a2e",
            activebackground="#2a2a4e",
            activeforeground="#00ffff",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.open_html_report
        )
        self.open_html_btn.pack(side=tk.LEFT, padx=5)

        self.open_video_btn = tk.Button(
            self.action_frame,
            text="Open Annotated Video",
            font=("Helvetica", 11),
            fg="#00ffff",
            bg="#1a1a2e",
            activebackground="#2a2a4e",
            activeforeground="#00ffff",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.open_video
        )
        self.open_video_btn.pack(side=tk.LEFT, padx=5)

        self.open_folder_btn = tk.Button(
            self.action_frame,
            text="Open Folder",
            font=("Helvetica", 11),
            fg="#888888",
            bg="#1a1a2e",
            activebackground="#2a2a4e",
            activeforeground="#888888",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.open_folder
        )
        self.open_folder_btn.pack(side=tk.LEFT, padx=5)

        # Footer
        footer = tk.Label(
            main,
            text="Built for investigative journalism • Use responsibly",
            font=("Helvetica", 9),
            fg="#444444",
            bg="#0a0a0f"
        )
        footer.pack(side=tk.BOTTOM, pady=(10, 0))

    def check_dependencies(self):
        """Check for required dependencies"""
        missing = []

        # Check ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True)
        except FileNotFoundError:
            missing.append("ffmpeg")

        # Check Python packages
        try:
            import cv2
        except ImportError:
            missing.append("opencv-python")

        try:
            import numpy
        except ImportError:
            missing.append("numpy")

        if missing:
            self.dep_label.configure(
                text=f"Missing: {', '.join(missing)} - Run install.sh",
                fg="#ff4444"
            )
        else:
            self.dep_label.configure(
                text="All dependencies OK",
                fg="#00ff88"
            )

    def browse_file(self, event=None):
        """Open file browser"""
        filepath = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.selected_file = filepath
            filename = Path(filepath).name
            if len(filename) > 40:
                filename = filename[:37] + "..."
            self.drop_label.config(
                text=f"Selected:\n{filename}",
                fg="#00ffff"
            )
            self.url_entry.delete(0, tk.END)

    def start_analysis(self):
        """Start the analysis"""
        target = self.url_entry.get().strip() or self.selected_file

        if not target:
            messagebox.showwarning(
                "No Input",
                "Please select a video file or enter a URL"
            )
            return

        if self.is_analyzing:
            return

        self.is_analyzing = True
        self.analyze_btn.configure(state=tk.DISABLED, text="ANALYZING...")
        self.progress['value'] = 0
        self.phase_label.configure(text="Starting analysis...", fg="#00ffff")
        self.detail_label.configure(text="")

        # Hide previous results
        self.verdict_frame.pack_forget()
        self.action_frame.pack_forget()

        # Run in thread
        self.analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(target,),
            daemon=True
        )
        self.analysis_thread.start()

        # Start progress polling
        self.poll_progress()

    def run_analysis(self, target):
        """Run VERITAS analysis"""
        try:
            script_dir = Path(__file__).parent

            # Find Python executable
            venv_python = script_dir / ".venv" / "bin" / "python"
            if not venv_python.exists():
                venv_python = script_dir / ".venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                venv_python = sys.executable

            veritas_script = script_dir / "veritas.py"

            cmd = [str(venv_python), str(veritas_script), target]

            if not self.gen_video_var.get():
                cmd.append("--no-video")
            if not self.gen_html_var.get():
                cmd.append("--no-html")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(script_dir)
            )

            self.root.after(0, self.analysis_complete, result)

        except Exception as e:
            self.root.after(0, self.analysis_error, str(e))

    def poll_progress(self):
        """Update progress based on output"""
        if not self.is_analyzing:
            return

        # Simulate progress updates (real progress would parse output)
        current = self.progress['value']
        if current < 95:
            self.progress['value'] = current + 2

        self.root.after(500, self.poll_progress)

    def analysis_complete(self, result):
        """Handle analysis completion"""
        self.is_analyzing = False
        self.analyze_btn.configure(state=tk.NORMAL, text="ANALYZE")
        self.progress['value'] = 100

        script_dir = Path(__file__).parent
        report_path = script_dir / "veritas_report.json"

        if result.returncode == 0 and report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)

                confidence = report.get('confidence', 0)
                verdict = report.get('verdict', 'Unknown')

                # Color based on confidence
                if confidence > 0.75:
                    color = "#ff4444"
                    bg = "#2a1a1a"
                elif confidence > 0.5:
                    color = "#ffa500"
                    bg = "#2a2a1a"
                else:
                    color = "#00ff88"
                    bg = "#1a2a1a"

                self.verdict_frame.configure(bg=bg)
                self.verdict_label.configure(
                    text=verdict,
                    fg=color,
                    bg=bg
                )
                self.confidence_label.configure(
                    text=f"{confidence*100:.0f}%",
                    fg=color,
                    bg=bg
                )

                self.verdict_frame.pack(fill=tk.X, pady=10)
                self.action_frame.pack(pady=10)

                self.phase_label.configure(
                    text="Analysis complete!",
                    fg="#00ff88"
                )
                self.detail_label.configure(
                    text=f"Files saved in: {script_dir}"
                )

            except Exception as e:
                self.phase_label.configure(
                    text="Analysis complete (couldn't read report)",
                    fg="#ffa500"
                )
        else:
            self.phase_label.configure(
                text="Analysis failed",
                fg="#ff4444"
            )
            self.detail_label.configure(
                text=result.stderr[:200] if result.stderr else "Unknown error"
            )

    def analysis_error(self, error):
        """Handle analysis error"""
        self.is_analyzing = False
        self.analyze_btn.configure(state=tk.NORMAL, text="ANALYZE")
        self.progress['value'] = 0
        self.phase_label.configure(text="Error", fg="#ff4444")
        self.detail_label.configure(text=str(error)[:100])
        messagebox.showerror("Error", f"Analysis failed:\n{error}")

    def open_html_report(self):
        """Open HTML report in browser"""
        script_dir = Path(__file__).parent
        report_path = script_dir / "veritas_report.html"
        if report_path.exists():
            webbrowser.open(f"file://{report_path}")
        else:
            messagebox.showinfo("Not Found", "HTML report not found")

    def open_video(self):
        """Open annotated video"""
        script_dir = Path(__file__).parent
        video_path = script_dir / "veritas_output.mp4"
        if video_path.exists():
            if sys.platform == "darwin":
                subprocess.run(["open", str(video_path)])
            elif sys.platform == "win32":
                os.startfile(str(video_path))
            else:
                subprocess.run(["xdg-open", str(video_path)])
        else:
            messagebox.showinfo("Not Found", "Video not found (was --no-video used?)")

    def open_folder(self):
        """Open output folder"""
        script_dir = Path(__file__).parent
        if sys.platform == "darwin":
            subprocess.run(["open", str(script_dir)])
        elif sys.platform == "win32":
            os.startfile(str(script_dir))
        else:
            subprocess.run(["xdg-open", str(script_dir)])

    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def main():
    app = VeritasGUI()
    app.run()


if __name__ == "__main__":
    main()
