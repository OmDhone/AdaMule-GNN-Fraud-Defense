#!/usr/bin/env python3
"""Launcher for the AdaMule Fintech Cyber War Room Web Application.

Usage:
    python scripts/launch_war_room.py --port 8080
"""

import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Launch the AdaMule Cyber Defense War Room.")
    parser.add_argument("--port", type=int, default=8080, help="Port to host the war room web application.")
    args = parser.parse_args()

    web_dir = Path(__file__).resolve().parent.parent / "web"
    if not web_dir.exists():
        print(f"Error: web directory not found at {web_dir}")
        return

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args_h, **kwargs):
            super().__init__(*args_h, directory=str(web_dir), **kwargs)
        def log_message(self, format, *args):
            # Suppress noisy HTTP request logging in console
            pass

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    port = args.port
    for p in range(port, port + 15):
        try:
            with ReusableTCPServer(("", p), CustomHandler) as httpd:
                url = f"http://localhost:{p}"
                print(r"""
========================================================================
   ___    ____   ___    __  __ _   _ _     _____ 
  / _ \  |  _ \ / _ \  |  \/  | | | | |   | ____|
 / /_\ \ | | | / /_\ \ | |\/| | | | | |   |  _|  
|  _   | | |_| |  _  | | |  | | |_| | |___| |___ 
|_| |__| |____/|_| |_| |_|  |_|\___/|_____|_____|
               CYBER HEIST & DEFENSE WAR ROOM
========================================================================
⚡ Active URL:         """ + url + r"""
🎮 Red Team Mode:      Syndicate Heist Simulator (Smurfing, Camouflage, Jitters)
🛡️ Blue Team Mode:     AdaMule AI Sentinel (Physics Graph, Freeze, SAR Export)
📱 Simulation Rails:   PhonePe / UPI Smartphone Mockup (Instant Payments)
📻 Procedural Audio:   Synthesized Web Audio Sound Effects (Sirens, Laser, Chimes)
========================================================================
🚀 Opening browser automatically... Press Ctrl+C in terminal to stop.
""")
                webbrowser.open(url)
                httpd.serve_forever()
                break
        except OSError:
            continue

if __name__ == "__main__":
    main()

