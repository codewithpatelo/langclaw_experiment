"""Tiny HTTP server that serves paper_jaiio.pdf with an embedded viewer."""
import http.server
import os
import webbrowser

PORT = 8384
DIR = os.path.dirname(os.path.abspath(__file__))

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>LangClaw Paper Preview</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1e1e2e; font-family: system-ui, sans-serif; }
  header {
    background: #181825; color: #cdd6f4; padding: 12px 24px;
    display: flex; align-items: center; gap: 16px;
    border-bottom: 1px solid #313244;
  }
  header h1 { font-size: 16px; font-weight: 600; }
  header span { font-size: 13px; color: #a6adc8; }
  header a {
    margin-left: auto; color: #89b4fa; text-decoration: none;
    font-size: 13px;
  }
  header a:hover { text-decoration: underline; }
  iframe {
    width: 100%; height: calc(100vh - 48px); border: none;
  }
</style>
</head>
<body>
  <header>
    <h1>LangClaw &mdash; Paper Preview</h1>
    <span>paper_jaiio.pdf (14 pages)</span>
    <a href="/paper_jaiio.pdf" download>Download PDF</a>
  </header>
  <iframe src="/paper_jaiio.pdf"></iframe>
</body>
</html>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode())
        else:
            super().do_GET()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    with http.server.HTTPServer(("0.0.0.0", PORT), Handler) as srv:
        url = f"http://localhost:{PORT}"
        print(f"Serving paper at {url}")
        print("Press Ctrl+C to stop.")
        webbrowser.open(url)
        srv.serve_forever()
