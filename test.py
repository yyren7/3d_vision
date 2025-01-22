# run a simple web server that serves the current directory
# usage: python3 -m http.server 8000
# then open a browser and go to http://localhost:8000

import http.server
import socketserver

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()

# to stop the server, press Ctrl+C