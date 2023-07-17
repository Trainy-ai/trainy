import http.server
import json
import logging
import os
from udjat.watcher import Watcher
from udjat.constants import LOCAL_RANK, WORLD_RANK, ip_address


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(
            self.headers["Content-Length"]
        )  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        profiler_config = json.loads(post_data.decode("utf-8"))
        Watcher.start(**profiler_config)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        reply = f'Starting trace on {ip_address}, WORLD_RANK = {WORLD_RANK}. Saving to {os.path.abspath(profiler_config["logdir"])}'
        logging.info(reply)
        self.wfile.write(reply.encode())


def start_server():
    server_address = ("0.0.0.0", 25000 + LOCAL_RANK)
    httpd = http.server.ThreadingHTTPServer(server_address, RequestHandler)
    httpd.serve_forever()
