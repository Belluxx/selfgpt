"""OpenAI-compatible API. The client talks as a normal "user"; roles are swapped before the template."""

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from selfgpt.chat import ReverseChat

SWAP = {"user": "assistant", "assistant": "user"}


def text_of(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content or ""


def serve(chat: ReverseChat, model_name: str, host: str, port: int):
    class Handler(BaseHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def send_json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_json({"object": "list", "data": [
                {"id": model_name, "object": "model", "created": 0, "owned_by": "selfgpt"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = [{"role": SWAP.get(m["role"], m["role"]), "content": text_of(m.get("content"))}
                        for m in req["messages"]]
            params = {k: req[k] for k in ("temperature", "top_p", "min_p", "top_k", "seed", "stop") if req.get(k) is not None}
            if max_tokens := req.get("max_completion_tokens") or req.get("max_tokens"):
                params["max_tokens"] = max_tokens
            base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time()), "model": model_name}

            if req.get("stream"):
                return self.stream_response(base, chat.stream(messages, **params))
            text, finish = "", "stop"
            for piece, reason in chat.stream(messages, **params):
                text, finish = text + piece, reason or finish
            self.send_json({**base, "object": "chat.completion", "choices": [{
                "index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text.strip()}}]})

        def stream_response(self, base, pieces):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(delta, finish=None):
                chunk = {**base, "object": "chat.completion.chunk",
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            send({"role": "assistant", "content": ""})
            finish = "stop"
            for piece, reason in pieces:
                if piece:
                    send({"content": piece})
                finish = reason or finish
            send({}, finish)
            self.wfile.write(b"data: [DONE]\n\n")

    # Single-threaded: one Llama instance can't serve concurrent requests.
    print(f"Serving OpenAI-compatible API on http://{host}:{port}/v1 (model: {model_name})")
    HTTPServer((host, port), Handler).serve_forever()
