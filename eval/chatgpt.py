"""OpenAI's Sign in with ChatGPT (open-source flow): model calls billed to the user's ChatGPT Plus/Pro plan.
https://developers.openai.com/siwc/token-sharing-open-source

  python -m eval.chatgpt login    # opens the browser; first time registers selfgpt as an app
  python -m eval.chatgpt models   # models available with the plan

Tokens are kept in ~/.config/selfgpt/chatgpt.json (0600). The access token lasts an hour and is refreshed as needed;
the refresh token lasts 30 days, after which login again.
"""

import base64
import hashlib
import json
import secrets
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import openai

ISSUER = "https://auth.openai.com"
AUTHORIZE, TOKEN = f"{ISSUER}/api/accounts/authorize", f"{ISSUER}/api/accounts/oauth/token"
API = "https://api.openai.com/v1"
SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
STORE = Path.home() / ".config" / "selfgpt" / "chatgpt.json"
lock = threading.Lock()


def load():
    return json.loads(STORE.read_text()) if STORE.exists() else {}


def save(d):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    tmp.touch(mode=0o600)
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(STORE)


def store_tokens(d, tokens):
    return d | {k: tokens[k] for k in ("access_token", "refresh_token", "id_token") if k in tokens} | {
        "expires_at": time.time() + tokens["expires_in"]}


def post_token(form):
    with urlopen(TOKEN, urlencode(form).encode(), timeout=60) as r:
        return json.load(r)


def claims(jwt):
    """The ID token's claims. Its signature isn't checked: it comes straight from the token endpoint over TLS."""
    part = jwt.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def login():
    d = load()
    d.setdefault("ext_agent_host_id", f"urn:uuid:{uuid.uuid4()}")
    verifier, state, nonce = secrets.token_urlsafe(64), secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    query = {}

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/callback":
                query.update({k: v[0] for k, v in parse_qs(url.query).items()})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Signed in, you can close this tab.")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Callback)
    redirect = f"http://127.0.0.1:{server.server_port}/callback"
    params = {"client_id": d.get("client_id", "dynamic_agent_client"), "ext_agent_host_id": d["ext_agent_host_id"],
              "response_type": "code", "redirect_uri": redirect, "scope": SCOPE, "resource": API, "state": state,
              "nonce": nonce, "code_challenge_method": "S256", "code_challenge": challenge}
    if "client_id" in d:
        params["id_token_hint"] = d["id_token"]
    else:
        params["agent_name_hint"] = "selfgpt"
    url = f"{AUTHORIZE}?{urlencode(params)}"
    print(f"Sign in at {url}", flush=True)
    webbrowser.open(url)
    while not query:
        server.handle_request()
    if "error" in query:
        sys.exit(f"sign-in failed: {query['error']} {query.get('error_description', '')}")
    if query.get("state") != state:
        sys.exit("sign-in failed: state mismatch")
    client_id = query.get("client_id", d.get("client_id"))
    tokens = post_token({"grant_type": "authorization_code", "client_id": client_id, "code": query["code"],
                         "redirect_uri": redirect, "code_verifier": verifier, "resource": API})
    c = claims(tokens["id_token"])
    if c["iss"] != ISSUER or c["aud"] not in (client_id, [client_id]) or c["nonce"] != nonce:
        sys.exit("sign-in failed: unexpected ID token")
    save(store_tokens(d | {"client_id": client_id, "email": c.get("email")}, tokens))
    print(f"signed in as {c.get('email')}", flush=True)


def access_token():
    with lock:
        d = load()
        if "refresh_token" not in d:
            sys.exit("not signed in to ChatGPT: run python -m eval.chatgpt login")
        if d["expires_at"] - time.time() < 300:
            try:
                tokens = post_token({"grant_type": "refresh_token", "client_id": d["client_id"],
                                     "refresh_token": d["refresh_token"], "resource": API})
            except HTTPError as e:
                if e.code in (400, 401):
                    sys.exit(f"ChatGPT session expired ({e.read().decode()}): run python -m eval.chatgpt login")
                raise
            d = store_tokens(d, tokens)
            save(d)
        return d["access_token"]


def client():
    return openai.OpenAI(api_key=access_token(), base_url=API, max_retries=0, timeout=120)


def respond(model, system, question, effort):
    """The model's text reply at that reasoning effort. The plan flow only streams, and takes no temperature or token limit."""
    text = ""
    with client() as c, c.responses.create(model=model, store=False, stream=True, reasoning={"effort": effort}, input=[
            {"role": "developer", "content": system}, {"role": "user", "content": question}]) as stream:
        for event in stream:
            if event.type == "response.created" and event.response.reasoning.effort != effort:
                sys.exit(f"{model} ran at reasoning effort {event.response.reasoning.effort}, not {effort}")
            if event.type == "response.output_text.delta":
                text += event.delta
            elif event.type in ("response.failed", "response.incomplete"):
                raise RuntimeError(f"{model}: {event.type} {getattr(event.response, 'error', None) or event.response.incomplete_details}")
            elif event.type == "response.completed":
                return text
    raise RuntimeError(f"{model}: stream ended before response.completed")


if __name__ == "__main__":
    if sys.argv[1:] == ["login"]:
        login()
    elif sys.argv[1:] == ["models"]:
        with client() as c:
            for m in c.get("/models", cast_to=object)["models"]:
                if m.get("visibility") == "list":
                    print(m["slug"])
    else:
        sys.exit(__doc__)
