# Deployment

How to run the ALS screening app in production, and the handful of decisions
that actually matter when you do.

**Before anything else:** this is a research screening aid, not a diagnostic
device. If you deploy it anywhere real patients or real recordings are involved,
the regulatory and clinical-governance questions are yours to answer, and they
are not questions this document can settle. Section 6 lists what the software
does *not* do for you.

---

## 1. Local — Docker Compose

The fastest correct way to run it.

```bash
cd als-webapp

# 1. put the trained artifacts in ./models  (see models/README.md)
ls models/
#   cnn_best.pt  meta_learner.pkl  florence_als_head.pt  florence_finetuned/

# 2. build and start
docker compose up --build

# 3. open http://localhost:8000
```

One container serves both the API and the built frontend on port 8000. Weights
are mounted **read-only** — the app has no reason to write there, and a
read-only mount means nothing in the container can corrupt them.

No artifacts yet? Bring the stack up on placeholder weights:

```bash
DEMO_MODE=1 docker compose up --build
```

Useful commands:

```bash
docker compose logs -f als-webapp          # JSON logs, one object per line
docker compose ps                          # healthcheck state
docker compose exec als-webapp curl -s localhost:8000/health | python3 -m json.tool
docker compose down                        # stop
```

### Knobs

Everything in `docker-compose.yml` reads from the environment, so a `.env` file
beside it (or real environment variables) overrides without editing YAML:

```bash
HOST_PORT=8080
WEB_CONCURRENCY=1              # see §3 before raising this
ENABLE_FLORENCE=1
FLORENCE_DEFAULT=0             # 1 = run the VLM on every request
RATE_LIMIT=20
MEMORY_LIMIT=6g
CORS_ORIGINS=https://als.example.org
```

---

## 2. Without Docker

```bash
cd als-webapp/backend
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt gunicorn

cd ../frontend && npm ci && VITE_API_BASE= npm run build && cd ../backend

ENV=production SERVE_FRONTEND=1 \
FRONTEND_DIR=../frontend/dist \
gunicorn -c gunicorn_conf.py main:app
```

`VITE_API_BASE=` (empty) matters. In the container the API lives at the **root**
(`/health`, `/predict`) and shares an origin with the SPA, so the base is empty.
`/api` is only the Vite **dev-server** proxy prefix. Build with `/api` for
same-origin serving and every API call falls through to `index.html`, and the UI
reports "server unreachable" with no other clue.

### systemd

```ini
# /etc/systemd/system/als-webapp.service
[Unit]
Description=ALS screening API
After=network.target

[Service]
Type=exec
User=als
WorkingDirectory=/opt/als-webapp/backend
EnvironmentFile=/opt/als-webapp/.env
ExecStart=/opt/als-webapp/backend/.venv/bin/gunicorn -c gunicorn_conf.py main:app
ExecReload=/bin/kill -s HUP $MAINPID
KillSignal=SIGTERM
TimeoutStopSec=90
Restart=always
RestartSec=5

# hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/als-webapp/outputs

[Install]
WantedBy=multi-user.target
```

`TimeoutStopSec=90` clears `graceful_timeout`, so a prediction in flight is
allowed to finish instead of being killed halfway through writing a PDF.

---

## 3. Sizing — the one decision that bites

**Each worker loads its own complete copy of every model.** They do not share
memory. Workers multiply RAM.

| Configuration | RAM per worker | Sensible workers | Latency per signal |
|---|---|---|---|
| CNN only (`ENABLE_FLORENCE=0`) | ~400 MB | 2–4 | ~1.5 s |
| CNN + Florence-2 loaded, VLM off by default | ~1.6 GB | 1–2 | ~1.5 s fast path |
| Florence-2 on every request (`FLORENCE_DEFAULT=1`) | ~1.6 GB | **1** | 10–30 s on CPU |
| Plus Qwen2.5 (`ENABLE_LLM=1`) | ~4.5 GB | 1 | +5–15 s |

Start with `WEB_CONCURRENCY=2` and the CNN fast path. Enable Florence-2 only
with the memory to back it, and drop to one worker when you do.

Two more consequences of the process model, both deliberate:

- **The inference lock is per process.** One prediction at a time per worker,
  because Florence-2 and Qwen are not thread-safe. Parallelism comes from
  workers, which is a memory trade, not a free one.
- **The rate limit is per process.** `RATE_LIMIT=20` with 2 workers admits up to
  40/min in the worst case. If you need an exact global limit, enforce it at the
  reverse proxy — a shared counter would mean Redis, and a Redis outage that
  fails open buys nothing over this.

Predictions are stored on disk in `outputs/` precisely so this works: any worker
can serve `/report/{id}` for a prediction another worker made.

---

## 4. Behind a reverse proxy — Nginx + Let's Encrypt

```nginx
server {
    listen 443 ssl http2;
    server_name als.example.org;

    ssl_certificate     /etc/letsencrypt/live/als.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/als.example.org/privkey.pem;

    # Uploads are ~100 KB, but leave room for the configured limit.
    client_max_body_size 30m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # A CPU prediction with the VLM on can take 30s+. Below Gunicorn's 120s.
        proxy_read_timeout 180s;
        proxy_send_timeout 180s;
    }
}

server {
    listen 80;
    server_name als.example.org;
    return 301 https://$host$request_uri;
}
```

```bash
sudo certbot --nginx -d als.example.org
```

Then set, on the app:

```bash
ENV=production
ENABLE_HSTS=1                       # only once TLS actually works
TRUST_PROXY_HEADERS=1               # ONLY behind a proxy you control
TRUSTED_HOSTS=als.example.org
CORS_ORIGINS=https://als.example.org
```

`TRUST_PROXY_HEADERS=1` makes the app believe `X-Forwarded-For`. Turn it on
**only** when a proxy you control overwrites that header — otherwise any client
can forge an IP and walk straight past the rate limiter.

---

## 5. Cloud

### Google Cloud Run

Cloud Run's filesystem is in-memory and instances are recycled freely, so mount
the weights rather than baking them in, and expect `outputs/` to be ephemeral —
which is fine, since results expire anyway.

```bash
gcloud builds submit --tag gcr.io/PROJECT/als-webapp als-webapp/

gcloud run deploy als-webapp \
  --image gcr.io/PROJECT/als-webapp \
  --region us-central1 \
  --cpu 2 --memory 4Gi \
  --concurrency 4 \
  --timeout 300 \
  --min-instances 1 \
  --set-env-vars ENV=production,ENABLE_HSTS=1,WEB_CONCURRENCY=1 \
  --add-volume name=models,type=cloud-storage,bucket=YOUR_BUCKET \
  --add-volume-mount volume=models,mount-path=/app/models
```

`--min-instances 1` is not optional in practice: a cold start loads the models,
and the first request would otherwise wait out the whole load.

### AWS ECS Fargate

- Task: 2 vCPU / 4 GB for the CNN fast path; 4 vCPU / 8 GB with Florence-2.
- Weights on EFS, mounted read-only at `/app/models`.
- ALB target group health check → `/health/ready`, healthy threshold 2,
  interval 30s, **timeout 10s**, and a start period long enough to cover model
  loading (120s+).
- Set `stopTimeout: 90` so ECS honours the graceful shutdown.

### A plain VPS

Docker Compose plus the Nginx block above is the whole deployment. 4 GB is
comfortable for the CNN fast path; 8 GB if you turn the VLM on.

---

## 6. Health checks, and which to use where

| Endpoint | Answers | Use it for |
|---|---|---|
| `GET /health/live` | is the process up? | container/liveness probes; never touches models |
| `GET /health/ready` | can this worker score a signal? | load-balancer readiness; 503 until the CNN is loaded |
| `GET /health` | full artifact and capability detail | dashboards, debugging, the UI's status panel |

Point liveness at `/health/live` and readiness at `/health/ready`. Pointing
liveness at `/health/ready` will restart a container that is merely still
loading Florence-2 — an infinite restart loop that looks like a crash.

---

## 7. Security checklist before you expose it

- [ ] `ENV=production` — hides `/docs` and `/openapi.json`, enables HSTS defaults
- [ ] `CORS_ORIGINS` lists exact origins. `*` is **refused** when `ENV=production`
- [ ] `TRUSTED_HOSTS` set, so Host-header injection cannot forge absolute URLs
- [ ] TLS terminated, `ENABLE_HSTS=1` only after it works
- [ ] `TRUST_PROXY_HEADERS=1` only behind a proxy that overwrites `X-Forwarded-For`
- [ ] `RATE_LIMIT` sized for your users (per worker — see §3)
- [ ] `MAX_UPLOAD_MB` no larger than you need; a signal is ~100 KB
- [ ] `models/` mounted read-only
- [ ] Container runs as non-root (the image does this already, UID 10001)
- [ ] Log destination is one you are allowed to send this data to

Headers the app sets on every response: `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`,
`Cross-Origin-Opener-Policy`, a strict `Content-Security-Policy`, and
`Strict-Transport-Security` when enabled. Prediction and report responses also
carry `Cache-Control: no-store` so no proxy retains a result.

### Air-gapped or font-blocked networks

The UI pulls Instrument Serif, Plus Jakarta Sans and JetBrains Mono from Google
Fonts, and the CSP allows exactly those two hosts. On a network that cannot reach
them the page still renders correctly through its fallback stacks (Georgia,
system sans, system mono) — verified, not assumed. To remove the dependency
entirely, drop the `<link>` from `frontend/index.html`, put the woff2 files in
`frontend/public/fonts/`, add `@font-face` rules to `styles.css`, and tighten the
CSP with `CONTENT_SECURITY_POLICY` so `font-src`/`style-src` are `'self'` only.

### What this app does NOT give you

Be clear-eyed about the gap between "hardened" and "clinical":

- **No authentication.** Every endpoint is open to anyone who can reach it. Put
  it behind an authenticating proxy, a VPN, or an SSO gateway before it touches
  anything real.
- **No audit trail.** Logs record that a signal was scored and what the verdict
  was; there is no tamper-evident record tying a result to a person.
- **No encryption at rest.** Uploads live in memory; generated images and PDFs
  sit unencrypted in `outputs/` until the TTL sweeps them.
- **No PHI handling.** Nothing here is HIPAA/GDPR-compliant on its own. Filenames
  are logged — do not put patient identifiers in them.
- **Uploads are not scanned.** `.pkl` input is unpickled, and **unpickling
  executes code**. Only accept `.pkl` from sources you trust; if that is not
  true for your deployment, remove `.pkl` and `.pickle` from
  `SUPPORTED_SUFFIXES` in `signal_io.py` and accept `.npy`/`.csv` only.

---

## 8. Observability

Logs are one JSON object per line, carrying `timestamp`, `level`, `logger`,
`message`, `pid`, and for anything in a request: `request_id`, `client_ip`,
`method`, `path`, `status`, `latency_ms`.

```json
{"timestamp":"2026-08-30T11:57:12.004Z","level":"INFO","logger":"als.api",
 "message":"scored signal","pid":2149,"request_id":"3f93aac0f168","client_ip":"10.0.0.4",
 "method":"POST","path":"/predict","record_id":"9f343a5c7b81","prediction":"ALS",
 "confidence":0.6698,"fusion":"cnn_only","inference_ms":1627.8}
```

Every response carries `X-Request-ID`; a client's `X-Request-ID` is honoured if
sent, so a trace survives across your proxy. `JSON_LOGS=0` gives human-readable
output for local work.

Worth alerting on: `/health/ready` returning 503 after startup, a rising rate of
`status: 500`, `latency_ms` on `/predict` drifting up (CPU contention), and
`rate limit exceeded` warnings (either an attack or an under-sized limit).

---

## 9. CI

`.github/workflows/ci.yml` runs three jobs: backend (ruff, the pipeline smoke
test, pytest), frontend (`tsc -b`, `vite build`), and docker (build the image,
boot it, assert security headers, SPA fallback, cache-control, a prediction
round-trip, and non-root).

It assumes `als-webapp/` sits inside the repository. If `als-webapp` **is** your
repository root, drop the `als-webapp/` prefixes from `working-directory` and set
the Docker build `context: .`.

CI runs entirely in `DEMO_MODE` with no weights. It verifies that the service
behaves — never that the model is accurate. Accuracy is a property of the
artifacts and cannot be tested here.

---

## 10. Upgrading

```bash
git pull
docker compose up --build -d      # weights are mounted, so they are untouched
curl -s localhost:8000/health | python3 -m json.tool
```

Rollback is `docker compose down && git checkout <previous> && docker compose up --build -d`.
Nothing in the app migrates state, because there is no state to migrate: results
expire on a TTL and models are read-only inputs.

Two pins to respect when upgrading dependencies:

- **`transformers==4.49.0`** — Florence-2's `trust_remote_code` path breaks on
  newer releases. This is the first thing to check if the VLM path stops working.
- **The CNN input length (23,437)** is baked into the trained weights. Changing
  `SIGNAL_LENGTH` without retraining does not adapt the model; it just feeds it
  the wrong shape.
