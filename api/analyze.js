import formidable from "formidable";
import { readFile } from "node:fs/promises";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

// Vercel: let formidable read the multipart body, not the default parser
export const config = { api: { bodyParser: false } };

const GEMINI_MODEL = "gemini-2.0-flash";
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

const PROMPT = `You are a cable-tracing vision system. The image shows a tangled pair of
earbuds/EarPods with a thin cable. Trace the single main cable from one end to the other.

Return STRICT JSON only, no prose, with this exact shape:
{
  "polyline": [[x, y], ...],           // ordered points along the cable, 12-60 points, each normalized 0..1 (x=left→right, y=top→bottom)
  "crossings": [                       // points where the cable visually crosses over itself
    { "x": 0..1, "y": 0..1, "overStrandIndex": int, "underStrandIndex": int, "confidence": 0..1 }
  ],
  "endpoints": [[x, y], [x, y]]        // the two visible ends of the cable, normalized 0..1
}
If you cannot find a cable, return {"polyline": [], "crossings": [], "endpoints": []}.`;

// --- Origin allowlist -------------------------------------------------------
// The Framer site is always allowed. Add the production custom domain (if any)
// via the ALLOWED_ORIGINS env var (comma-separated) without touching code.
const ALLOWED_ORIGINS = [
  "https://gray-card-413823.framer.app",
  ...(process.env.ALLOWED_ORIGINS
    ? process.env.ALLOWED_ORIGINS.split(",").map((o) => o.trim()).filter(Boolean)
    : []),
];

// --- Rate limiting (Upstash Redis, sliding window) --------------------------
// Two independent limits per caller IP; a request is rejected if EITHER trips.
// Lazily built so the function still boots if the Upstash env vars aren't set
// yet (rate limiting simply stays inactive until they exist).
const SHORT_LIMIT = 10; // requests / 60s
const DAILY_LIMIT = 50; // requests / 1 day

let _limiters = null;
function getLimiters() {
  if (_limiters) return _limiters;
  // The Vercel<>Upstash Marketplace integration provisions KV_REST_API_* names;
  // a manual Upstash setup uses UPSTASH_REDIS_REST_*. Accept either.
  const url = process.env.UPSTASH_REDIS_REST_URL || process.env.KV_REST_API_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN || process.env.KV_REST_API_TOKEN;
  if (!url || !token) {
    return null;
  }
  const redis = new Redis({ url, token });
  _limiters = {
    short: new Ratelimit({
      redis,
      limiter: Ratelimit.slidingWindow(SHORT_LIMIT, "60 s"),
      prefix: "rl:short",
      analytics: false,
    }),
    daily: new Ratelimit({
      redis,
      limiter: Ratelimit.slidingWindow(DAILY_LIMIT, "1 d"),
      prefix: "rl:daily",
      analytics: false,
    }),
  };
  return _limiters;
}

function getClientIp(req) {
  const xff = req.headers["x-forwarded-for"];
  const value = Array.isArray(xff) ? xff[0] : xff || "";
  const first = value.split(",")[0].trim();
  return first || "anonymous";
}

export default async function handler(req, res) {
  const origin = req.headers.origin || "";
  const originAllowed = ALLOWED_ORIGINS.includes(origin);

  // CORS headers — only ever echo an allowed origin back.
  if (originAllowed) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  }

  // CORS preflight: allowed origins get a clean 204, everyone else 403.
  if (req.method === "OPTIONS") {
    return res.status(originAllowed ? 204 : 403).end();
  }

  // 1) Cheap origin allowlist check — runs before anything expensive.
  if (!originAllowed) {
    return res.status(403).json({ error: "Forbidden" });
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  // 2) Rate limit BEFORE any Gemini call so blocked requests cost nothing.
  const ip = getClientIp(req);
  const limiters = getLimiters();
  if (limiters) {
    try {
      // Short window first; if it trips we don't consume the daily budget.
      const short = await limiters.short.limit(ip);
      if (!short.success) {
        res.setHeader("X-RateLimit-Limit", SHORT_LIMIT);
        res.setHeader("X-RateLimit-Remaining", Math.max(0, short.remaining));
        return res.status(429).json({ error: "Too many requests, slow down." });
      }

      const daily = await limiters.daily.limit(ip);
      if (!daily.success) {
        res.setHeader("X-RateLimit-Limit", DAILY_LIMIT);
        res.setHeader("X-RateLimit-Remaining", Math.max(0, daily.remaining));
        return res.status(429).json({ error: "Too many requests, slow down." });
      }

      // Surface the more constrained of the two windows on successful requests.
      if (short.remaining <= daily.remaining) {
        res.setHeader("X-RateLimit-Limit", SHORT_LIMIT);
        res.setHeader("X-RateLimit-Remaining", Math.max(0, short.remaining));
      } else {
        res.setHeader("X-RateLimit-Limit", DAILY_LIMIT);
        res.setHeader("X-RateLimit-Remaining", Math.max(0, daily.remaining));
      }
    } catch (err) {
      // Fail open: a Redis hiccup should not take the endpoint down. Log and continue.
      console.error("Rate limiter error, allowing request:", err?.message || err);
    }
  } else {
    console.warn("Upstash env vars missing; rate limiting is inactive.");
  }

  try {
    const form = formidable({ maxFileSize: 15 * 1024 * 1024 });
    const [, files] = await form.parse(req);
    const file = Array.isArray(files.image) ? files.image[0] : files.image;
    if (!file) return res.status(400).json({ error: "Missing 'image' field" });

    const bytes = await readFile(file.filepath);
    const base64 = bytes.toString("base64");
    const mimeType = file.mimetype || "image/jpeg";

    const geminiRes = await fetch(`${GEMINI_URL}?key=${process.env.GEMINI_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{
          parts: [
            { text: PROMPT },
            { inline_data: { mime_type: mimeType, data: base64 } }
          ]
        }],
        generationConfig: { responseMimeType: "application/json", temperature: 0.1 }
      })
    });

    if (!geminiRes.ok) {
      const detail = await geminiRes.text();
      return res.status(502).json({ error: "Vision model error", detail });
    }

    const data = await geminiRes.json();
    const raw = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "{}";

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return res.status(502).json({ error: "Model returned non-JSON", raw });
    }

    // Normalize / clamp defensively so the uploader always gets a clean contract
    const clamp = (n) => Math.max(0, Math.min(1, Number(n) || 0));

    const polyline = Array.isArray(parsed.polyline)
      ? parsed.polyline.filter(p => Array.isArray(p) && p.length >= 2).map(p => [clamp(p[0]), clamp(p[1])])
      : [];

    const crossings = Array.isArray(parsed.crossings)
      ? parsed.crossings.map(c => ({
          x: clamp(c.x), y: clamp(c.y),
          overStrandIndex: Number.isFinite(c.overStrandIndex) ? c.overStrandIndex : 0,
          underStrandIndex: Number.isFinite(c.underStrandIndex) ? c.underStrandIndex : 0,
          confidence: clamp(c.confidence ?? 0.5)
        }))
      : [];

    const endpoints = Array.isArray(parsed.endpoints)
      ? parsed.endpoints.filter(p => Array.isArray(p) && p.length >= 2).map(p => [clamp(p[0]), clamp(p[1])])
      : [];

    return res.status(200).json({ polyline, crossings, endpoints });
  } catch (err) {
    return res.status(500).json({ error: "Server error", detail: String(err?.message || err) });
  }
}
