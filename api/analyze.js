import formidable from "formidable";
import { readFile } from "node:fs/promises";

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

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  // Optional shared secret: set SHARED_SECRET in Vercel and the same value as the
  // uploader's API key to keep the endpoint private.
  if (process.env.SHARED_SECRET) {
    const auth = req.headers.authorization || "";
    if (auth !== `Bearer ${process.env.SHARED_SECRET}`) {
      return res.status(401).json({ error: "Unauthorized" });
    }
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
