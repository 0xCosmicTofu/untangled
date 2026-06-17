import {
    startTransition,
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type ChangeEvent,
    type CSSProperties,
    type DragEvent,
    type MouseEvent,
} from "react"
import { addPropertyControls, ControlType } from "framer"

// Update the existing code component file `EarPodsUploader.tsx` (default export `EarPodsUploader`) to add a real, automatic cable-topology analyzer that runs right after an image is selected. Keep all existing upload behavior, props, styling, and object-URL handling intact.
// EXPECTED VISION API CONTRACT:
// POST multipart/form-data to `visionApiUrl` with:
//   - image: File (field name exactly "image")
//   - Authorization: Bearer <visionApiKey> header if key is provided
// Response JSON consumed by this component:
// {
//   polyline: [[x,y], ...], // ordered cable centerline in normalized image coords [0..1]
//   crossings?: [{ x, y, overStrandIndex, underStrandIndex, confidence }],
//   endpoints?: [[x,y],[x,y]]
// }
// Notes:
// - `crossings` and `endpoints` are optional and parsed defensively.
// - Low-confidence crossings (< confidenceThreshold) are treated as uncertain and excluded from invariant math.
// - Invariants are marked "best estimate" whenever uncertain crossings exist.

type Point = [number, number]

interface VisionCrossing {
    x: number
    y: number
    overStrandIndex?: number
    underStrandIndex?: number
    confidence?: number
}

interface VisionApiResponse {
    polyline?: unknown
    crossings?: unknown
    endpoints?: unknown
}

interface CrossingEvent {
    id: number
    sign: 1 | -1
    confidence: number
    uncertain: boolean
    positions: [number, number]
    coord: Point
}

interface DiagramEstimate {
    gaussCode: string
    crossingNumber: number
    writhe: number
    uncertainCrossings: number
    jonesPolynomial: string
    bestEstimate: boolean
    knotName: string
    knotAdvice: string
    knotIdentified: boolean
}

interface UnknotPlanStep {
    marker: number
    coord: { x: number; y: number }
    text: string
    uncertain: boolean
}

interface UnknotPlan {
    summary: string
    steps: UnknotPlanStep[]
}

interface AnalysisPayload {
    diagram: DiagramEstimate
    plan: UnknotPlan
    crossingCount: number
}

type AnalysisState = "idle" | "loading" | "success" | "fallback"

function asPoint(value: unknown): Point | null {
    if (!Array.isArray(value) || value.length < 2) return null
    const x = Number(value[0])
    const y = Number(value[1])
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null
    return [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))]
}

function parsePolyline(value: unknown): Point[] {
    if (!Array.isArray(value)) return []
    return value.map(asPoint).filter((p): p is Point => p !== null)
}

function parseCrossings(value: unknown): VisionCrossing[] {
    if (!Array.isArray(value)) return []
    return value
        .map((item) => {
            if (!item || typeof item !== "object") return null
            const c = item as Record<string, unknown>
            const x = Number(c.x)
            const y = Number(c.y)
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null
            const over = Number(c.overStrandIndex)
            const under = Number(c.underStrandIndex)
            const confidence = Number(c.confidence)
            const parsed: VisionCrossing = {
                x: Math.max(0, Math.min(1, x)),
                y: Math.max(0, Math.min(1, y)),
                overStrandIndex: Number.isFinite(over) ? over : undefined,
                underStrandIndex: Number.isFinite(under) ? under : undefined,
                confidence: Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence)) : undefined,
            }
            return parsed
        })
        .filter((c): c is VisionCrossing => c !== null)
}

function parseEndpoints(value: unknown): Point[] {
    if (!Array.isArray(value)) return []
    return value.map(asPoint).filter((p): p is Point => p !== null).slice(0, 2)
}

function distance(a: Point, b: Point): number {
    const dx = a[0] - b[0]
    const dy = a[1] - b[1]
    return Math.sqrt(dx * dx + dy * dy)
}

function polylineLength(polyline: Point[]): number {
    let total = 0
    for (let i = 0; i < polyline.length - 1; i++) total += distance(polyline[i], polyline[i + 1])
    return total
}

function tangentAt(polyline: Point[], index: number): Point {
    const i0 = Math.max(0, index - 1)
    const i1 = Math.min(polyline.length - 1, index + 1)
    const dx = polyline[i1][0] - polyline[i0][0]
    const dy = polyline[i1][1] - polyline[i0][1]
    const n = Math.sqrt(dx * dx + dy * dy) || 1
    return [dx / n, dy / n]
}

function nearestTwoPolylineIndices(polyline: Point[], target: Point): [number, number] {
    const ranked = polyline
        .map((p, index) => ({ index, d: distance(p, target) }))
        .sort((a, b) => a.d - b.d)
    const first = ranked[0]?.index ?? 0
    let second = ranked[1]?.index ?? Math.min(polyline.length - 1, first + 2)
    for (let i = 1; i < ranked.length; i++) {
        if (Math.abs(ranked[i].index - first) > 2) {
            second = ranked[i].index
            break
        }
    }
    if (second === first) second = Math.min(polyline.length - 1, first + 1)
    return first < second ? [first, second] : [second, first]
}

function buildCrossingEvents(
    polyline: Point[],
    crossings: VisionCrossing[],
    confidenceThreshold: number
): CrossingEvent[] {
    return crossings
        .map((crossing, id) => {
            const pos = nearestTwoPolylineIndices(polyline, [crossing.x, crossing.y])
            const overIndex = Number.isFinite(crossing.overStrandIndex) ? Number(crossing.overStrandIndex) : pos[0]
            const underIndex = Number.isFinite(crossing.underStrandIndex) ? Number(crossing.underStrandIndex) : pos[1]
            const overTan = tangentAt(polyline, Math.max(0, Math.min(polyline.length - 1, overIndex)))
            const underTan = tangentAt(polyline, Math.max(0, Math.min(polyline.length - 1, underIndex)))
            const crossZ = overTan[0] * underTan[1] - overTan[1] * underTan[0]
            const sign: 1 | -1 = crossZ >= 0 ? 1 : -1
            const confidence = crossing.confidence ?? 0.5
            const coord: Point = [crossing.x, crossing.y]
            return {
                id,
                sign,
                confidence,
                uncertain: confidence < confidenceThreshold,
                positions: pos,
                coord,
            }
        })
        .filter((e) => Number.isFinite(e.positions[0]) && Number.isFinite(e.positions[1]))
}

function reduceCrossings(events: CrossingEvent[]): CrossingEvent[] {
    const withoutR1 = events.filter((e) => Math.abs(e.positions[1] - e.positions[0]) > 2)
    const consumed = new Set<number>()
    for (let i = 0; i < withoutR1.length; i++) {
        if (consumed.has(i)) continue
        for (let j = i + 1; j < withoutR1.length; j++) {
            if (consumed.has(j)) continue
            const a = withoutR1[i]
            const b = withoutR1[j]
            const [a1, a2] = a.positions
            const [b1, b2] = b.positions
            const interleave = (a1 < b1 && b1 < a2 && a2 < b2) || (b1 < a1 && a1 < b2 && b2 < a2)
            if (interleave && a.sign !== b.sign && !a.uncertain && !b.uncertain) {
                consumed.add(i)
                consumed.add(j)
                break
            }
        }
    }
    return withoutR1.filter((_, idx) => !consumed.has(idx))
}

function buildGaussCode(events: CrossingEvent[]): string {
    const sorted = [...events].sort((a, b) => a.positions[0] - b.positions[0])
    const labels = new Map<number, number>()
    sorted.forEach((e, idx) => labels.set(e.id, idx + 1))
    const tokens: { pos: number; token: string }[] = []
    sorted.forEach((e) => {
        const label = labels.get(e.id) ?? e.id + 1
        const suffix = e.sign > 0 ? "+" : "-"
        tokens.push({ pos: e.positions[0], token: `${label}${suffix}` })
        tokens.push({ pos: e.positions[1], token: `${label}${suffix}` })
    })
    return tokens
        .sort((a, b) => a.pos - b.pos)
        .map((t) => t.token)
        .join(" ")
}

function addPoly(a: Map<number, number>, b: Map<number, number>): Map<number, number> {
    const out = new Map<number, number>(a)
    for (const [exp, coeff] of b.entries()) out.set(exp, (out.get(exp) ?? 0) + coeff)
    for (const [exp, coeff] of out.entries()) if (Math.abs(coeff) < 1e-9) out.delete(exp)
    return out
}

function mulPoly(a: Map<number, number>, b: Map<number, number>): Map<number, number> {
    const out = new Map<number, number>()
    for (const [ea, ca] of a.entries()) {
        for (const [eb, cb] of b.entries()) {
            const exp = ea + eb
            out.set(exp, (out.get(exp) ?? 0) + ca * cb)
        }
    }
    for (const [exp, coeff] of out.entries()) if (Math.abs(coeff) < 1e-9) out.delete(exp)
    return out
}

function powPoly(base: Map<number, number>, pow: number): Map<number, number> {
    let out = new Map<number, number>([[0, 1]])
    for (let i = 0; i < pow; i++) out = mulPoly(out, base)
    return out
}

function countLoopsForState(events: CrossingEvent[], stateMask: number): number {
    const n = events.length
    if (n === 0) return 1
    const m = n * 2
    const nodeCount = m * 2
    const parent = Array.from({ length: nodeCount }, (_, i) => i)
    const find = (x: number): number => {
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]]
            x = parent[x]
        }
        return x
    }
    const union = (a: number, b: number): void => {
        const ra = find(a)
        const rb = find(b)
        if (ra !== rb) parent[ra] = rb
    }
    const inNode = (pos: number) => pos * 2
    const outNode = (pos: number) => pos * 2 + 1
    const ordered = [...events].sort((a, b) => a.positions[0] - b.positions[0])
    const firstPosMap = new Map<number, number>()
    const secondPosMap = new Map<number, number>()
    ordered.forEach((ev, idx) => {
        firstPosMap.set(ev.id, idx * 2)
        secondPosMap.set(ev.id, idx * 2 + 1)
    })
    for (let pos = 0; pos < m; pos++) union(outNode(pos), inNode((pos + 1) % m))
    ordered.forEach((ev, idx) => {
        const p = firstPosMap.get(ev.id) ?? idx * 2
        const q = secondPosMap.get(ev.id) ?? idx * 2 + 1
        const isA = ((stateMask >> idx) & 1) === 0
        if (isA) {
            union(inNode(p), inNode(q))
            union(outNode(p), outNode(q))
        } else {
            union(inNode(p), outNode(q))
            union(outNode(p), inNode(q))
        }
    })
    const roots = new Set<number>()
    for (let i = 0; i < nodeCount; i++) roots.add(find(i))
    return roots.size
}

function normalizeByWrithe(bracket: Map<number, number>, writhe: number): Map<number, number> {
    const shift = -3 * writhe
    const sign = Math.abs(shift) % 2 === 0 ? 1 : -1
    const out = new Map<number, number>()
    for (const [exp, coeff] of bracket.entries()) out.set(exp + shift, coeff * sign)
    return out
}

function polyToString(poly: Map<number, number>, variable = "A"): string {
    if (poly.size === 0) return "0"
    const parts = [...poly.entries()]
        .sort((a, b) => b[0] - a[0])
        .map(([exp, coeff], idx) => {
            const sign = coeff < 0 ? "-" : idx > 0 ? "+" : ""
            const abs = Math.abs(coeff)
            const coeffPart = abs === 1 && exp !== 0 ? "" : `${Math.round(abs * 1000) / 1000}`
            if (exp === 0) return `${sign}${abs}`
            if (exp === 1) return `${sign}${coeffPart}${variable}`
            return `${sign}${coeffPart}${variable}^${exp}`
        })
    return parts.join(" ").trim()
}

interface JonesBracket {
    status: "ok" | "insufficient" | "toolarge"
    poly: Map<number, number>
}

function computeJonesBracketA(events: CrossingEvent[]): JonesBracket {
    const certain = events.filter((e) => !e.uncertain)
    const crossingCount = certain.length
    // Zero confident crossings => treat as the unknot, whose normalized bracket is 1.
    if (crossingCount === 0) return { status: "ok", poly: new Map<number, number>([[0, 1]]) }
    const maxExactCrossings = 12
    if (crossingCount > maxExactCrossings) return { status: "toolarge", poly: new Map<number, number>() }
    const d = new Map<number, number>([
        [2, -1],
        [-2, -1],
    ])
    let bracket = new Map<number, number>()
    const states = 1 << crossingCount
    for (let mask = 0; mask < states; mask++) {
        const aCount = crossingCount - mask.toString(2).split("1").length + 1
        const bCount = crossingCount - aCount
        const loops = countLoopsForState(certain, mask)
        const smoothing = new Map<number, number>([[aCount - bCount, 1]])
        const loopFactor = powPoly(d, Math.max(0, loops - 1))
        const term = mulPoly(smoothing, loopFactor)
        bracket = addPoly(bracket, term)
    }
    const writhe = certain.reduce((sum, c) => sum + c.sign, 0)
    const normalized = normalizeByWrithe(bracket, writhe)
    return { status: "ok", poly: normalized }
}

function jonesBracketToString(bracket: JonesBracket): string {
    if (bracket.status === "insufficient") return "Insufficient confident crossings"
    if (bracket.status === "toolarge") return "Estimate omitted (too many crossings for exact computation)"
    return polyToString(bracket.poly, "A")
}

// Convert the normalized Kauffman bracket (in A) to the Jones polynomial in t,
// using the standard substitution t = A^-4. Returns null if the polynomial is
// not a clean knot Jones polynomial (exponents not divisible by 4).
function jonesAToT(aPoly: Map<number, number>): Map<number, number> | null {
    const out = new Map<number, number>()
    for (const [exp, coeff] of aPoly.entries()) {
        if (exp % 4 !== 0) return null
        out.set(-exp / 4, coeff)
    }
    return out
}

// --- Knot identification ----------------------------------------------------
// Adapted from the invariant-lookup approach of pyknotid (MIT License),
// https://github.com/SPOCKnots/pyknotid . The Jones polynomials below (in t)
// are the standard textbook values for the unknot, trefoil (3_1) and
// figure-eight (4_1) — the cases that dominate real tangled-earbud knots.
interface KnotEntry {
    name: string
    jones: Record<string, number>
    advice: string
}

const KNOT_TABLE: KnotEntry[] = [
    {
        name: "unknot",
        jones: { "0": 1 },
        advice:
            "Good news — it isn’t truly knotted, just looped over itself. Hold both earbuds and gently pull them apart; the cable should fall open.",
    },
    {
        name: "trefoil (3₁)",
        jones: { "-1": 1, "-3": 1, "-4": -1 },
        advice:
            "A single overhand knot. Find the one loop an earbud passed through, then feed that same earbud back out through the loop once.",
    },
    {
        name: "figure-eight (4₁)",
        jones: { "-2": 1, "-1": -1, "0": 1, "1": -1, "2": 1 },
        advice:
            "A figure-eight knot. Loosen the whole knot first, locate the doubled-back strand, then pull the nearest earbud back along the path it came in.",
    },
]

function polyEqual(a: Map<number, number>, b: Map<number, number>): boolean {
    if (a.size !== b.size) return false
    for (const [exp, coeff] of a.entries()) if (b.get(exp) !== coeff) return false
    return true
}

function mirrorPoly(poly: Map<number, number>): Map<number, number> {
    const out = new Map<number, number>()
    for (const [exp, coeff] of poly.entries()) out.set(-exp, coeff)
    return out
}

function recordToPoly(record: Record<string, number>): Map<number, number> {
    const out = new Map<number, number>()
    for (const [key, value] of Object.entries(record)) out.set(Number(key), value)
    return out
}

interface KnotIdentity {
    name: string
    advice: string
    exact: boolean
}

function identifyKnot(jonesT: Map<number, number> | null, crossingNumber: number): KnotIdentity {
    // 1) Exact invariant match (also matches each knot's mirror image).
    if (jonesT) {
        for (const entry of KNOT_TABLE) {
            const target = recordToPoly(entry.jones)
            if (polyEqual(jonesT, target) || polyEqual(jonesT, mirrorPoly(target))) {
                return { name: entry.name, advice: entry.advice, exact: true }
            }
        }
    }
    // 2) Fall back to a best guess from the reduced crossing number.
    if (crossingNumber === 0) return { name: "unknot", advice: KNOT_TABLE[0].advice, exact: false }
    if (crossingNumber === 3) return { name: "trefoil (3₁)?", advice: KNOT_TABLE[1].advice, exact: false }
    if (crossingNumber === 4) return { name: "figure-eight (4₁)?", advice: KNOT_TABLE[2].advice, exact: false }
    return {
        name: `${crossingNumber}-crossing tangle`,
        advice:
            "Start with the loosest crossing and work the nearest earbud back out through it, one crossing at a time.",
        exact: false,
    }
}

function estimatePlan(
    polyline: Point[],
    endpoints: Point[],
    events: CrossingEvent[],
    knot: KnotIdentity
): UnknotPlan {
    const start = endpoints[0] ?? polyline[0]
    const end = endpoints[1] ?? polyline[polyline.length - 1]
    const sortByEase = [...events]
        .sort((a, b) => {
            const da = Math.min(distance(a.coord, start), distance(a.coord, end)) + (1 - a.confidence) * 0.1
            const db = Math.min(distance(b.coord, start), distance(b.coord, end)) + (1 - b.confidence) * 0.1
            return da - db
        })
        .slice(0, 5)
    const sideOf = (c: Point): string => {
        if (c[0] > 0.66) return "right"
        if (c[0] < 0.33) return "left"
        return "middle"
    }
    const verticalOf = (c: Point): string => {
        if (c[1] > 0.66) return "lower"
        if (c[1] < 0.33) return "upper"
        return "central"
    }
    const steps: UnknotPlanStep[] = sortByEase.map((c, i) => {
        const marker = i + 1
        const location = `${verticalOf(c.coord)}-${sideOf(c.coord)}`
        const uncertainty = c.uncertain ? " (less certain crossing — move gently)" : ""
        return {
            marker,
            coord: { x: c.coord[0], y: c.coord[1] },
            uncertain: c.uncertain,
            text: `Crossing ${marker} (${location}): ease the strand backward through it.${uncertainty}`,
        }
    })
    return {
        summary: knot.exact
            ? `Identified: ${knot.name}`
            : `Looks like a ${events.length}-crossing tangle`,
        steps,
    }
}

function analyzeTopology(
    polyline: Point[],
    crossings: VisionCrossing[],
    endpoints: Point[],
    confidenceThreshold: number
): AnalysisPayload {
    const events = buildCrossingEvents(polyline, crossings, confidenceThreshold)
    const reduced = reduceCrossings(events.filter((e) => !e.uncertain))
    const uncertainCrossings = events.filter((e) => e.uncertain).length
    const gaussCode = buildGaussCode(events.filter((e) => !e.uncertain))
    const writhe = reduced.reduce((sum, c) => sum + c.sign, 0)
    const bracket = computeJonesBracketA(events)
    const jones = jonesBracketToString(bracket)
    const jonesT = bracket.status === "ok" ? jonesAToT(bracket.poly) : null
    const knot = identifyKnot(jonesT, reduced.length)
    return {
        diagram: {
            gaussCode: gaussCode || "No confident crossings available",
            crossingNumber: reduced.length,
            writhe,
            uncertainCrossings,
            jonesPolynomial: jones,
            bestEstimate: uncertainCrossings > 0,
            knotName: knot.name,
            knotAdvice: knot.advice,
            knotIdentified: knot.exact,
        },
        plan: estimatePlan(polyline, endpoints, events, knot),
        crossingCount: events.length,
    }
}

interface MyComponentProps {
    accentColor: string
    surfaceColor: string
    borderColor: string
    textColor: string
    mutedColor: string
    cornerRadius: number
    promptTitle: string
    promptSubtitle: string
    successText: string
    enableAnalysis: boolean
    visionApiUrl: string
    visionApiKey: string
    confidenceThreshold: number
    analysisTimeoutSeconds: number
    enableCamera: boolean
    cameraButtonLabel: string
    uploadButtonLabel: string
    fallbackNoteText: string
}

/**
 * @framerSupportedLayoutWidth any-prefer-fixed
 * @framerSupportedLayoutHeight auto
 */
export default function EarPodsUploader(props: MyComponentProps) {
    const {
        accentColor,
        surfaceColor,
        borderColor,
        textColor,
        mutedColor,
        cornerRadius,
        promptTitle,
        promptSubtitle,
        successText,
        enableAnalysis,
        visionApiUrl,
        visionApiKey,
        confidenceThreshold,
        analysisTimeoutSeconds,
        enableCamera,
        cameraButtonLabel,
        uploadButtonLabel,
        fallbackNoteText,
    } = props

    const [previewUrl, setPreviewUrl] = useState<string | null>(null)
    const [isDragging, setIsDragging] = useState(false)
    const [analysisState, setAnalysisState] = useState<AnalysisState>("idle")
    const [analysisError, setAnalysisError] = useState("")
    const [analysisResult, setAnalysisResult] = useState<AnalysisPayload | null>(null)
    const [fallbackNotice, setFallbackNotice] = useState("")
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [showDetails, setShowDetails] = useState(false)
    const [hoveredMarker, setHoveredMarker] = useState<number | null>(null)
    const [cameraMode, setCameraMode] = useState<"idle" | "live" | "error">("idle")
    const [cameraError, setCameraError] = useState("")
    const fileInputRef = useRef<HTMLInputElement | null>(null)
    const cameraInputRef = useRef<HTMLInputElement | null>(null)
    const videoRef = useRef<HTMLVideoElement | null>(null)
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const streamRef = useRef<MediaStream | null>(null)
    const requestIdRef = useRef(0)

    const stopCameraStream = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop())
            streamRef.current = null
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null
        }
    }, [])

    useEffect(() => {
        return () => {
            if (previewUrl && typeof window !== "undefined") {
                URL.revokeObjectURL(previewUrl)
            }
            stopCameraStream()
        }
    }, [previewUrl, stopCameraStream])

    useEffect(() => {
        if (cameraMode === "live" && videoRef.current && streamRef.current) {
            videoRef.current.srcObject = streamRef.current
            void videoRef.current.play().catch(() => {})
        }
    }, [cameraMode])

    const runAnalysis = useCallback(
        async (file: File) => {
            if (!enableAnalysis) {
                startTransition(() => {
                    setAnalysisState("idle")
                    setAnalysisResult(null)
                    setAnalysisError("")
                    setFallbackNotice("")
                })
                return
            }
            if (!visionApiUrl.trim()) {
                startTransition(() => {
                    setAnalysisState("fallback")
                    setAnalysisResult(null)
                    setAnalysisError("")
                    setFallbackNotice(
                        "Automatic analysis is not connected right now. You can still follow the general untangling steps below."
                    )
                })
                return
            }
            const currentRequest = ++requestIdRef.current
            startTransition(() => {
                setAnalysisState("loading")
                setAnalysisResult(null)
                setAnalysisError("")
                setFallbackNotice("")
            })
            const timeoutMs = Math.max(5, analysisTimeoutSeconds) * 1000
            const maxAttempts = 2
            for (let attempt = 0; attempt < maxAttempts; attempt++) {
                let timeoutId = 0
                const abortController = new AbortController()
                try {
                    timeoutId = window.setTimeout(() => {
                        abortController.abort()
                    }, timeoutMs)
                    const formData = new FormData()
                    formData.append("image", file)
                    const headers: Record<string, string> = {}
                    if (visionApiKey.trim()) headers.Authorization = `Bearer ${visionApiKey.trim()}`
                    const response = await fetch(visionApiUrl, {
                        method: "POST",
                        body: formData,
                        headers,
                        signal: abortController.signal,
                    })
                    if (!response.ok) {
                        throw new Error(`Vision API error (${response.status})`)
                    }
                    const json = (await response.json()) as VisionApiResponse
                    const polyline = parsePolyline(json.polyline)
                    const crossings = parseCrossings(json.crossings)
                    const endpoints = parseEndpoints(json.endpoints)
                    if (requestIdRef.current !== currentRequest) return
                    if (polyline.length < 6 || polylineLength(polyline) < 0.15) {
                        startTransition(() => {
                            setAnalysisState("fallback")
                            setAnalysisResult(null)
                            setAnalysisError("")
                            setFallbackNotice(
                                "I couldn’t get a confident cable trace this time. Follow the general steps below, or try a clearer, well-lit photo on a plain background."
                            )
                        })
                        return
                    }
                    const result = analyzeTopology(polyline, crossings, endpoints, confidenceThreshold)
                    startTransition(() => {
                        setAnalysisResult(result)
                        setAnalysisState("success")
                        setFallbackNotice("")
                        setAnalysisError("")
                    })
                    return
                } catch (error) {
                    if (requestIdRef.current !== currentRequest) return
                    const isTimeout = error instanceof Error && error.name === "AbortError"
                    if (isTimeout && attempt < maxAttempts - 1) {
                        continue
                    }
                    startTransition(() => {
                        setAnalysisState("fallback")
                        setAnalysisResult(null)
                        setAnalysisError("")
                        if (isTimeout) {
                            setFallbackNotice(
                                "The analysis service took too long to respond. This is usually a temporary slowdown — please try again."
                            )
                        } else {
                            setFallbackNotice(
                                "Something went wrong reaching the analysis service. Please try again."
                            )
                        }
                    })
                    return
                } finally {
                    if (timeoutId) clearTimeout(timeoutId)
                }
            }
        },
        [analysisTimeoutSeconds, confidenceThreshold, enableAnalysis, visionApiKey, visionApiUrl]
    )

    const applyFile = useCallback(
        (file: File | null) => {
            if (!file || !file.type.startsWith("image/")) return
            if (typeof window === "undefined") return
            const nextUrl = URL.createObjectURL(file)
            startTransition(() => {
                setPreviewUrl((prev) => {
                    if (prev) URL.revokeObjectURL(prev)
                    return nextUrl
                })
                setSelectedFile(file)
                setShowDetails(false)
            })
            void runAnalysis(file)
        },
        [runAnalysis]
    )

    const onFileInputChange = useCallback(
        (event: ChangeEvent<HTMLInputElement>) => {
            const file = event.target.files?.[0] ?? null
            applyFile(file)
        },
        [applyFile]
    )

    const onCameraFileInputChange = useCallback(
        (event: ChangeEvent<HTMLInputElement>) => {
            const file = event.target.files?.[0] ?? null
            applyFile(file)
            if (cameraInputRef.current) cameraInputRef.current.value = ""
        },
        [applyFile]
    )

    const onDragOver = useCallback((event: DragEvent<HTMLLabelElement>) => {
        event.preventDefault()
        event.stopPropagation()
        startTransition(() => setIsDragging(true))
    }, [])

    const onDragLeave = useCallback((event: DragEvent<HTMLLabelElement>) => {
        event.preventDefault()
        event.stopPropagation()
        startTransition(() => setIsDragging(false))
    }, [])

    const onDrop = useCallback(
        (event: DragEvent<HTMLLabelElement>) => {
            event.preventDefault()
            event.stopPropagation()
            startTransition(() => setIsDragging(false))
            const file = event.dataTransfer?.files?.[0] ?? null
            applyFile(file)
        },
        [applyFile]
    )

    const resetSelection = useCallback(() => {
        stopCameraStream()
        startTransition(() => {
            setPreviewUrl((prev) => {
                if (prev && typeof window !== "undefined") {
                    URL.revokeObjectURL(prev)
                }
                return null
            })
            setSelectedFile(null)
            setAnalysisState("idle")
            setAnalysisResult(null)
            setAnalysisError("")
            setFallbackNotice("")
            setShowDetails(false)
            setHoveredMarker(null)
            setCameraMode("idle")
            setCameraError("")
        })
        if (fileInputRef.current) {
            fileInputRef.current.value = ""
        }
        if (cameraInputRef.current) {
            cameraInputRef.current.value = ""
        }
    }, [stopCameraStream])

    const retryAnalysis = useCallback(() => {
        if (selectedFile) void runAnalysis(selectedFile)
    }, [runAnalysis, selectedFile])

    const isLikelyMobile = useCallback((): boolean => {
        if (typeof window === "undefined" || typeof navigator === "undefined") return false
        const ua = navigator.userAgent || ""
        const coarse = typeof window.matchMedia !== "undefined" && window.matchMedia("(pointer: coarse)").matches
        return /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || coarse
    }, [])

    const openLiveCamera = useCallback(async () => {
        if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
            startTransition(() => {
                setCameraMode("error")
                setCameraError("Camera preview is not supported in this browser.")
            })
            return
        }
        try {
            stopCameraStream()
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment" },
                audio: false,
            })
            streamRef.current = stream
            startTransition(() => {
                setCameraMode("live")
                setCameraError("")
            })
        } catch (error) {
            const message =
                error instanceof Error
                    ? error.message
                    : "Couldn’t access camera. Check permissions and try again."
            startTransition(() => {
                setCameraMode("error")
                setCameraError(message)
            })
        }
    }, [stopCameraStream])

    const onUseCamera = useCallback(
        (event: MouseEvent<HTMLButtonElement>) => {
            event.preventDefault()
            event.stopPropagation()
            if (isLikelyMobile()) {
                cameraInputRef.current?.click()
                return
            }
            void openLiveCamera()
        },
        [isLikelyMobile, openLiveCamera]
    )

    const onUseUpload = useCallback((event: MouseEvent<HTMLButtonElement>) => {
        event.preventDefault()
        event.stopPropagation()
        fileInputRef.current?.click()
    }, [])

    const cancelCamera = useCallback(() => {
        stopCameraStream()
        startTransition(() => {
            setCameraMode("idle")
        })
    }, [stopCameraStream])

    const captureFromVideo = useCallback(async () => {
        if (typeof window === "undefined") return
        const video = videoRef.current
        const canvas = canvasRef.current
        if (!video || !canvas) return
        const width = video.videoWidth || 1280
        const height = video.videoHeight || 720
        canvas.width = width
        canvas.height = height
        const context = canvas.getContext("2d")
        if (!context) {
            startTransition(() => {
                setCameraMode("error")
                setCameraError("Could not capture frame from camera.")
            })
            return
        }
        context.drawImage(video, 0, 0, width, height)
        const blob = await new Promise<Blob | null>((resolve) => {
            canvas.toBlob((b) => resolve(b), "image/jpeg", 0.92)
        })
        if (!blob) {
            startTransition(() => {
                setCameraMode("error")
                setCameraError("Capture failed. Please try again.")
            })
            return
        }
        const capturedFile = new File([blob], "earpods-camera.jpg", { type: "image/jpeg" })
        stopCameraStream()
        startTransition(() => {
            setCameraMode("idle")
            setCameraError("")
        })
        applyFile(capturedFile)
    }, [applyFile, stopCameraStream])

    const containerStyle = useMemo<CSSProperties>(
        () => ({
            position: "relative",
            width: "100%",
            height: "auto",
            boxSizing: "border-box",
            backgroundColor: surfaceColor,
            border: `1px solid ${borderColor}`,
            borderRadius: cornerRadius,
            padding: 14,
            fontFamily:
                'Inter, -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", system-ui, sans-serif',
            color: textColor,
        }),
        [surfaceColor, borderColor, cornerRadius, textColor]
    )

    const dropZoneStyle = useMemo<CSSProperties>(
        () => ({
            position: "relative",
            width: "100%",
            minHeight: 240,
            borderRadius: Math.max(12, cornerRadius - 6),
            border: `1px dashed ${isDragging ? accentColor : borderColor}`,
            backgroundColor: isDragging ? "rgba(10, 132, 255, 0.06)" : "rgba(0, 0, 0, 0.01)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            overflow: "hidden",
            transition: "border-color 180ms ease, background-color 180ms ease",
            boxSizing: "border-box",
            padding: 18,
        }),
        [cornerRadius, isDragging, accentColor, borderColor]
    )

    const analysisCardStyle = useMemo<CSSProperties>(
        () => ({
            position: "relative",
            marginTop: 12,
            border: `1px solid ${borderColor}`,
            borderRadius: Math.max(12, cornerRadius - 6),
            background: "rgba(0,0,0,0.01)",
            padding: 12,
        }),
        [borderColor, cornerRadius]
    )

    return (
        <div style={containerStyle}>
            <label
                style={dropZoneStyle}
                onDragOver={onDragOver}
                onDragEnter={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                aria-label="Upload EarPods photo"
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={onFileInputChange}
                    style={{ display: "none" }}
                />
                <input
                    ref={cameraInputRef}
                    type="file"
                    accept="image/*"
                    capture="environment"
                    onChange={onCameraFileInputChange}
                    style={{ display: "none" }}
                />
                <canvas ref={canvasRef} style={{ display: "none" }} />

                {previewUrl ? (
                    <img
                        src={previewUrl}
                        alt="Uploaded EarPods"
                        style={{
                            width: "100%",
                            maxHeight: 360,
                            objectFit: "contain",
                            borderRadius: Math.max(10, cornerRadius - 8),
                            display: "block",
                        }}
                    />
                ) : (
                    <div
                        style={{
                            position: "relative",
                            textAlign: "center",
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                            width: "100%",
                            maxWidth: 520,
                        }}
                    >
                        <div
                            style={{
                                fontSize: 18,
                                lineHeight: 1.2,
                                fontWeight: 650,
                                color: textColor,
                                marginBottom: 6,
                            }}
                        >
                            {promptTitle}
                        </div>
                        <div
                            style={{
                                fontSize: 14,
                                lineHeight: 1.35,
                                color: mutedColor,
                            }}
                        >
                            {promptSubtitle}
                        </div>
                        <div
                            style={{
                                marginTop: 12,
                                display: "flex",
                                flexWrap: "wrap",
                                justifyContent: "center",
                                gap: 8,
                            }}
                        >
                            {enableCamera ? (
                                <button
                                    type="button"
                                    onClick={onUseCamera}
                                    style={{
                                        appearance: "none",
                                        border: "none",
                                        borderRadius: 999,
                                        background: accentColor,
                                        color: surfaceColor,
                                        fontSize: 13,
                                        fontWeight: 600,
                                        lineHeight: 1,
                                        padding: "10px 15px",
                                        cursor: "pointer",
                                        display: "inline-flex",
                                        alignItems: "center",
                                        gap: 7,
                                    }}
                                >
                                    <span aria-hidden>📷</span>
                                    <span>{cameraButtonLabel}</span>
                                </button>
                            ) : null}
                            <button
                                type="button"
                                onClick={onUseUpload}
                                style={{
                                    appearance: "none",
                                    border: `1px solid ${borderColor}`,
                                    borderRadius: 999,
                                    background: surfaceColor,
                                    color: textColor,
                                    fontSize: 13,
                                    fontWeight: 600,
                                    lineHeight: 1,
                                    padding: "10px 15px",
                                    cursor: "pointer",
                                }}
                            >
                                {uploadButtonLabel}
                            </button>
                        </div>
                    </div>
                )}
            </label>
            {cameraMode === "live" && !previewUrl ? (
                <div
                    style={{
                        position: "relative",
                        marginTop: 10,
                        border: `1px solid ${borderColor}`,
                        borderRadius: Math.max(12, cornerRadius - 6),
                        background: "rgba(0,0,0,0.01)",
                        overflow: "hidden",
                    }}
                >
                    <video
                        ref={videoRef}
                        autoPlay
                        muted
                        playsInline
                        style={{
                            width: "100%",
                            maxHeight: 320,
                            objectFit: "cover",
                            display: "block",
                            background: "#000000",
                        }}
                    />
                    <div
                        style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            gap: 8,
                            padding: 10,
                        }}
                    >
                        <button
                            type="button"
                            onClick={captureFromVideo}
                            style={{
                                appearance: "none",
                                border: "none",
                                borderRadius: 999,
                                background: accentColor,
                                color: surfaceColor,
                                fontSize: 13,
                                fontWeight: 600,
                                padding: "8px 14px",
                                cursor: "pointer",
                            }}
                        >
                            Capture
                        </button>
                        <button
                            type="button"
                            onClick={cancelCamera}
                            style={{
                                appearance: "none",
                                border: `1px solid ${borderColor}`,
                                borderRadius: 999,
                                background: surfaceColor,
                                color: mutedColor,
                                fontSize: 13,
                                fontWeight: 600,
                                padding: "8px 14px",
                                cursor: "pointer",
                            }}
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            ) : null}
            {cameraMode === "error" && !previewUrl ? (
                <div
                    style={{
                        marginTop: 10,
                        color: mutedColor,
                        fontSize: 13,
                        lineHeight: 1.4,
                    }}
                >
                    Camera isn’t available right now: {cameraError || "Try camera permissions or use file upload."}
                </div>
            ) : null}

            {previewUrl ? (
                <div
                    style={{
                        position: "relative",
                        marginTop: 10,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: 10,
                        flexWrap: "wrap",
                    }}
                >
                    <div
                        style={{
                            fontSize: 14,
                            lineHeight: 1.35,
                            color: mutedColor,
                        }}
                    >
                        {successText}
                    </div>
                    <button
                        type="button"
                        onClick={resetSelection}
                        style={{
                            appearance: "none",
                            border: "none",
                            background: "transparent",
                            color: accentColor,
                            fontSize: 13,
                            fontWeight: 600,
                            cursor: "pointer",
                            padding: 0,
                        }}
                    >
                        Choose another photo
                    </button>
                </div>
            ) : null}
            {previewUrl && enableAnalysis ? (
                <div style={analysisCardStyle}>
                    {analysisState === "loading" ? (
                        <div style={{ color: mutedColor, fontSize: 14 }}>Tracing the cable…</div>
                    ) : null}
                    {analysisState === "fallback" ? (
                        <div
                            style={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "space-between",
                                gap: 10,
                                flexWrap: "wrap",
                            }}
                        >
                            <div style={{ color: mutedColor, fontSize: 14, lineHeight: 1.4 }}>
                                {fallbackNotice || fallbackNoteText}
                                {analysisError ? ` (${analysisError})` : ""}
                            </div>
                            {visionApiUrl.trim() ? (
                                <button
                                    type="button"
                                    onClick={retryAnalysis}
                                    style={{
                                        appearance: "none",
                                        border: `1px solid ${borderColor}`,
                                        borderRadius: 999,
                                        background: surfaceColor,
                                        color: accentColor,
                                        fontSize: 13,
                                        fontWeight: 600,
                                        cursor: "pointer",
                                        padding: "6px 10px",
                                    }}
                                >
                                    Retry analysis
                                </button>
                            ) : null}
                        </div>
                    ) : null}
                    {analysisState === "success" && analysisResult ? (
                        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                            <div style={{ color: textColor, fontSize: 16, fontWeight: 600 }}>
                                {analysisResult.plan.summary}
                            </div>
                            <div
                                style={{
                                    borderRadius: Math.max(8, cornerRadius - 12),
                                    border: `1px solid ${borderColor}`,
                                    background: "rgba(10, 132, 255, 0.04)",
                                    padding: 10,
                                }}
                            >
                                <div style={{ color: textColor, fontSize: 14, fontWeight: 600 }}>
                                    {analysisResult.diagram.knotIdentified
                                        ? "Knot identified"
                                        : "Best guess"}
                                    : {analysisResult.diagram.knotName}
                                </div>
                                <div style={{ color: mutedColor, fontSize: 13, lineHeight: 1.45, marginTop: 4 }}>
                                    {analysisResult.diagram.knotAdvice}
                                </div>
                            </div>
                            {previewUrl && analysisResult.plan.steps.length > 0 ? (
                                <div
                                    style={{
                                        position: "relative",
                                        width: "100%",
                                        overflow: "hidden",
                                        borderRadius: Math.max(10, cornerRadius - 8),
                                    }}
                                >
                                    <img
                                        src={previewUrl}
                                        alt="Annotated EarPods crossings"
                                        style={{
                                            width: "100%",
                                            display: "block",
                                        }}
                                    />
                                    {analysisResult.plan.steps.map((step) => {
                                        const isHovered = hoveredMarker === step.marker
                                        return (
                                            <div
                                                key={`marker-${step.marker}`}
                                                style={{
                                                    position: "absolute",
                                                    left: `${step.coord.x * 100}%`,
                                                    top: `${step.coord.y * 100}%`,
                                                    transform: `translate(-50%, -50%) scale(${isHovered ? 1.12 : 1})`,
                                                    width: 24,
                                                    height: 24,
                                                    borderRadius: "50%",
                                                    background: accentColor,
                                                    color: surfaceColor,
                                                    border: "2px solid #FFFFFF",
                                                    boxShadow: isHovered
                                                        ? "0 4px 12px rgba(0,0,0,0.35)"
                                                        : "0 2px 8px rgba(0,0,0,0.25)",
                                                    display: "flex",
                                                    alignItems: "center",
                                                    justifyContent: "center",
                                                    fontSize: 12,
                                                    fontWeight: 700,
                                                    zIndex: 2,
                                                    transition: "transform 160ms ease, box-shadow 160ms ease",
                                                }}
                                            >
                                                {step.marker}
                                            </div>
                                        )
                                    })}
                                </div>
                            ) : null}
                            <ol
                                style={{
                                    margin: 0,
                                    paddingLeft: 0,
                                    listStyle: "none",
                                    color: textColor,
                                    fontSize: 14,
                                    lineHeight: 1.45,
                                }}
                            >
                                {analysisResult.plan.steps.map((step, index) => (
                                    <li
                                        key={step.marker}
                                        style={{ marginBottom: 6 }}
                                        onMouseEnter={() =>
                                            startTransition(() => {
                                                setHoveredMarker(step.marker)
                                            })
                                        }
                                        onMouseLeave={() =>
                                            startTransition(() => {
                                                setHoveredMarker(null)
                                            })
                                        }
                                    >
                                        {step.text}
                                    </li>
                                ))}
                            </ol>
                            <button
                                type="button"
                                onClick={() => startTransition(() => setShowDetails((prev) => !prev))}
                                style={{
                                    appearance: "none",
                                    border: "none",
                                    background: "transparent",
                                    color: accentColor,
                                    textAlign: "left",
                                    fontSize: 13,
                                    fontWeight: 600,
                                    padding: 0,
                                    cursor: "pointer",
                                }}
                            >
                                {showDetails ? "Hide topology details" : "Show topology details"}
                            </button>
                            {showDetails ? (
                                <div
                                    style={{
                                        borderTop: `1px solid ${borderColor}`,
                                        paddingTop: 8,
                                        color: mutedColor,
                                        fontSize: 13,
                                        lineHeight: 1.45,
                                    }}
                                >
                                    <div>
                                        Knot type: {analysisResult.diagram.knotName}
                                        {analysisResult.diagram.knotIdentified
                                            ? " (matched by Jones polynomial)"
                                            : " (estimated from crossing count)"}
                                    </div>
                                    <div>Gauss code: {analysisResult.diagram.gaussCode}</div>
                                    <div>
                                        Crossing number: {analysisResult.diagram.crossingNumber}
                                        {analysisResult.diagram.bestEstimate ? " (best estimate)" : ""}
                                    </div>
                                    <div>
                                        Jones polynomial {analysisResult.diagram.bestEstimate ? "estimate" : ""}:{" "}
                                        {analysisResult.diagram.jonesPolynomial}
                                    </div>
                                    <div>Writhe: {analysisResult.diagram.writhe}</div>
                                    <div>
                                        Uncertain crossings: {analysisResult.diagram.uncertainCrossings}
                                        {analysisResult.diagram.uncertainCrossings > 0
                                            ? ` (confidence below ${confidenceThreshold.toFixed(2)})`
                                            : ""}
                                    </div>
                                </div>
                            ) : null}
                        </div>
                    ) : null}
                </div>
            ) : null}
        </div>
    )
}

addPropertyControls(EarPodsUploader, {
    accentColor: {
        type: ControlType.Color,
        defaultValue: "#0A84FF",
    },
    surfaceColor: {
        type: ControlType.Color,
        defaultValue: "#FFFFFF",
    },
    borderColor: {
        type: ControlType.Color,
        defaultValue: "#E6E6EA",
    },
    textColor: {
        type: ControlType.Color,
        defaultValue: "#1D1D1F",
    },
    mutedColor: {
        type: ControlType.Color,
        defaultValue: "#6E6E73",
    },
    cornerRadius: {
        type: ControlType.Number,
        defaultValue: 20,
        min: 8,
        max: 40,
        step: 1,
        unit: "px",
    },
    promptTitle: {
        type: ControlType.String,
        defaultValue: "Take a photo of your tangled EarPods",
    },
    promptSubtitle: {
        type: ControlType.String,
        defaultValue: "or upload an existing image (PNG or JPG)",
    },
    successText: {
        type: ControlType.String,
        defaultValue: "Great — now follow the steps below 👇",
    },
    enableAnalysis: {
        type: ControlType.Boolean,
        defaultValue: true,
    },
    visionApiUrl: {
        type: ControlType.String,
        defaultValue: "",
        placeholder: "https://your-api/analyze",
    },
    visionApiKey: {
        type: ControlType.String,
        defaultValue: "",
        obscured: true,
        placeholder: "Optional bearer key",
    },
    confidenceThreshold: {
        type: ControlType.Number,
        defaultValue: 0.6,
        min: 0,
        max: 1,
        step: 0.01,
    },
    analysisTimeoutSeconds: {
        type: ControlType.Number,
        title: "Analysis Timeout (s)",
        defaultValue: 45,
        min: 10,
        max: 120,
        step: 1,
    },
    enableCamera: {
        type: ControlType.Boolean,
        defaultValue: true,
    },
    cameraButtonLabel: {
        type: ControlType.String,
        defaultValue: "Use camera",
    },
    uploadButtonLabel: {
        type: ControlType.String,
        defaultValue: "Upload an image",
    },
    fallbackNoteText: {
        type: ControlType.String,
        defaultValue:
            "Automatic analysis couldn’t get a confident read this time. Please follow the general untangling steps below, or try a clearer, well-lit photo on a plain background.",
        displayTextArea: true,
    },
})
