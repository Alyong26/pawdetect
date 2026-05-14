/**
 * Client-only TensorFlow helpers.
 * Import these modules only from `"use client"` components so the bundle stays out of RSC.
 */
import * as tf from "@tensorflow/tfjs";
import {
  alignPixelNormWithModelTopology,
  setActiveInferConfig,
  type PawdetectInferJson,
} from "./inferConfig";

let cachedModel: tf.LayersModel | null = null;
let loadPromise: Promise<tf.LayersModel> | null = null;

/** Last successfully loaded model URL (same tab / until dispose). */
let lastLoadedModelUrl: string | null = null;
/** Fingerprint of model.json used for Cache Storage bucket (debug). */
let lastModelFingerprint: string | null = null;

export function getLoadedModelDebugInfo(): { modelUrl: string | null; fingerprint: string | null } {
  return { modelUrl: lastLoadedModelUrl, fingerprint: lastModelFingerprint };
}

const CACHE_PREFIX = "pawdetect-tfjs-";

/** Stable fingerprint without SubtleCrypto (LAN http:// IPs are not a “secure context”). */
function fingerprintModelJsonText(text: string): string {
  let h = 5381;
  for (let i = 0; i < text.length; i++) {
    h = (Math.imul(33, h) + text.charCodeAt(i)) | 0;
  }
  const mid = text.length >> 1;
  let t = 0;
  for (let i = mid; i < Math.min(text.length, mid + 800); i++) {
    t = (Math.imul(31, t) + text.charCodeAt(i)) | 0;
  }
  return `${(h >>> 0).toString(16)}-${text.length}-${(t >>> 0).toString(16)}`;
}

/**
 * Bucket name for Cache Storage only from topology + manifest (not whole file whitespace /
 * metadata). Avoids orphaning the shard cache when inconsequential JSON changes.
 */
function fingerprintStructuredModelSubset(raw: Record<string, unknown>): string {
  const subset = {
    modelTopology: raw.modelTopology,
    weightsManifest: raw.weightsManifest,
  };
  return fingerprintModelJsonText(JSON.stringify(subset));
}

function shardGetRequest(shardUrl: string): Request {
  return new Request(shardUrl, { method: "GET", mode: "same-origin", credentials: "same-origin" });
}

async function pruneTfjsCaches(keep: string): Promise<void> {
  if (typeof caches === "undefined") return;
  const keys = await caches.keys();
  for (const name of keys) {
    if (name.startsWith(CACHE_PREFIX) && name !== keep) {
      await caches.delete(name);
    }
  }
}

function toAbsoluteUrl(modelUrl: string): string {
  return new URL(modelUrl, typeof window !== "undefined" ? window.location.href : "http://localhost/").href;
}

/**
 * Read model.json text (HTTP cache only — small file). Shards use Cache Storage below.
 */
async function fetchModelJsonText(modelUrl: string): Promise<string> {
  const url = toAbsoluteUrl(modelUrl);
  const res = await fetch(url, { credentials: "same-origin", cache: "default" });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  const text = await res.text();
  if (text.length === 0) {
    throw new Error(`Empty model JSON: ${url}`);
  }
  return text;
}

async function fetchInferJson(prefixUrl: string): Promise<PawdetectInferJson | null> {
  const url = new URL("infer.json", prefixUrl).href;
  try {
    const res = await fetch(url, { credentials: "same-origin", cache: "default" });
    if (!res.ok) return null;
    const text = await res.text();
    if (!text.trim()) return null;
    const parsed = JSON.parse(text) as PawdetectInferJson;
    if (!Array.isArray(parsed.kerasClassIndexFoldersAlphabetical)) return null;
    if (parsed.kerasClassIndexFoldersAlphabetical.length !== 2) return null;
    const [a, b] = parsed.kerasClassIndexFoldersAlphabetical;
    if (typeof a !== "string" || typeof b !== "string") return null;
    return {
      kerasClassIndexFoldersAlphabetical: [a, b],
      pixelNorm: parsed.pixelNorm,
    };
  } catch {
    return null;
  }
}

/** Fetch shard from network (HTTP cache / SW allowed). Avoid `reload` — it bypasses cache every time. */
async function fetchShardArrayBufferOnce(shardUrl: string): Promise<ArrayBuffer> {
  const res = await fetch(shardGetRequest(shardUrl));
  if (res.ok) {
    const buf = await res.arrayBuffer();
    if (buf.byteLength > 0) return buf;
  }
  const bust = `${shardUrl}${shardUrl.includes("?") ? "&" : "?"}__tfjs=${Date.now()}`;
  const resBust = await fetch(bust, { credentials: "same-origin", cache: "no-store" });
  if (!resBust.ok) {
    throw new Error(`Failed to fetch weight shard (${shardUrl}): ${resBust.status} ${resBust.statusText}`);
  }
  const buf = await resBust.arrayBuffer();
  if (buf.byteLength === 0) {
    throw new Error(
      `Empty weight shard: ${shardUrl}. If you use a PWA build, unregister the service worker once (Application tab).`,
    );
  }
  return buf;
}

/**
 * Weight shard: Cache Storage first (survives full page reload), then network.
 * Bucket name includes fingerprint of model.json so new exports invalidate automatically.
 */
async function getShardBufferCached(
  shardUrl: string,
  cacheName: string,
): Promise<ArrayBuffer> {
  if (typeof caches === "undefined") {
    return fetchShardArrayBufferOnce(shardUrl);
  }
  const req = shardGetRequest(shardUrl);
  try {
    const cache = await caches.open(cacheName);
    const hit = await cache.match(req);
    if (hit) {
      const buf = await hit.arrayBuffer();
      if (buf.byteLength > 0) return buf;
      await cache.delete(req);
    }
    const buf = await fetchShardArrayBufferOnce(shardUrl);
    await cache.put(
      req,
      new Response(buf, {
        headers: { "Content-Type": "application/octet-stream" },
      }),
    );
    return buf;
  } catch (err) {
    if (process.env.NODE_ENV === "development" && typeof console !== "undefined") {
      console.warn("[PawDetect] Cache Storage unavailable for weights; using network only.", err);
    }
    return fetchShardArrayBufferOnce(shardUrl);
  }
}

const KERAS_REGULARIZER_KEYS = ["kernel_regularizer", "bias_regularizer", "activity_regularizer"] as const;

/**
 * Keras / keras→tfjs exports `kernel_regularizer` with class_name `L2`, which tfjs-layers
 * does not deserialize ("Unknown regularizer: L2"). Regularizers do not affect inference.
 */
function stripKerasRegularizersForTfjs(obj: unknown): void {
  if (obj === null || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const x of obj) stripKerasRegularizersForTfjs(x);
    return;
  }
  const o = obj as Record<string, unknown>;
  for (const k of KERAS_REGULARIZER_KEYS) {
    if (k in o) delete o[k];
  }
  for (const key of Object.keys(o)) stripKerasRegularizersForTfjs(o[key]);
}

/** Keras 3 JSON uses `batch_shape` on InputLayer; tfjs-layers expects `batchInputShape`. */
function patchKeras3InputLayerBatchShape(obj: unknown): void {
  if (obj === null || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const x of obj) patchKeras3InputLayerBatchShape(x);
    return;
  }
  const o = obj as Record<string, unknown>;
  if (o.class_name === "InputLayer" && o.config && typeof o.config === "object") {
    const cfg = o.config as Record<string, unknown>;
    if (cfg.batch_shape != null && cfg.batchInputShape == null) {
      cfg.batchInputShape = cfg.batch_shape;
      delete cfg.batch_shape;
    }
  }
  for (const k of Object.keys(o)) patchKeras3InputLayerBatchShape(o[k]);
}

function isKeras3InboundNodeRecord(node: unknown): node is Record<string, unknown> & { args: unknown[] } {
  if (node === null || typeof node !== "object" || Array.isArray(node)) return false;
  const o = node as Record<string, unknown>;
  if (Array.isArray(o.args)) return true;
  if (o.args !== null && typeof o.args === "object" && !Array.isArray(o.args)) return true;
  return false;
}

/**
 * Keras 3 `args` is usually `[__keras_tensor__, …]` but multi-input layers (e.g. `Add`) nest
 * tensors as `[[tensorA, tensorB]]`. Collect every `__keras_tensor__` / `KerasTensor` dict.
 */
function keras3CollectTensorSpecs(args: unknown): unknown[] {
  const specs: unknown[] = [];
  function visit(node: unknown): void {
    if (Array.isArray(node)) {
      for (const x of node) visit(x);
      return;
    }
    if (node === null || typeof node !== "object") return;
    const o = node as Record<string, unknown>;
    const cn = (o.class_name ?? o.className) as string | undefined;
    if (cn === "__keras_tensor__" || cn === "KerasTensor") {
      specs.push(node);
    }
  }
  visit(args);
  return specs;
}

/**
 * Keras 3 nests each inbound edge as `{ args: [__keras_tensor__], kwargs }`.
 * tfjs-layers expects each inboundNodes entry to be an array of
 * `[inboundLayerName, nodeIndex, tensorIndex, optionalKwargs]`.
 */
function keras3InboundNodeToLegacy(node: unknown): unknown {
  if (!isKeras3InboundNodeRecord(node)) return node;
  const n = node as Record<string, unknown>;
  const tuples: unknown[] = [];
  const outerKwargs = n.kwargs;
  const hasOuterKwargs =
    outerKwargs && typeof outerKwargs === "object" && !Array.isArray(outerKwargs)
      ? Object.keys(outerKwargs as object).length > 0
      : false;

  for (const arg of keras3CollectTensorSpecs(n.args)) {
    if (arg === null || typeof arg !== "object" || Array.isArray(arg)) continue;
    const a = arg as Record<string, unknown>;
    const cn = (a.class_name ?? a.className) as string | undefined;
    if (cn !== "__keras_tensor__" && cn !== "KerasTensor") continue;
    const cfg = a.config;
    if (cfg === null || typeof cfg !== "object" || Array.isArray(cfg)) continue;
    const c = cfg as Record<string, unknown>;
    let hist = c.keras_history ?? c.kerasHistory;
    if (!Array.isArray(hist) && hist && typeof hist === "object" && !Array.isArray(hist)) {
      const h = hist as Record<string, unknown>;
      if (h.layer != null || h[0] != null) {
        hist = [h.layer ?? h[0], h.node_index ?? h.nodeIndex ?? h[1], h.tensor_index ?? h.tensorIndex ?? h[2]];
      }
    }
    if (!Array.isArray(hist) || hist.length < 3) continue;
    const layerName = hist[0];
    const nodeIdx = Number(hist[1]);
    const tensorIdx = Number(hist[2]);
    if (typeof layerName !== "string" || !Number.isFinite(nodeIdx) || !Number.isFinite(tensorIdx)) continue;
    const row: unknown[] = [layerName, nodeIdx, tensorIdx];
    if (hasOuterKwargs) row.push(outerKwargs);
    tuples.push(row);
  }
  if (tuples.length === 0) return node;
  return tuples;
}

function isPlainRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/** `inbound_nodes` is sometimes a dict keyed by "0","1",… instead of an array. */
function recordWithNumericKeysToArray(v: Record<string, unknown>): unknown[] | null {
  const keys = Object.keys(v);
  if (keys.length === 0) return null;
  if (!keys.every((k) => /^\d+$/.test(k))) return null;
  return keys.sort((a, b) => Number(a) - Number(b)).map((k) => v[k]);
}

function patchKeras3InboundNodes(obj: unknown): void {
  if (obj === null || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const x of obj) patchKeras3InboundNodes(x);
    return;
  }
  const o = obj as Record<string, unknown>;
  for (const key of Object.keys(o)) {
    if (key === "inbound_nodes" || key === "inboundNodes") {
      const v = o[key];
      if (Array.isArray(v)) {
        o[key] = v.map((node) => keras3InboundNodeToLegacy(node));
      } else if (isPlainRecord(v)) {
        const asArr = recordWithNumericKeysToArray(v);
        if (asArr) {
          o[key] = asArr.map((node) => keras3InboundNodeToLegacy(node));
        } else if (isKeras3InboundNodeRecord(v)) {
          o[key] = [keras3InboundNodeToLegacy(v)];
        } else {
          patchKeras3InboundNodes(v);
        }
      }
    } else {
      patchKeras3InboundNodes(o[key]);
    }
  }
}

/** Directory URL for resolving `weightsManifest` paths (must end with `/`). */
function weightPathPrefixFromModelUrl(modelUrl: string): string {
  const u = new URL(modelUrl, typeof window !== "undefined" ? window.location.href : "http://localhost/");
  u.pathname = u.pathname.replace(/\/[^/]+$/, "/");
  return u.href;
}

/**
 * Keras 3 Sequential exports weight names as `{sequentialName}/conv2d/kernel`;
 * tfjs-layers expects `conv2d/kernel` after topology deserialize.
 */
function patchKeras3SequentialWeightManifestNames(modelAndWeights: Record<string, unknown>): void {
  const topology = modelAndWeights.modelTopology;
  if (!topology || typeof topology !== "object") return;
  const modelConfig = (topology as Record<string, unknown>).model_config;
  if (!modelConfig || typeof modelConfig !== "object") return;
  const mc = modelConfig as Record<string, unknown>;
  if (mc.class_name !== "Sequential") return;
  const cfg = mc.config;
  if (!cfg || typeof cfg !== "object") return;
  const seqName = (cfg as Record<string, unknown>).name;
  if (typeof seqName !== "string" || !seqName) return;
  const prefix = `${seqName}/`;
  const manifest = modelAndWeights.weightsManifest;
  if (!Array.isArray(manifest)) return;
  for (const group of manifest) {
    if (!group || typeof group !== "object") continue;
    const weights = (group as Record<string, unknown>).weights;
    if (!Array.isArray(weights)) continue;
    for (const w of weights) {
      if (!w || typeof w !== "object") continue;
      const entry = w as Record<string, unknown>;
      const n = entry.name;
      if (typeof n === "string" && n.startsWith(prefix)) {
        entry.name = n.slice(prefix.length);
      }
    }
  }
}

/**
 * Keras 3 names DepthwiseConv2D's kernel weight `kernel`, but tfjs-layers expects
 * `depthwise_kernel`. Rename the manifest entries so weight binding succeeds.
 */
function patchKeras3DepthwiseKernelManifestNames(raw: Record<string, unknown>): void {
  const depthwiseNames = collectLayerNamesByClass(raw, "DepthwiseConv2D");
  if (depthwiseNames.size === 0) return;
  const manifest = raw.weightsManifest;
  if (!Array.isArray(manifest)) return;
  for (const group of manifest) {
    if (!group || typeof group !== "object") continue;
    const weights = (group as Record<string, unknown>).weights;
    if (!Array.isArray(weights)) continue;
    for (const w of weights) {
      if (!w || typeof w !== "object") continue;
      const entry = w as Record<string, unknown>;
      const n = entry.name;
      if (typeof n !== "string") continue;
      const idx = n.lastIndexOf("/kernel");
      if (idx <= 0 || idx !== n.length - "/kernel".length) continue;
      const layerName = n.slice(0, idx);
      if (depthwiseNames.has(layerName)) {
        entry.name = `${layerName}/depthwise_kernel`;
      }
    }
  }
}

function collectLayerNamesByClass(obj: unknown, className: string, into?: Set<string>): Set<string> {
  const acc = into ?? new Set<string>();
  if (obj === null || typeof obj !== "object") return acc;
  if (Array.isArray(obj)) {
    for (const x of obj) collectLayerNamesByClass(x, className, acc);
    return acc;
  }
  const o = obj as Record<string, unknown>;
  if (o.class_name === className && typeof o.name === "string") acc.add(o.name);
  for (const k of Object.keys(o)) collectLayerNamesByClass(o[k], className, acc);
  return acc;
}

const FUNCTIONAL_IO_KEYS = ["input_layers", "inputLayers", "output_layers", "outputLayers"] as const;

/**
 * Keras 3 may emit Functional `input_layers` / `output_layers` as a flat triple
 * `[name, nodeIdx, tensorIdx]`. tfjs `convertPythonicToTs` mishandles that; wrap as
 * `[[name, nodeIdx, tensorIdx]]` on the **raw** JSON before `loadLayersModel` so nested
 * MobileNet graphs stay connected (post-convert repair breaks nested Functional wiring).
 */
function patchKeras3FunctionalFlatIoLists(obj: unknown): void {
  if (obj === null || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const x of obj) patchKeras3FunctionalFlatIoLists(x);
    return;
  }
  const o = obj as Record<string, unknown>;
  for (const key of FUNCTIONAL_IO_KEYS) {
    if (!(key in o)) continue;
    const v = o[key];
    if (!Array.isArray(v) || v.length !== 3) continue;
    const [a, b, c] = v;
    const nodeIdx = Number(b);
    const tensorIdx = Number(c);
    if (typeof a === "string" && Number.isFinite(nodeIdx) && Number.isFinite(tensorIdx)) {
      o[key] = [[a, nodeIdx, tensorIdx]];
    }
  }
  for (const k of Object.keys(o)) patchKeras3FunctionalFlatIoLists(o[k]);
}

type FunctionalEdge = [string, number, number] | [string, number, number, Record<string, unknown>];

function isFunctionalEdge(v: unknown): v is FunctionalEdge {
  return (
    Array.isArray(v) &&
    v.length >= 3 &&
    typeof v[0] === "string" &&
    typeof v[1] === "number" &&
    typeof v[2] === "number"
  );
}

function getInboundNodes(layer: Record<string, unknown>): unknown[][] | null {
  const v = layer.inbound_nodes ?? layer.inboundNodes;
  if (!Array.isArray(v)) return null;
  return v as unknown[][];
}

function getFunctionalConfig(layer: Record<string, unknown>): Record<string, unknown> | null {
  if (layer.class_name !== "Functional") return null;
  const cfg = layer.config;
  if (cfg === null || typeof cfg !== "object" || Array.isArray(cfg)) return null;
  const c = cfg as Record<string, unknown>;
  if (!Array.isArray(c.layers)) return null;
  return c;
}

function ioEntriesAsEdgeList(value: unknown): FunctionalEdge[] {
  if (!Array.isArray(value)) return [];
  if (isFunctionalEdge(value)) return [value];
  const out: FunctionalEdge[] = [];
  for (const item of value) {
    if (isFunctionalEdge(item)) out.push(item);
  }
  return out;
}

/**
 * tfjs-layers cannot deserialize a Keras Functional model whose `layers` contain another
 * `Functional` (e.g. MobileNet wrapped inside the user model): the inner `InputLayer`
 * becomes unreachable from the outer feed and validation throws "Graph disconnected".
 *
 * This rewrites the parsed `model.json` in place to inline every nested Functional:
 *   - Drop the inner `InputLayer` nodes.
 *   - Remap any inbound edge pointing at an inner `InputLayer` to the outer feeder
 *     (taken from the nested Functional's own `inbound_nodes`).
 *   - Splice the inner non-InputLayer layers into the outer `layers` array.
 *   - Rewrite any reference to the nested Functional's name (siblings and `output_layers`)
 *     to point at the corresponding inner output layer (using its `tensor_index`).
 * Weight names are unchanged, so the existing `group1-shard.bin` still binds correctly.
 */
function inlineNestedFunctionalsInPlace(raw: Record<string, unknown>): boolean {
  const topology = raw.modelTopology as Record<string, unknown> | undefined;
  if (!topology || typeof topology !== "object") return false;
  const modelConfig = topology.model_config as Record<string, unknown> | undefined;
  if (!modelConfig || modelConfig.class_name !== "Functional") return false;
  const parentConfig = getFunctionalConfig(modelConfig);
  if (!parentConfig) return false;

  let inlinedAny = false;
  for (let iter = 0; iter < 64; iter++) {
    if (!inlineOneNestedFunctional(parentConfig)) break;
    inlinedAny = true;
  }
  return inlinedAny;
}

function inlineOneNestedFunctional(parentConfig: Record<string, unknown>): boolean {
  const layers = parentConfig.layers as Record<string, unknown>[];
  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i];
    const innerConfig = getFunctionalConfig(layer);
    if (!innerConfig) continue;
    const innerLayers = innerConfig.layers as Record<string, unknown>[];
    if (!Array.isArray(innerLayers) || innerLayers.length === 0) continue;

    const innerName = typeof layer.name === "string" ? layer.name : null;
    if (!innerName) continue;

    const innerInputEdges = ioEntriesAsEdgeList(innerConfig.input_layers ?? innerConfig.inputLayers);
    const innerOutputEdges = ioEntriesAsEdgeList(innerConfig.output_layers ?? innerConfig.outputLayers);
    if (innerInputEdges.length === 0 || innerOutputEdges.length === 0) continue;

    const outerCalls = getInboundNodes(layer) ?? [];
    const firstCall = Array.isArray(outerCalls[0]) ? (outerCalls[0] as unknown[]) : [];

    const inputNameToFeeder: Record<string, FunctionalEdge> = {};
    for (let k = 0; k < innerInputEdges.length; k++) {
      const innerIn = innerInputEdges[k];
      const feederRaw = firstCall[k] ?? firstCall[0];
      if (!isFunctionalEdge(feederRaw)) continue;
      const feeder: FunctionalEdge = [feederRaw[0], feederRaw[1] as number, feederRaw[2] as number];
      inputNameToFeeder[innerIn[0]] = feeder;
    }

    const innerInputNames = new Set(innerInputEdges.map((e) => e[0]));
    const innerOutputNameByTensorIdx: Record<number, string> = {};
    innerOutputEdges.forEach((e, idx) => {
      innerOutputNameByTensorIdx[idx] = e[0];
      if (typeof e[2] === "number") innerOutputNameByTensorIdx[e[2]] = e[0];
    });

    const flatInner: Record<string, unknown>[] = [];
    for (const innerLayer of innerLayers) {
      if (
        innerLayer.class_name === "InputLayer" &&
        typeof innerLayer.name === "string" &&
        innerInputNames.has(innerLayer.name)
      ) {
        continue;
      }
      rewriteInboundEdges(innerLayer, (edge) => {
        const mapped = inputNameToFeeder[edge[0]];
        if (!mapped) return null;
        const next: FunctionalEdge = [mapped[0], mapped[1] as number, mapped[2] as number];
        if (edge.length > 3) (next as unknown[]).push(edge[3]);
        return next;
      });
      flatInner.push(innerLayer);
    }

    layers.splice(i, 1, ...flatInner);

    for (const sibling of layers) {
      if (flatInner.includes(sibling)) continue;
      rewriteInboundEdges(sibling, (edge) => {
        if (edge[0] !== innerName) return null;
        const tIdx = typeof edge[2] === "number" ? edge[2] : 0;
        const targetName = innerOutputNameByTensorIdx[tIdx] ?? innerOutputNameByTensorIdx[0];
        if (!targetName) return null;
        const next: FunctionalEdge = [targetName, 0, 0];
        if (edge.length > 3) (next as unknown[]).push(edge[3]);
        return next;
      });
    }

    for (const key of ["input_layers", "output_layers", "inputLayers", "outputLayers"] as const) {
      const v = parentConfig[key];
      if (!Array.isArray(v)) continue;
      for (let k = 0; k < v.length; k++) {
        const entry = v[k];
        if (!isFunctionalEdge(entry)) continue;
        if (entry[0] !== innerName) continue;
        const tIdx = typeof entry[2] === "number" ? entry[2] : 0;
        const target = innerOutputNameByTensorIdx[tIdx] ?? innerOutputNameByTensorIdx[0];
        if (!target) continue;
        (v as unknown[])[k] = [target, 0, 0];
      }
    }

    return true;
  }
  return false;
}

function rewriteInboundEdges(
  layer: Record<string, unknown>,
  rewrite: (edge: FunctionalEdge) => FunctionalEdge | null,
): void {
  const calls = getInboundNodes(layer);
  if (!calls) return;
  for (const call of calls) {
    if (!Array.isArray(call)) continue;
    for (let i = 0; i < call.length; i++) {
      const edge = call[i];
      if (!isFunctionalEdge(edge)) continue;
      const next = rewrite(edge);
      if (next) call[i] = next;
    }
  }
}

type ModelJsonRoot = {
  modelTopology?: object;
  weightsManifest?: tf.io.WeightsManifestConfig;
  format?: string;
  generatedBy?: string;
  convertedBy?: string | null;
};

/**
 * Fetch `model.json`, apply Keras3→tfjs compatibility patches (batch input shape, Sequential
 * weight names, strip Keras-only regularizers, Keras 3 `inbound_nodes` → legacy arrays), fetch
 * shards, and build {@link tf.io.ModelArtifacts} for `fromMemory`.
 */
async function loadPatchedModelArtifacts(modelUrl: string): Promise<tf.io.ModelArtifacts> {
  setActiveInferConfig(null);
  const jsonText = await fetchModelJsonText(modelUrl);
  const raw = JSON.parse(jsonText) as Record<string, unknown>;
  patchKeras3FunctionalFlatIoLists(raw);
  const fp = fingerprintStructuredModelSubset(raw);
  const cacheName = `${CACHE_PREFIX}${fp}`;
  await pruneTfjsCaches(cacheName);
  lastModelFingerprint = fp;

  patchKeras3InputLayerBatchShape(raw);
  patchKeras3SequentialWeightManifestNames(raw);
  patchKeras3DepthwiseKernelManifestNames(raw);
  stripKerasRegularizersForTfjs(raw);
  patchKeras3InboundNodes(raw);
  const inlined = inlineNestedFunctionalsInPlace(raw);

  if (process.env.NODE_ENV === "development" && typeof console !== "undefined") {
    const r = raw as ModelJsonRoot;
    console.info("[PawDetect] TF.js model assets", {
      modelUrl: toAbsoluteUrl(modelUrl),
      cacheBucket: cacheName,
      inlinedNestedFunctionals: inlined,
      format: r.format,
      generatedBy: r.generatedBy,
      convertedBy: r.convertedBy,
      preprocess: "224×224, RGB; normalization from infer.json pixelNorm (div255 or mobilenet_v2)",
    });
  }

  const root = raw as ModelJsonRoot;
  if (root.modelTopology == null) {
    throw new Error(`Missing modelTopology in ${modelUrl}`);
  }

  const manifest = root.weightsManifest;
  if (manifest == null || manifest.length === 0) {
    setActiveInferConfig(null);
    return {
      modelTopology: root.modelTopology,
      format: root.format,
      generatedBy: root.generatedBy,
      convertedBy: root.convertedBy ?? undefined,
    };
  }

  const prefix = weightPathPrefixFromModelUrl(modelUrl);
  const infer = await fetchInferJson(prefix);
  setActiveInferConfig(infer);
  alignPixelNormWithModelTopology(root.modelTopology, infer);

  const buffers: ArrayBuffer[] = [];
  for (const group of manifest) {
    for (const p of group.paths) {
      const shardUrl = new URL(p.replace(/^\//, ""), prefix).href;
      const buf = await getShardBufferCached(shardUrl, cacheName);
      buffers.push(buf);
    }
  }

  const weightData = tf.io.concatenateArrayBuffers(buffers);
  const weightSpecs = tf.io.getWeightSpecs(manifest);

  return {
    modelTopology: root.modelTopology,
    weightSpecs,
    weightData,
    format: root.format,
    generatedBy: root.generatedBy,
    convertedBy: root.convertedBy ?? undefined,
  };
}

/**
 * Loads the Keras-style model exported to `public/model/model.json` once and caches it in memory.
 * Weight shards are stored in Cache Storage (same bucket while topology + manifest match) so a
 * normal reload reuses bytes without a full download. Production also uses Workbox CacheFirst
 * for `/model/*`. Note: DevTools “Disable cache” or “Empty cache and hard reload” can still force
 * network fetches — that is browser behaviour, not the app clearing weights on purpose.
 */
export async function loadModel(modelUrl = "/model/model.json"): Promise<tf.LayersModel> {
  if (cachedModel) return cachedModel;
  if (!loadPromise) {
    loadPromise = (async () => {
      try {
        const artifacts = await loadPatchedModelArtifacts(modelUrl);
        const model = await tf.loadLayersModel(tf.io.fromMemory(artifacts));
        cachedModel = model;
        lastLoadedModelUrl = toAbsoluteUrl(modelUrl);
        return model;
      } catch (err) {
        loadPromise = null;
        throw err;
      }
    })();
  }
  return loadPromise;
}

export function disposeCachedModel(): void {
  if (cachedModel) {
    cachedModel.dispose();
    cachedModel = null;
    loadPromise = null;
    lastLoadedModelUrl = null;
    lastModelFingerprint = null;
    setActiveInferConfig(null);
  }
}
