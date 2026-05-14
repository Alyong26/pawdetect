/**
 * One-off: verify public/model/model.json loads under the same patches as loadPatchedModelArtifacts.
 * Run: node scripts/test-tfjs-load.mjs
 */
import fs from "node:fs";
import * as tf from "@tensorflow/tfjs";

const KERAS_REGULARIZER_KEYS = ["kernel_regularizer", "bias_regularizer", "activity_regularizer"];

function stripKerasRegularizersForTfjs(obj) {
  if (obj === null || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const x of obj) stripKerasRegularizersForTfjs(x);
    return;
  }
  const o = obj;
  for (const k of KERAS_REGULARIZER_KEYS) {
    if (k in o) delete o[k];
  }
  for (const key of Object.keys(o)) stripKerasRegularizersForTfjs(o[key]);
}

function patchKeras3InputLayerBatchShape(obj) {
  if (obj === null || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const x of obj) patchKeras3InputLayerBatchShape(x);
    return;
  }
  const o = obj;
  if (o.class_name === "InputLayer" && o.config && typeof o.config === "object") {
    const cfg = o.config;
    if (cfg.batch_shape != null && cfg.batchInputShape == null) {
      cfg.batchInputShape = cfg.batch_shape;
      delete cfg.batch_shape;
    }
  }
  for (const k of Object.keys(o)) patchKeras3InputLayerBatchShape(o[k]);
}

const FUNCTIONAL_IO_KEYS = ["input_layers", "inputLayers", "output_layers", "outputLayers"];

function patchKeras3FunctionalFlatIoLists(obj) {
  if (obj === null || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const x of obj) patchKeras3FunctionalFlatIoLists(x);
    return;
  }
  const o = obj;
  for (const key of FUNCTIONAL_IO_KEYS) {
    if (!(key in o)) continue;
    const v = o[key];
    if (!Array.isArray(v) || v.length !== 3) continue;
    const [a, b, c] = v;
    if (typeof a === "string" && typeof b === "number" && typeof c === "number") {
      o[key] = [[a, b, c]];
    }
  }
  for (const k of Object.keys(o)) patchKeras3FunctionalFlatIoLists(o[k]);
}

function patchKeras3SequentialWeightManifestNames(modelAndWeights) {
  const topology = modelAndWeights.modelTopology;
  if (!topology || typeof topology !== "object") return;
  const modelConfig = topology.model_config;
  if (!modelConfig || typeof modelConfig !== "object") return;
  if (modelConfig.class_name !== "Sequential") return;
  const cfg = modelConfig.config;
  if (!cfg || typeof cfg !== "object") return;
  const seqName = cfg.name;
  if (typeof seqName !== "string" || !seqName) return;
  const prefix = `${seqName}/`;
  const manifest = modelAndWeights.weightsManifest;
  if (!Array.isArray(manifest)) return;
  for (const group of manifest) {
    if (!group || typeof group !== "object") continue;
    const weights = group.weights;
    if (!Array.isArray(weights)) continue;
    for (const w of weights) {
      if (!w || typeof w !== "object") continue;
      const n = w.name;
      if (typeof n === "string" && n.startsWith(prefix)) {
        w.name = n.slice(prefix.length);
      }
    }
  }
}

function isKeras3InboundNodeRecord(node) {
  if (node === null || typeof node !== "object" || Array.isArray(node)) return false;
  const o = node;
  if (Array.isArray(o.args)) return true;
  if (o.args !== null && typeof o.args === "object" && !Array.isArray(o.args)) return true;
  return false;
}

function keras3CollectTensorSpecs(args) {
  const specs = [];
  function visit(node) {
    if (Array.isArray(node)) {
      for (const x of node) visit(x);
      return;
    }
    if (node === null || typeof node !== "object") return;
    const o = node;
    const cn = o.class_name ?? o.className;
    if (cn === "__keras_tensor__" || cn === "KerasTensor") {
      specs.push(node);
    }
  }
  visit(args);
  return specs;
}

function keras3InboundNodeToLegacy(node) {
  if (!isKeras3InboundNodeRecord(node)) return node;
  const n = node;
  const tuples = [];
  const outerKwargs = n.kwargs;
  const hasOuterKwargs =
    outerKwargs && typeof outerKwargs === "object" && !Array.isArray(outerKwargs)
      ? Object.keys(outerKwargs).length > 0
      : false;

  for (const arg of keras3CollectTensorSpecs(n.args)) {
    if (arg === null || typeof arg !== "object" || Array.isArray(arg)) continue;
    const a = arg;
    const cn = a.class_name ?? a.className;
    if (cn !== "__keras_tensor__" && cn !== "KerasTensor") continue;
    const cfg = a.config;
    if (cfg === null || typeof cfg !== "object" || Array.isArray(cfg)) continue;
    const c = cfg;
    let hist = c.keras_history ?? c.kerasHistory;
    if (!Array.isArray(hist) && hist && typeof hist === "object" && !Array.isArray(hist)) {
      const h = hist;
      if (h.layer != null || h[0] != null) {
        hist = [h.layer ?? h[0], h.node_index ?? h.nodeIndex ?? h[1], h.tensor_index ?? h.tensorIndex ?? h[2]];
      }
    }
    if (!Array.isArray(hist) || hist.length < 3) continue;
    const layerName = hist[0];
    const nodeIdx = Number(hist[1]);
    const tensorIdx = Number(hist[2]);
    if (typeof layerName !== "string" || !Number.isFinite(nodeIdx) || !Number.isFinite(tensorIdx)) continue;
    const row = [layerName, nodeIdx, tensorIdx];
    if (hasOuterKwargs) row.push(outerKwargs);
    tuples.push(row);
  }
  if (tuples.length === 0) return node;
  return tuples;
}

function isPlainRecord(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function recordWithNumericKeysToArray(v) {
  const keys = Object.keys(v);
  if (keys.length === 0) return null;
  if (!keys.every((k) => /^\d+$/.test(k))) return null;
  return keys.sort((a, b) => Number(a) - Number(b)).map((k) => v[k]);
}

function patchKeras3InboundNodes(obj) {
  if (obj === null || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const x of obj) patchKeras3InboundNodes(x);
    return;
  }
  const o = obj;
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

function isFunctionalEdge(v) {
  return (
    Array.isArray(v) &&
    v.length >= 3 &&
    typeof v[0] === "string" &&
    typeof v[1] === "number" &&
    typeof v[2] === "number"
  );
}

function getInboundNodes(layer) {
  const v = layer.inbound_nodes ?? layer.inboundNodes;
  return Array.isArray(v) ? v : null;
}

function getFunctionalConfig(layer) {
  if (layer.class_name !== "Functional") return null;
  const cfg = layer.config;
  if (cfg === null || typeof cfg !== "object" || Array.isArray(cfg)) return null;
  if (!Array.isArray(cfg.layers)) return null;
  return cfg;
}

function ioEntriesAsEdgeList(value) {
  if (!Array.isArray(value)) return [];
  if (isFunctionalEdge(value)) return [value];
  const out = [];
  for (const item of value) if (isFunctionalEdge(item)) out.push(item);
  return out;
}

function rewriteInboundEdges(layer, rewrite) {
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

function inlineOneNestedFunctional(parentConfig) {
  const layers = parentConfig.layers;
  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i];
    const innerConfig = getFunctionalConfig(layer);
    if (!innerConfig) continue;
    const innerLayers = innerConfig.layers;
    if (!Array.isArray(innerLayers) || innerLayers.length === 0) continue;
    const innerName = typeof layer.name === "string" ? layer.name : null;
    if (!innerName) continue;

    const innerInputEdges = ioEntriesAsEdgeList(innerConfig.input_layers ?? innerConfig.inputLayers);
    const innerOutputEdges = ioEntriesAsEdgeList(innerConfig.output_layers ?? innerConfig.outputLayers);
    if (!innerInputEdges.length || !innerOutputEdges.length) continue;

    const outerCalls = getInboundNodes(layer) ?? [];
    const firstCall = Array.isArray(outerCalls[0]) ? outerCalls[0] : [];

    const inputNameToFeeder = {};
    for (let k = 0; k < innerInputEdges.length; k++) {
      const innerIn = innerInputEdges[k];
      const feederRaw = firstCall[k] ?? firstCall[0];
      if (!isFunctionalEdge(feederRaw)) continue;
      inputNameToFeeder[innerIn[0]] = [feederRaw[0], feederRaw[1], feederRaw[2]];
    }

    const innerInputNames = new Set(innerInputEdges.map((e) => e[0]));
    const innerOutputNameByTensorIdx = {};
    innerOutputEdges.forEach((e, idx) => {
      innerOutputNameByTensorIdx[idx] = e[0];
      if (typeof e[2] === "number") innerOutputNameByTensorIdx[e[2]] = e[0];
    });

    const flatInner = [];
    for (const innerLayer of innerLayers) {
      if (innerLayer.class_name === "InputLayer" && innerInputNames.has(innerLayer.name)) continue;
      rewriteInboundEdges(innerLayer, (edge) => {
        const mapped = inputNameToFeeder[edge[0]];
        if (!mapped) return null;
        const next = [mapped[0], mapped[1], mapped[2]];
        if (edge.length > 3) next.push(edge[3]);
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
        const target = innerOutputNameByTensorIdx[tIdx] ?? innerOutputNameByTensorIdx[0];
        if (!target) return null;
        const next = [target, 0, 0];
        if (edge.length > 3) next.push(edge[3]);
        return next;
      });
    }

    for (const key of ["input_layers", "output_layers", "inputLayers", "outputLayers"]) {
      const v = parentConfig[key];
      if (!Array.isArray(v)) continue;
      for (let k = 0; k < v.length; k++) {
        const entry = v[k];
        if (!isFunctionalEdge(entry)) continue;
        if (entry[0] !== innerName) continue;
        const tIdx = typeof entry[2] === "number" ? entry[2] : 0;
        const target = innerOutputNameByTensorIdx[tIdx] ?? innerOutputNameByTensorIdx[0];
        if (!target) continue;
        v[k] = [target, 0, 0];
      }
    }

    return true;
  }
  return false;
}

function inlineNestedFunctionalsInPlace(raw) {
  const mc = raw.modelTopology?.model_config;
  if (!mc || mc.class_name !== "Functional") return false;
  const parentConfig = getFunctionalConfig(mc);
  if (!parentConfig) return false;
  let any = false;
  for (let iter = 0; iter < 64; iter++) {
    if (!inlineOneNestedFunctional(parentConfig)) break;
    any = true;
  }
  return any;
}

const raw = JSON.parse(fs.readFileSync(new URL("../public/model/model.json", import.meta.url), "utf8"));

function collectLayerNamesByClass(obj, className, into) {
  const acc = into ?? new Set();
  if (obj === null || typeof obj !== "object") return acc;
  if (Array.isArray(obj)) {
    for (const x of obj) collectLayerNamesByClass(x, className, acc);
    return acc;
  }
  if (obj.class_name === className && typeof obj.name === "string") acc.add(obj.name);
  for (const k of Object.keys(obj)) collectLayerNamesByClass(obj[k], className, acc);
  return acc;
}

function patchKeras3DepthwiseKernelManifestNames(raw) {
  const names = collectLayerNamesByClass(raw, "DepthwiseConv2D");
  if (!names.size) return;
  const manifest = raw.weightsManifest;
  if (!Array.isArray(manifest)) return;
  for (const group of manifest) {
    const weights = group?.weights;
    if (!Array.isArray(weights)) continue;
    for (const w of weights) {
      const n = w?.name;
      if (typeof n !== "string") continue;
      const idx = n.lastIndexOf("/kernel");
      if (idx <= 0 || idx !== n.length - "/kernel".length) continue;
      const layer = n.slice(0, idx);
      if (names.has(layer)) w.name = `${layer}/depthwise_kernel`;
    }
  }
}

patchKeras3FunctionalFlatIoLists(raw);
patchKeras3InputLayerBatchShape(raw);
patchKeras3SequentialWeightManifestNames(raw);
patchKeras3DepthwiseKernelManifestNames(raw);
stripKerasRegularizersForTfjs(raw);
patchKeras3InboundNodes(raw);
const inlined = inlineNestedFunctionalsInPlace(raw);
console.log("patches: all + depthwise inlinedNested:", inlined);

const buf = fs.readFileSync(new URL("../public/model/group1-shard.bin", import.meta.url));
const manifest = raw.weightsManifest;
const specs = tf.io.getWeightSpecs(manifest);
const art = {
  modelTopology: raw.modelTopology,
  weightSpecs: specs,
  weightData: buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  format: raw.format,
  generatedBy: raw.generatedBy,
  convertedBy: raw.convertedBy,
};

const model = await tf.loadLayersModel(tf.io.fromMemory(art));
console.log("load ok:", model.name);
const x = tf.zeros([1, 224, 224, 3]);
const y = model.predict(x);
console.log("predict shape:", y.shape);
x.dispose();
y.dispose();
model.dispose();
