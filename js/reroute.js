import { app } from "../../scripts/app.js";
import { getInputLink, getOutputLinks } from "./base.js";
import { ExtensionNames, NodeNames } from "./constants.js";

const NODE_TYPE = NodeNames.Reroute;
const DEFAULT_SIZE = [120, 30];

function normalizeType(type) {
    if (Array.isArray(type)) return type.join(",");
    return String(type ?? "*");
}

function getPrimaryType(type) {
    return normalizeType(type).split(",")[0].trim().toUpperCase();
}

function getLinkColor(type) {
    const normalized = normalizeType(type);
    const primary = getPrimaryType(normalized);

    return (
        LGraphCanvas.link_type_colors?.[normalized] ??
        LGraphCanvas.link_type_colors?.[primary] ??
        LiteGraph.slot_types_default_out?.[normalized]?.color ??
        LiteGraph.slot_types_default_out?.[primary]?.color ??
        LiteGraph.slot_types_default_in?.[normalized]?.color ??
        LiteGraph.slot_types_default_in?.[primary]?.color ??
        undefined
    );
}

function markDirty() {
    app.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function isRerouteNode(node) {
    return node?.type === NODE_TYPE || node?.constructor?.type === NODE_TYPE;
}

function hideSlotLabels(node) {
    for (const input of node.inputs ?? []) {
        input.name = "";
        input.label = "";
        input.localized_name = "";
    }

    for (const output of node.outputs ?? []) {
        output.name = "";
        output.label = "";
        output.localized_name = "";
    }
}

function clearSlotColors(node) {
    for (const input of node.inputs ?? []) {
        delete input.color_on;
        delete input.color_off;
    }

    for (const output of node.outputs ?? []) {
        delete output.color_on;
        delete output.color_off;
    }
}

function normalizeRerouteSlots(node, type = "*") {
    node.inputs ??= [];
    node.outputs ??= [];

    if (!node.inputs.length) node.addInput("", type);
    if (!node.outputs.length) node.addOutput("", type);

    node.inputs.length = 1;
    node.outputs.length = 1;

    node.inputs[0].name = "";
    node.inputs[0].label = "";
    node.inputs[0].localized_name = "";
    node.inputs[0].type = "*";

    node.outputs[0].name = "";
    node.outputs[0].label = "";
    node.outputs[0].localized_name = "";
    node.outputs[0].type = type;

    clearSlotColors(node);

    const color = getLinkColor(type);

    if (color && type !== "*") {
        node.inputs[0].color_on = color;
        node.inputs[0].color_off = color;
        node.outputs[0].color_on = color;
        node.outputs[0].color_off = color;
    }
}

function darkenHex(hex, amount = 0.45) {
    if (!hex || !hex.startsWith("#")) return hex;

    const num = parseInt(hex.slice(1), 16);
    const r = Math.round(((num >> 16) & 255) * amount);
    const g = Math.round(((num >> 8) & 255) * amount);
    const b = Math.round((num & 255) * amount);

    return `#${[r, g, b].map(v => v.toString(16).padStart(2, "0")).join("")}`;
}

function applyNodeColor(node, type) {
    const linkColor = getLinkColor(type);
    const borderColor = linkColor ?? "#353535";
    const backgroundColor = linkColor ? darkenHex(linkColor, 0.38) : "#212121";

    node.__neApplyingAutoColor = true;

    node.color = borderColor;
    node.bgcolor = backgroundColor;
    node.background_color = backgroundColor;

    node.properties ??= {};
    node.properties.color = borderColor;
    node.properties.bgcolor = backgroundColor;

    node.__neApplyingAutoColor = false;
}

function getConnectedSlotFromLink(graph, link, direction) {
    const connectedNode =
        direction === "INPUT"
            ? graph?.getNodeById(link.origin_id)
            : graph?.getNodeById(link.target_id);

    const connectedSlot =
        direction === "INPUT"
            ? connectedNode?.outputs?.[link.origin_slot]
            : connectedNode?.inputs?.[link.target_slot];

    return { connectedNode, connectedSlot };
}

function propagateTitleFrom(startNode) {
    const title = String(startNode.title ?? "").trim();

    if (!title) return;

    const queue = [...getNeighborReroutes(startNode)];
    const visited = new Set([startNode.id]);

    while (queue.length) {
        const node = queue.shift();

        if (!node || visited.has(node.id)) continue;
        visited.add(node.id);

        if (isRerouteNode(node)) {
            if (node.title !== title) {
                node.title = title;
                node.__neLastTitle = title;
            }

            for (const neighbor of getNeighborReroutes(node)) {
                queue.push(neighbor);
            }
        }
    }

    markDirty();
}

function resolveTypeThroughReroutes(startNode, direction, visited = new Set()) {
    if (!startNode || visited.has(startNode.id)) return null;

    visited.add(startNode.id);

    const slotCount = direction === "INPUT"
        ? startNode.inputs?.length ?? 0
        : startNode.outputs?.length ?? 0;

    for (let index = 0; index < slotCount; index++) {
        const links = direction === "INPUT"
            ? [getInputLink(startNode, index)]
            : getOutputLinks(startNode, index);

        for (const link of links.filter(Boolean)) {
            const { connectedNode, connectedSlot } = getConnectedSlotFromLink(startNode.graph, link, direction);

            if (!connectedNode || !connectedSlot) continue;

            if (isRerouteNode(connectedNode)) {
                const resolved = resolveTypeThroughReroutes(connectedNode, direction, visited);
                if (resolved) return resolved;
                continue;
            }

            const type = connectedSlot.type;

            if (type != null && type !== "*") {
                return {
                    type,
                    label: connectedSlot.label,
                    name: connectedSlot.name,
                };
            }
        }
    }

    return null;
}

function getNeighborReroutes(node) {
    const neighbors = new Set();

    const inputNode = getInputReroute(node);
    if (inputNode) neighbors.add(inputNode);

    for (const outputNode of getOutputReroutes(node)) {
        neighbors.add(outputNode);
    }

    return neighbors;
}

function getInputReroute(node) {
    const inputLink = getInputLink(node, 0);
    if (!inputLink) return null;

    const inputNode = node.graph.getNodeById(inputLink.origin_id);
    return isRerouteNode(inputNode) ? inputNode : null;
}

function getOutputReroutes(node) {
    const result = [];

    for (const link of getOutputLinks(node, 0)) {
        const outputNode = node.graph.getNodeById(link.target_id);
        if (isRerouteNode(outputNode)) result.push(outputNode);
    }

    return result;
}

function findUpstreamTitleSource(startNode) {
    const visited = new Set();
    let node = startNode;
    let fallback = null;

    while (node && !visited.has(node.id)) {
        visited.add(node.id);

        if (node.title?.trim()) {
            fallback = node;
        }

        node = getInputReroute(node);
    }

    return fallback;
}

function findAnyTitleSource(graph, nodeIds) {
    for (const nodeId of nodeIds) {
        const node = graph?.getNodeById(nodeId);
        if (node?.title?.trim()) return node;
    }

    return null;
}

function updateRerouteChain(startNode, preferredTitleSource = null) {
    const queue = [startNode];
    const visited = new Set();

    while (queue.length) {
        const node = queue.shift();

        if (!node || visited.has(node.id)) continue;

        visited.add(node.id);
        node.updateTypeAndStyle?.(true);

        for (const neighbor of getNeighborReroutes(node)) {
            queue.push(neighbor);
        }
    }

    const titleSource =
        (preferredTitleSource?.title?.trim() ? preferredTitleSource : null) ??
        findUpstreamTitleSource(startNode) ??
        findAnyTitleSource(startNode.graph, visited);

    if (titleSource) {
        propagateTitleFrom(titleSource);
    }

    markDirty();
}

function resetRerouteType(node) {
    node.resolvedType = "*";
    node.nodeType = "*";

    node.properties ??= {};
    node.properties.resolvedTypeLabel = "";

    normalizeRerouteSlots(node, "*");
    clearSlotColors(node);
    hideSlotLabels(node);

    if (node.properties.autoColor !== false) {
        node.__neApplyingAutoColor = true;

        node.color = "#353535";
        node.bgcolor = "#212121";
        node.background_color = "#212121";
        node.properties.color = "#353535";
        node.properties.bgcolor = "#212121";

        node.__neApplyingAutoColor = false;
    }
}

function updateTypeAndStyle(node, skipChain = false) {
    const connectedType =
        resolveTypeThroughReroutes(node, "INPUT") ??
        resolveTypeThroughReroutes(node, "OUTPUT");

    const type = connectedType?.type ?? "*";
    const label =
        type === "*"
            ? ""
            : connectedType?.label || connectedType?.name || normalizeType(type);

    node.resolvedType = type;
    node.nodeType = type;

    node.properties ??= {};
    node.properties.resolvedTypeLabel = label;

    normalizeRerouteSlots(node, type);
    hideSlotLabels(node);

    const linkColor = getLinkColor(type);

    const inputLink = getInputLink(node, 0);
    if (inputLink) {
        inputLink.type = type;
        inputLink.color = linkColor;
    }

    for (const link of getOutputLinks(node, 0)) {
        link.type = type;
        link.color = linkColor;
    }

    if (node.properties.autoColor !== false) {
        applyNodeColor(node, type);
    }

    if (!skipChain) {
        updateRerouteChain(node);
    }
}

function getDisplayTitle(node) {
    const manualTitle = String(node.title ?? "").trim();

    if (manualTitle) {
        return manualTitle;
    }

    const typeLabel = node.properties?.resolvedTypeLabel;

    if (typeLabel && typeLabel !== "*") {
        return String(typeLabel);
    }

    return "";
}

function syncTitleChange(node) {
    const title = String(node.title ?? "");

    if (node.__neLastTitle === undefined) {
        node.__neLastTitle = title;
        return;
    }

    if (node.__neLastTitle !== title) {
        node.__neLastTitle = title;

        if (title.trim()) {
            propagateTitleFrom(node);
        }

        markDirty();
    }
}

function drawRerouteTitle(node, ctx, canvas) {
    syncTitleChange(node);
    hideSlotLabels(node);

    const scale = canvas?.ds?.scale ?? 1;
    if (scale < 0.45 || node.size[0] < 24 || node.size[1] < 16) return;

    const title = getDisplayTitle(node);

    ctx.save();
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#ccc";
    ctx.font = `${Math.min(12, Math.max(9, node.size[1] * 0.45))}px Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(title, node.size[0] / 2, node.size[1] / 2, Math.max(10, node.size[0] - 14));
    ctx.restore();
}

function setupRerouteNode(node, forceAutoColor = false) {
    if (node.title == null || node.title === "Reroute") {
        node.title = "";
    }

    node.isVirtualNode = true;
    node.hideSlotLabels = true;
    node.shape = LiteGraph.BOX_SHAPE;
    node.constructor.shape = LiteGraph.BOX_SHAPE;
    node.size = [...DEFAULT_SIZE];

    node.properties ??= {};
    
    if (forceAutoColor) {
        node.properties.autoColor = true;
    } else {
        node.properties.autoColor ??= true;
    }

    node.properties.resolvedTypeLabel ??= "";

    node.resetType = function () {
        resetRerouteType(this);
    };

    node.updateTypeAndStyle = function (skipChain = false) {
        updateTypeAndStyle(this, skipChain);
    };

    normalizeRerouteSlots(node, node.resolvedType ?? "*");
    hideSlotLabels(node);
    updateTypeAndStyle(node, true);

    node.__neLastTitle = String(node.title ?? "");
}

function disconnectExistingInput(node) {
    if (!node.isInputConnected(0)) return;

    node.disconnectInput(0);
    node.inputs[0].type = "*";

    resetRerouteType(node);
    markDirty();
}

app.registerExtension({
    name: ExtensionNames.Reroute,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        nodeType.category = "NE.Tools";
        nodeType.comfyClass = NODE_TYPE;
        nodeType.title = "";
        nodeType.title_mode = LiteGraph.NO_TITLE;
        nodeType.collapsable = false;
        nodeType.shape = LiteGraph.BOX_SHAPE;
        nodeType.size = [...DEFAULT_SIZE];

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            setupRerouteNode(this, true);

            requestAnimationFrame(() => {
                setupRerouteNode(this, true);
                markDirty();
            });

            return result;
        };

        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            if (info?.inputs?.length) info.inputs.length = 1;
            if (info?.outputs?.length) info.outputs.length = 1;

            const result = configure?.apply(this, arguments);

            setupRerouteNode(this);

            requestAnimationFrame(() => {
                updateTypeAndStyle(this, true);
                markDirty();
            });

            return result;
        };

        nodeType.prototype.clone = function () {
            const cloned = LiteGraph.createNode(NODE_TYPE);

            if (!cloned) return null;

            cloned.pos = [
                (this.pos?.[0] ?? 0) + 20,
                (this.pos?.[1] ?? 0) + 20,
            ];

            cloned.size = [...DEFAULT_SIZE];

            setupRerouteNode(cloned, true);

            cloned.title = "";
            cloned.__neLastTitle = "";

            cloned.resolvedType = "*";
            cloned.nodeType = "*";

            cloned.properties ??= {};
            cloned.properties.autoColor = true;
            cloned.properties.resolvedTypeLabel = "";

            clearSlotColors(cloned);
            normalizeRerouteSlots(cloned, "*");
            hideSlotLabels(cloned);

            applyNodeColor(cloned, "*");

            markDirty();

            return cloned;
        };

        nodeType.prototype.findInputSlot = function () {
            return 0;
        };

        nodeType.prototype.findOutputSlot = function () {
            return 0;
        };

        const onConnectInput = nodeType.prototype.onConnectInput;
        nodeType.prototype.onConnectInput = function (inputIndex, outputType, outputSlot, outputNode, outputIndex) {
            if (inputIndex === 0) {
                disconnectExistingInput(this);
            }

            const result = onConnectInput?.apply(this, arguments) ?? true;

            if (result !== false && isRerouteNode(outputNode)) {
                setTimeout(() => updateRerouteChain(this, outputNode), 0);
            }

            return result;
        };

        const connectByType = nodeType.prototype.connectByType;
        nodeType.prototype.connectByType = function (slot, sourceNode, sourceSlotType, opts) {
            disconnectExistingInput(this);
            this.resetType?.();

            const result = connectByType?.apply(this, arguments);

            setTimeout(() => updateRerouteChain(this, isRerouteNode(sourceNode) ? sourceNode : null), 0);

            return result;
        };

        const connectByTypeOutput = nodeType.prototype.connectByTypeOutput;
        nodeType.prototype.connectByTypeOutput = function (slot, sourceNode, sourceSlotType, opts) {
            this.resetType?.();

            const result = connectByTypeOutput?.apply(this, arguments);

            setTimeout(() => updateRerouteChain(this), 0);

            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);

            this.resetType?.();

            setTimeout(() => {
                updateRerouteChain(this);
            }, 0);

            return result;
        };

        const onPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function (property, value) {
            if (this.__neApplyingAutoColor || this.__neInitializing) {
                return onPropertyChanged?.apply(this, arguments);
            }

            if (property === "color" || property === "bgcolor") {
                this.properties ??= {};
                this.properties.autoColor = false;
            }

            if (property === "title") {
                propagateTitleFrom(this);
                markDirty();
            }

            return onPropertyChanged?.apply(this, arguments);
        };

        nodeType.prototype.onDrawForeground = function (ctx, canvas) {
            drawRerouteTitle(this, ctx, canvas);
        };

        if (!nodeType.prototype.__neRerouteMenuPatched) {
            nodeType.prototype.__neRerouteMenuPatched = true;

            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;

            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                const result = getExtraMenuOptions?.apply(this, arguments);

                if (!options || !isRerouteNode(this)) {
                    return result ?? options;
                }

                const hasRerouteMenu = options.some((item) =>
                    item?.__neRerouteMenu === true
                );

                if (hasRerouteMenu) {
                    return result ?? options;
                }

                options.push(null);

                options.push({
                    __neRerouteMenu: true,
                    content: `Auto Color: ${this.properties?.autoColor !== false ? "On" : "Off"}`,
                    callback: () => {
                        this.properties ??= {};
                        this.properties.autoColor = this.properties.autoColor === false;
                        updateTypeAndStyle(this);
                        markDirty();
                    },
                });

                return result ?? options;
            };
        }
    },
});