import { app } from "../../scripts/app.js";
import { ExtensionNames } from "./constants.js";

if (!LGraphCanvas.prototype.__ne_adjustMouseEventPatched) {
    const adjustMouseEvent = LGraphCanvas.prototype.adjustMouseEvent;

    LGraphCanvas.prototype.adjustMouseEvent = function (e) {
        const result = adjustMouseEvent?.apply(this, arguments);
        app.lastAdjustedMouseEvent = e;
        return result;
    };

    LGraphCanvas.prototype.__ne_adjustMouseEventPatched = true;
}

export function $el(selector, attrs = {}, children = []) {
    const element = document.createElement(selector.split(/[.#\[]/)[0] || "div");

    const classMatch = selector.match(/\.([a-zA-Z0-9_-]+)/g);
    if (classMatch) {
        element.className = classMatch.map((x) => x.slice(1)).join(" ");
    }

    for (const [key, value] of Object.entries(attrs)) {
        if (key === "parent") {
            value.appendChild(element);
        } else if (key === "text" || key === "textContent") {
            element.textContent = value;
        } else if (key.startsWith("on") && typeof value === "function") {
            element.addEventListener(key.slice(2), value);
        } else {
            element[key] = value;
        }
    }

    const childList = Array.isArray(children) ? children : [children];
    for (const child of childList) {
        if (child instanceof Node) {
            element.appendChild(child);
        }
    }

    return element;
}

// Frontend 1.57 deprecates reading a slot's `link` / `links`: links are read from the node's own graph instead,
// which is also the graph to look connected nodes up in — a subgraph's ids overlap the root graph's.
export function getInputLink(node, slotNum) {
    return node.graph && slotNum < (node.inputs?.length ?? 0) ? node.getInputLink(slotNum) : null;
}

// There is no public read for an output's links, so they are picked out of the graph's link store.
export function getOutputLinks(node, slotNum) {
    const links = [];
    for (const link of node.graph?.links?.values() ?? []) {
        if (link.origin_id === node.id && link.origin_slot === slotNum) {
            links.push(link);
        }
    }
    return links;
}

function getSlotLinks(node, slotNum, dir) {
    if (dir === "OUTPUT") {
        return getOutputLinks(node, slotNum);
    }
    const link = getInputLink(node, slotNum);
    return link ? [link] : [];
}

function getTypeFromSlot(node, slotNum, dir, skipSelf = false) {
    const slot = (dir === "OUTPUT" ? node.outputs : node.inputs)?.[slotNum];
    if (!skipSelf && slot?.type != null && slot.type != "*") {
        return { type: slot.type, label: slot.label, name: slot.name };
    }
    for (const link of getSlotLinks(node, slotNum, dir)) {
        const connectedId = dir == "OUTPUT" ? link.target_id : link.origin_id;
        const connectedSlotNum = dir == "OUTPUT" ? link.target_slot : link.origin_slot;

        const connectedNode = node.graph.getNodeById(connectedId);
        if (!connectedNode) continue;

        const connectedSlots = dir === "OUTPUT" ? connectedNode.inputs : connectedNode.outputs;
        if (!connectedSlots) continue;

        const connectedSlot = connectedSlots[connectedSlotNum];
        if (connectedSlot?.type != null && connectedSlot.type != "*") {
            return { type: connectedSlot.type, label: connectedSlot.label, name: connectedSlot.name };
        }
        else if (connectedSlot?.type == "*") {
            return followConnectionUntilType(connectedNode, dir);
        }
    }
    return null;
}

export function followConnectionUntilType(node, dir, slotNum, skipSelf = false) {
    const slots = dir === "OUTPUT" ? node.outputs : node.inputs;
    if (!slots || !slots.length) {
        return null;
    }
    let type = null;
    if (slotNum != null) {
        if (!slots[slotNum]) {
            return null;
        }
        type = getTypeFromSlot(node, slotNum, dir, skipSelf);
    }
    else {
        for (let index = 0; index < slots.length; index++) {
            type = getTypeFromSlot(node, index, dir, skipSelf);
            if (type) {
                break;
            }
        }
    }
    return type;
}

export function removeUnusedInputsFromEnd(node, minNumber = 1, nameMatch) {
    for (let i = node.inputs.length - 1; i >= minNumber; i--) {
        if (!node.isInputConnected(i)) {
            if (!nameMatch || nameMatch.test(node.inputs[i].name)) {
                node.removeInput(i);
            }
            continue;
        }
        break;
    }
}

export function moveArrayItem(arr, itemOrFrom, to) {
    const from = typeof itemOrFrom === "number" ? itemOrFrom : arr.indexOf(itemOrFrom);

    if (from < 0 || to < 0 || from === to) {
        return;
    }

    arr.splice(to, 0, arr.splice(from, 1)[0]);
}

export function removeArrayItem(arr, itemOrIndex) {
    const index = typeof itemOrIndex === "number" ? itemOrIndex : arr.indexOf(itemOrIndex);

    if (index < 0) {
        return;
    }

    arr.splice(index, 1);
}

export function getUrl(path, baseUrl) {
    if (baseUrl) {
        return new URL(path, baseUrl).toString();
    } else {
        return new URL("../" + path, import.meta.url).toString();
    }
}

export function addStylesheet(url) {
    if (url.endsWith(".js")) {
        url = url.substr(0, url.length - 2) + "css";
    }
    $el("link", {
        parent: document.head,
        rel: "stylesheet",
        type: "text/css",
        href: url.startsWith("http") ? url : getUrl(url),
    });
}

if (!LiteGraph.__ne_isValidConnectionPatched) {
    const oldIsValidConnection = LiteGraph.isValidConnection;

    LiteGraph.isValidConnection = function (typeA, typeB) {
        let isValid = oldIsValidConnection.call(this, typeA, typeB);

        if (!isValid) {
            typeA = String(typeA);
            typeB = String(typeB);

            const areCombos =
                (typeA.includes(",") && typeB === "COMBO") ||
                (typeA === "COMBO" && typeB.includes(","));

            isValid = areCombos;
        }

        return isValid;
    };

    LiteGraph.__ne_isValidConnectionPatched = true;
}

export class BaseWidget {
    constructor(name) {
        this.last_y = 0;
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.hitAreas = {};
        this.downedHitAreasForMove = [];
        this.name = name;
    }
    clickWasWithinBounds(pos, bounds) {
        let xStart = bounds[0];
        let xEnd = xStart + (bounds.length > 2 ? bounds[2] : bounds[1]);
        const clickedX = pos[0] >= xStart && pos[0] <= xEnd;
        if (bounds.length === 2) {
            return clickedX;
        }
        return clickedX && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3];
    }
    mouse(event, pos, node) {
        var _a, _b, _c;
        const canvas = app.canvas;
        if (event.type == "pointerdown") {
            this.mouseDowned = [...pos];
            this.isMouseDownedAndOver = true;
            this.downedHitAreasForMove.length = 0;
            let anyHandled = false;
            for (const part of Object.values(this.hitAreas)) {
                if ((part.onDown || part.onMove) && this.clickWasWithinBounds(pos, part.bounds)) {
                    if (part.onMove) {
                        this.downedHitAreasForMove.push(part);
                    }
                    if (part.onDown) {
                        const thisHandled = part.onDown.apply(this, [event, pos, node, part]);
                        anyHandled = anyHandled || thisHandled == true;
                    }
                }
            }
            return (_a = this.onMouseDown(event, pos, node)) !== null && _a !== void 0 ? _a : anyHandled;
        }
        if (event.type == "pointerup") {
            if (!this.mouseDowned)
                return true;
            this.downedHitAreasForMove.length = 0;
            this.cancelMouseDown();
            let anyHandled = false;
            for (const part of Object.values(this.hitAreas)) {
                if (part.onUp && this.clickWasWithinBounds(pos, part.bounds)) {
                    const thisHandled = part.onUp.apply(this, [event, pos, node, part]);
                    anyHandled = anyHandled || thisHandled == true;
                }
            }
            return (_b = this.onMouseUp(event, pos, node)) !== null && _b !== void 0 ? _b : anyHandled;
        }
        if (event.type == "pointermove") {
            this.isMouseDownedAndOver = !!this.mouseDowned;
            if (this.mouseDowned &&
                (pos[0] < 15 ||
                    pos[0] > node.size[0] - 15 ||
                    pos[1] < this.last_y ||
                    pos[1] > this.last_y + LiteGraph.NODE_WIDGET_HEIGHT)) {
                this.isMouseDownedAndOver = false;
            }
            for (const part of this.downedHitAreasForMove) {
                part.onMove.apply(this, [event, pos, node, part]);
            }
            return (_c = this.onMouseMove(event, pos, node)) !== null && _c !== void 0 ? _c : true;
        }
        return false;
    }
    cancelMouseDown() {
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.downedHitAreasForMove.length = 0;
    }
    onMouseDown(event, pos, node) {
        return;
    }
    onMouseUp(event, pos, node) {
        return;
    }
    onMouseMove(event, pos, node) {
        return;
    }
}

function styleNode(node, markDirty = true) {
    node.shape = LiteGraph.BOX_SHAPE;
    node.constructor.shape = LiteGraph.BOX_SHAPE;

    node.color = "#353535";
    node.bgcolor = "#212121";

    if (markDirty) {
        app.canvas?.setDirty?.(true, true);
    }
}

app.registerExtension({
    name: ExtensionNames.NodeStyle,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Not on the prototype: `shape` is an accessor over the per-instance `_state`, which the prototype lacks.
        nodeType.shape = LiteGraph.BOX_SHAPE;

        const original = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            styleNode(this);
            return result;
        };
    },

    nodeCreated(node) {
        styleNode(node);
    },

    async loadedGraphNode(node) {
        styleNode(node, false);
    }
});