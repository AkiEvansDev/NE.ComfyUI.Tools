import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { EventNames, ExtensionNames, NodeNames } from "./constants.js";
import { neApi } from "./api.js";

const RANGE_FEEDBACK_EVENT = EventNames.RangeFeedback;
const XY_RANGE_FEEDBACK_EVENT = EventNames.XYRangeFeedback;
const XYZ_RANGE_FEEDBACK_EVENT = EventNames.XYZRangeFeedback;

const RANGE_NODE = NodeNames.Range;
const XY_RANGE_NODE = NodeNames.XYRange;
const XYZ_RANGE_NODE = NodeNames.XYZRange;
const RANGE_LIST_NODE = NodeNames.RangeList;

const RANGE_NODE_NAMES = new Set([
    RANGE_NODE,
    XY_RANGE_NODE,
    XYZ_RANGE_NODE,
    RANGE_LIST_NODE,
]);

function getNodeById(nodeId) {
    return app.graph?._nodes_by_id?.[nodeId];
}

function findWidget(node, name) {
    return node?.widgets?.find((widget) => widget.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = findWidget(node, name);

    if (widget) {
        widget.value = value;
    }
}

function updateRangeNode(event) {
    const node = getNodeById(event.detail.node_id);

    if (!node) {
        return;
    }

    setWidgetValue(node, "current", event.detail.current);
    setWidgetValue(node, "start", event.detail.start);
    setWidgetValue(node, "end", event.detail.end);
}

function updateXYRangeNode(event) {
    const node = getNodeById(event.detail.node_id);

    if (!node) {
        return;
    }

    setWidgetValue(node, "x", event.detail.x);
    setWidgetValue(node, "y", event.detail.y);
    setWidgetValue(node, "x_start", event.detail.x_start);
    setWidgetValue(node, "x_end", event.detail.x_end);
    setWidgetValue(node, "y_start", event.detail.y_start);
    setWidgetValue(node, "y_end", event.detail.y_end);
}

function updateXYZRangeNode(event) {
    const node = getNodeById(event.detail.node_id);

    if (!node) {
        return;
    }

    setWidgetValue(node, "x", event.detail.x);
    setWidgetValue(node, "y", event.detail.y);
    setWidgetValue(node, "z", event.detail.z);
    setWidgetValue(node, "x_start", event.detail.x_start);
    setWidgetValue(node, "x_end", event.detail.x_end);
    setWidgetValue(node, "y_start", event.detail.y_start);
    setWidgetValue(node, "y_end", event.detail.y_end);
    setWidgetValue(node, "z_start", event.detail.z_start);
    setWidgetValue(node, "z_end", event.detail.z_end);
}

if (!window.__neRangeListenersRegistered) {
    api.addEventListener(RANGE_FEEDBACK_EVENT, updateRangeNode);
    api.addEventListener(XY_RANGE_FEEDBACK_EVENT, updateXYRangeNode);
    api.addEventListener(XYZ_RANGE_FEEDBACK_EVENT, updateXYZRangeNode);
    window.__neRangeListenersRegistered = true;
}

function reset(type, node) {
    neApi.resetNode(node.id).catch((error) => {
        console.error("[NE.Range] Failed to reset node:", error);
    });

    if (type === RANGE_NODE) {
        const current = findWidget(node, "current");
        const start = findWidget(node, "start");

        if (current && start) {
            current.value = start.value - 1;
        }
    } else if (type === XY_RANGE_NODE || type === XYZ_RANGE_NODE) {
        const x = findWidget(node, "x");
        const xStart = findWidget(node, "x_start");
        const y = findWidget(node, "y");
        const yStart = findWidget(node, "y_start");
        const z = findWidget(node, "z");
        const zStart = findWidget(node, "z_start");

        if (x && xStart) {
            x.value = xStart.value - 1;
        }

        if (y && yStart) {
            y.value = yStart.value - 1;
        }

        if (z && zStart) {
            z.value = zStart.value - 1;
        }
    }
}

function queue(type, node) {
    reset(type, node);

    let batchSize = 0;

    if (type === RANGE_NODE) {
        const start = findWidget(node, "start");
        const end = findWidget(node, "end");

        if (start && end) {
            batchSize = Math.abs(end.value - start.value) + 1;
        }
    } else if (type === XY_RANGE_NODE || type === XYZ_RANGE_NODE) {
        const xStart = findWidget(node, "x_start");
        const xEnd = findWidget(node, "x_end");
        const yStart = findWidget(node, "y_start");
        const yEnd = findWidget(node, "y_end");
        const zStart = findWidget(node, "z_start");
        const zEnd = findWidget(node, "z_end");

        if (xStart && xEnd && yStart && yEnd) {
            const xCount = Math.abs(xEnd.value - xStart.value) + 1;
            const yCount = Math.abs(yEnd.value - yStart.value) + 1;
            const zCount = zStart && zEnd ? Math.abs(zEnd.value - zStart.value) + 1 : 1;
            batchSize = xCount * yCount * zCount;
        }
    }

    if (batchSize > 0) {
        app.queuePrompt?.(0, batchSize);
    }
}

app.registerExtension({
    name: ExtensionNames.Range,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!RANGE_NODE_NAMES.has(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            if (nodeData.name === RANGE_NODE || nodeData.name === RANGE_LIST_NODE) {
                const current = findWidget(this, "current");

                if (current) {
                    current.callback = () => {
                        neApi.resetNode(this.id);
                    };
                }
            } else if (nodeData.name === XY_RANGE_NODE || nodeData.name === XYZ_RANGE_NODE) {
                const x = findWidget(this, "x");
                const y = findWidget(this, "y");
                const z = findWidget(this, "z");

                if (x) {
                    x.callback = () => {
                        neApi.resetNode(this.id);
                    };
                }

                if (y) {
                    y.callback = () => {
                        neApi.resetNode(this.id);
                    };
                }

                if (z) {
                    z.callback = () => {
                        neApi.resetNode(this.id);
                    };
                }
            }

            if (nodeData.name !== RANGE_LIST_NODE) {
                if (!this.__neRangeButtonsAdded) {
                    this.__neRangeButtonsAdded = true;
                    
                    this.addWidget("button", "Queue Full", "QueueButton", (source, canvas, node) => {
                        queue(nodeData.name, node);
                    });

                    this.addWidget("button", "Reset Range", "ResetButton", (source, canvas, node) => {
                        reset(nodeData.name, node);
                    });
                }
            }

            return result;
        };
    },
});