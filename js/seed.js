import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { EventNames, ExtensionNames, NE_PREFIX, WidgetNames } from "./constants.js";
import { neApi } from "./api.js";

const SEED_FEEDBACK_EVENT = EventNames.SeedFeedback;

// Found by the widget rather than by a list of node names: the seed keeps its own state on the python
// side and only adopts a typed value once this callback has reported the edit, so a node left off a
// list silently ignores whatever is typed into it. DetailerStep was missing for exactly that reason.
function hasSeedWidget(nodeData) {
    const inputs = { ...(nodeData?.input?.required ?? {}), ...(nodeData?.input?.optional ?? {}) };
    const spec = inputs[WidgetNames.SeedValue];

    // A forceInput seed is a link, not something anyone types into: ChangeSamplerConfig overrides the
    // value from upstream and keeps no state of its own, so it needs none of this.
    return Array.isArray(spec) && !spec[1]?.forceInput;
}

function getNodeById(nodeId) {
    return app.graph?._nodes_by_id?.[nodeId];
}

function findWidget(node, name) {
    return node?.widgets?.find((widget) => widget.name === name);
}

function updateSeedNode(event) {
    const node = getNodeById(event.detail.node_id);
    const widget = findWidget(node, WidgetNames.SeedValue);

    if (widget) {
        widget.value = event.detail.seed;
        node?.setDirtyCanvas?.(true, true);
    }
}

if (!window.__neSeedListenersRegistered) {
    api.addEventListener(SEED_FEEDBACK_EVENT, updateSeedNode);
    window.__neSeedListenersRegistered = true;
}

app.registerExtension({
    name: ExtensionNames.Seed,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!String(nodeData.name).startsWith(NE_PREFIX) || !hasSeedWidget(nodeData)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const widget = findWidget(this, WidgetNames.SeedValue);
            if (widget) {
                widget.callback = () => {
                    neApi.resetNode(this.id).catch((error) => {
                        console.error("[NE.Seed] Failed to reset node:", error);
                    });
                };
            }

            return result;
        };
    },
});