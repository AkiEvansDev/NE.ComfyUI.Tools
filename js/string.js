import { app } from "../../scripts/app.js";
import { ExtensionNames, NodeNames } from "./constants.js";
import { removeUnusedInputsFromEnd } from "./base.js";

function addAnyInput(node, count = 1) {
    node.inputs ??= [];

    for (let index = 0; index < count; index++) {
        const number = node.inputs.filter((input) => input.type === "*").length + 1;
        node.addInput(String(number).padStart(2, "0"), "*");
    }
}

function refreshInputs(node, count = 1) {
    if (!node.inputs) {
        return;
    }

    const oldWidth = node.size?.[0] ?? 0;

    removeUnusedInputsFromEnd(node, 1, /^\d+$/);
    addAnyInput(node, count);

    node.setSize?.([
        Math.max(oldWidth, node.size?.[0] ?? oldWidth),
        node.size?.[1] ?? 0,
    ]);
}

app.registerExtension({
    name: ExtensionNames.ToStringConcat,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NodeNames.ToStringConcat) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            refreshInputs(this, 2);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;

        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            refreshInputs(this, 1);
            return result;
        };
    },
});