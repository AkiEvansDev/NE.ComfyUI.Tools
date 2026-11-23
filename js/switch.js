import { app } from "../../scripts/app.js";
import { ExtensionNames, NodeNames } from "./constants.js";
import { followConnectionUntilType, getInputLink, getOutputLinks, removeUnusedInputsFromEnd } from "./base.js";

function addAnyInput(node, count = 1, type = "*") {
    node.inputs ??= [];
    for (let index = 0; index < count; index++) {
        node.addInput(String(node.inputs.length + 1).padStart(2, "0"), type);
    }
}

app.registerExtension({
    name: ExtensionNames.AnySwitch,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NodeNames.AnySwitch && nodeData.name !== NodeNames.AnyTypeSwitch) {
            return;
        }

        if (nodeData.name === NodeNames.AnySwitch) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                removeUnusedInputsFromEnd(this, 1);
                addAnyInput(this, 2);

                return result;
            };
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;

        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
            const result = onConnectionsChange?.apply(this, arguments);

            let connectedType = followConnectionUntilType(this, "INPUT", undefined, true);
            if (!connectedType) {
                connectedType = followConnectionUntilType(this, "OUTPUT", undefined, true);
            }

            if (nodeData.name === NodeNames.AnySwitch) {
                removeUnusedInputsFromEnd(this, 1);
                addAnyInput(this, 1, connectedType?.type ?? "*");
            }

            this.nodeType = connectedType?.type ?? "*";

            for (const [index, input] of (this.inputs || []).entries()) {
                // Widget-backed inputs have a type of their own — AnyTypeSwitch picks with a BOOLEAN
                // condition, and retyping that to the value type breaks what can be plugged into it.
                if (input.widget) {
                    continue;
                }

                input.type = this.nodeType;

                const link = getInputLink(this, index);
                if (link) {
                    link.type = this.nodeType;
                }
            }

            for (const [index, output] of (this.outputs || []).entries()) {
                output.type = this.nodeType;
                output.label = Array.isArray(this.nodeType) || String(this.nodeType).includes(",")
                    ? connectedType?.label ?? String(this.nodeType)
                    : String(this.nodeType);

                for (const link of getOutputLinks(this, index)) {
                    link.type = this.nodeType;
                }
            }

            return result;
        };
    },
});