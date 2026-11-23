import { app } from "../../scripts/app.js";
import { getInputLink, getOutputLinks } from "./base.js";
import { ExtensionNames } from "./constants.js";

function addMenuHandler(nodeType, callback) {
    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;

    nodeType.prototype.getExtraMenuOptions = function (_, options) {
        const result = getExtraMenuOptions?.apply(this, arguments);

        if (options) {
            callback.apply(this, arguments);
        }

        return result ?? options;
    };
}

function copyNodeConnections(source, destination, connectOutputs = true, copyStyle = false) {
    const graph = source.graph;

    for (const [index, input] of (source.inputs || []).entries()) {
        if (input.widget !== undefined) {
            const destinationWidget = destination.widgets?.find((widget) => widget.name === input.widget.name);
            if (destinationWidget) {
                destination.convertWidgetToInput?.(destinationWidget);
            }
        }

        const link = getInputLink(source, index);
        if (link) {
            const sourceNode = graph.getNodeById(link.origin_id);
            if (sourceNode) {
                sourceNode.connect(link.origin_slot, destination, input.name);
            }
        }
    }

    if (connectOutputs) {
        const outputLinks = {};

        for (const [index, output] of (source.outputs || []).entries()) {
            outputLinks[output.name] = getOutputLinks(source, index);
        }

        for (let index = 0; index < (destination.outputs || []).length; index++) {
            const output = destination.outputs[index];
            const links = outputLinks[output.name];

            if (!links) continue;

            for (const link of links) {
                const targetNode = graph.getNodeById(link.target_id);
                if (targetNode) {
                    destination.connect(index, targetNode, link.target_slot);
                }
            }
        }
    }

    if (copyStyle) {
        destination.color = source.color;
        destination.bgcolor = source.bgcolor;
        destination.size = [
            Math.max(source.size?.[0] ?? 0, destination.size?.[0] ?? 0),
            Math.max(source.size?.[1] ?? 0, destination.size?.[1] ?? 0),
        ];
    }

    graph.afterChange();
    app.canvas?.setDirty?.(true, true);
}

app.registerExtension({
    name: ExtensionNames.NodeFixer,

    beforeRegisterNodeDef(nodeType) {
        addMenuHandler(nodeType, function (_, options) {
            options.push({
                content: "Fix node",
                callback: () => {
                    const nodeClass = nodeType.comfyClass ?? nodeType.type;
                    const newNode = LiteGraph.createNode(nodeClass);

                    if (!newNode) {
                        console.error("[NE.NodeFixer] Failed to create node", nodeClass);
                        return;
                    }

                    const graph = this.graph;

                    newNode.pos = [...this.pos];
                    graph.add(newNode, false);

                    copyNodeConnections(this, newNode, true, true);

                    graph.remove(this);
                    graph.afterChange();
                    app.canvas?.setDirty?.(true, true);
                },
            });
        });
    },
});