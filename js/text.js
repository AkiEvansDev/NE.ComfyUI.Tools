import { app } from "../../scripts/app.js";
import { ExtensionNames, NodeNames, WidgetNames } from "./constants.js";

app.registerExtension({
    name: ExtensionNames.Text,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NodeNames.Text) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const input = this.widgets?.find((w) => w.name === WidgetNames.Value)?.element;
            if (input) {
                input.style.border = `0.5px solid ${LiteGraph.WIDGET_OUTLINE_COLOR}`;
                input.style.borderRadius = "4px";
            }

            return result;
        };
    },
});