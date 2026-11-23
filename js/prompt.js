import { app } from "../../scripts/app.js";
import { ExtensionNames, NodeNames } from "./constants.js";

const PROMPT_NODE_NAMES = new Set([
    NodeNames.PromptConfig,
    NodeNames.VideoModelConfig,
    NodeNames.DetailerStep,
]);

const PROMPT_WIDGET_NAMES = [
    "positive",
    "negative",
    "posture",
    "background",
    "positive_style",
    "negative_style",
    "region_positive",
    "region_negative",
];

app.registerExtension({
    name: ExtensionNames.Prompt,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PROMPT_NODE_NAMES.has(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this);

            for (const name of PROMPT_WIDGET_NAMES) {
                const widget = this.widgets?.find((w) => w.name === name);

                if (widget?.element) {
                    widget.element.style.border = `0.5px solid ${LiteGraph.WIDGET_OUTLINE_COLOR}`;
                    widget.element.style.borderRadius = "4px";
                }
            }

            return result;
        };
    },
});