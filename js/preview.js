import { app } from "../../scripts/app.js";
import { ExtensionNames, NodeNames, SettingNames } from "./constants.js";

const DEFAULT_MAX_HEIGHT = 640;

function maxPreviewHeight() {
    const value = app.extensionManager?.setting?.get(SettingNames.PreviewMaxHeight);
    return typeof value === "number" ? value : DEFAULT_MAX_HEIGHT;
}

// The stock preview widget reports its minimum height as node_width / media_aspect and caps it at nothing
// (`fitDimensionsToNodeWidth` in the frontend bundle), so an 832-tall clip pins the node to a height it
// cannot be dragged out of — and widening the node only makes it taller still. Bounding that minimum is
// all it takes: the media itself is `object-fit: contain`, so it letterboxes rather than crops.
function clampPreviewWidgets(node) {
    for (const widget of node.widgets ?? []) {
        if (widget.__ne_previewClamped || typeof widget.computeLayoutSize !== "function") {
            continue;
        }

        const computeLayoutSize = widget.computeLayoutSize.bind(widget);

        // Read the setting per call, not per patch, so changing it takes effect without a reload.
        widget.computeLayoutSize = (target) => {
            const size = computeLayoutSize(target);
            return { ...size, minHeight: Math.min(size?.minHeight ?? 0, maxPreviewHeight()) };
        };

        widget.__ne_previewClamped = true;
    }
}

app.registerExtension({
    name: ExtensionNames.Preview,

    settings: [
        {
            id: SettingNames.PreviewMaxHeight,
            name: "Video preview max height",
            category: ["NE Tools", "Preview", "Video preview max height"],
            tooltip: "How tall a Preview Video node may grow to fit its clip, in pixels. The node can still be "
                + "resized by hand; this only bounds what the preview demands. Image previews are drawn on the "
                + "node background and were never affected.",
            type: "slider",
            attrs: { min: 120, max: 1200, step: 10 },
            defaultValue: DEFAULT_MAX_HEIGHT,
            onChange: () => app.canvas?.setDirty(true, true),
        },
    ],

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NodeNames.PreviewVideo) {
            return;
        }

        const onDrawBackground = nodeType.prototype.onDrawBackground;

        nodeType.prototype.onDrawBackground = function () {
            const result = onDrawBackground?.apply(this, arguments);
            // The widget does not exist until the media has loaded, and that happens inside this very
            // call — so the patch is applied on the way out rather than at node creation.
            clampPreviewWidgets(this);
            return result;
        };
    },
});
