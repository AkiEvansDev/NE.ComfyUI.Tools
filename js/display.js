import { app } from "../../scripts/app.js";
import { EventNames, ExtensionNames, NodeNames, WidgetNames } from "./constants.js";

const DISPLAY_WIDGET_MIN_WIDTH = 80;
const DISPLAY_WIDGET_MIN_HEIGHT = 80;
const DISPLAY_NODE_MIN_WIDTH = 260;
const DISPLAY_NODE_MIN_HEIGHT = 180;

function mediaUrl(filename) {
    // The temp file carries one name per node, so this url would be identical between runs — and assigning
    // an src an element already holds is a no-op, no fetch and no repaint. The preview would freeze on the
    // first run after every page load. The token is the only thing that makes it reload.
    return `/view?filename=${encodeURIComponent(filename)}&type=temp&ne=${Date.now()}`;
}

function setVisible(element, visible) {
    if (!element) return;
    element.style.display = visible ? "block" : "none";
}

function clearMedia(node) {
    if (!node?.showVideoElement) return;
    node.showVideoElement.pause?.();
    node.showVideoElement.removeAttribute("src");
    node.showVideoElement.load?.();
}

function updateDisplayNode(detail) {
    const node = app.graph.getNodeById(Number(detail.node));
    if (!node?.showTextElement || !node?.showImageElement || !node?.showVideoElement) return;

    const hasVideo = !!detail.video;
    const hasImage = !!detail.image;
    const hasText = !hasVideo && !hasImage;

    setVisible(node.showTextElement, hasText);
    setVisible(node.showImageElement, hasImage);
    setVisible(node.showVideoElement, hasVideo);

    if (hasText) {
        clearMedia(node);
        node.showTextElement.value = detail.text ?? "";
    } else if (hasImage) {
        clearMedia(node);
        node.showImageElement.src = mediaUrl(detail.image);
    } else if (hasVideo) {
        node.showVideoElement.src = mediaUrl(detail.video);
        node.showVideoElement.load?.();
    }

    node.setDirtyCanvas?.(true, true);
}

function styleDisplayContainer(container) {
    container.style.width = `${DISPLAY_WIDGET_MIN_WIDTH}px`;
    container.style.height = `${DISPLAY_WIDGET_MIN_HEIGHT}px`;
    container.style.boxSizing = "border-box";
    container.style.overflow = "hidden";
    container.style.minWidth = "0";
    container.style.minHeight = `${DISPLAY_WIDGET_MIN_HEIGHT}px`;
    container.style.borderRadius = "6px";
}

function styleTextElement(text) {
    text.className = "comfy-multiline-input";
    text.readOnly = true;
    text.value = "";
    text.style.width = "100%";
    text.style.height = "100%";
    text.style.boxSizing = "border-box";
    text.style.resize = "none";
    text.style.overflow = "auto";
}

function styleMediaElement(element) {
    element.style.display = "none";
    element.style.width = "100%";
    element.style.height = "100%";
    element.style.boxSizing = "border-box";
    element.style.objectFit = "contain";
    element.style.overflow = "hidden";
}

function setElementSize(element, width, height) {
    if (!element) return;
    element.style.width = `${width}px`;
    element.style.height = `${height}px`;
}

function computeDisplayWidgetSize(node, width) {
    const nodeWidth = width ?? node.size?.[0] ?? DISPLAY_NODE_MIN_WIDTH;
    const nodeHeight = node.size?.[1] ?? DISPLAY_NODE_MIN_HEIGHT;
    const widgetWidth = Math.max(DISPLAY_WIDGET_MIN_WIDTH, nodeWidth - 20);
    const widgetHeight = Math.max(DISPLAY_WIDGET_MIN_HEIGHT, nodeHeight - 80);

    setElementSize(node.showContainerElement, widgetWidth, widgetHeight);

    return [widgetWidth, widgetHeight];
}

function fitDisplayNodeSize(node) {
    if (!node?.size) return;

    node.setSize?.([
        Math.max(node.size?.[0] ?? 0, DISPLAY_NODE_MIN_WIDTH),
        Math.max(node.size?.[1] ?? 0, DISPLAY_NODE_MIN_HEIGHT),
    ]);

    computeDisplayWidgetSize(node, node.size?.[0]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: ExtensionNames.DisplayAny,

    async setup() {
        app.api.addEventListener(EventNames.DisplayFeedback, ({ detail }) => updateDisplayNode(detail));
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NodeNames.DisplayAny) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const container = document.createElement("div");
            styleDisplayContainer(container);

            const text = document.createElement("textarea");
            styleTextElement(text);

            const image = document.createElement("img");
            styleMediaElement(image);

            const video = document.createElement("video");
            styleMediaElement(video);
            video.controls = true;
            video.loop = true;
            video.muted = true;
            video.playsInline = true;

            container.appendChild(text);
            container.appendChild(image);
            container.appendChild(video);

            this.showContainerElement = container;
            this.showTextElement = text;
            this.showImageElement = image;
            this.showVideoElement = video;

            // Two flags, because the frontend checks two places: `options.serialize` decides whether the
            // widget is sent with the prompt, the top-level one whether it takes a slot in widgets_values.
            const widget = this.addDOMWidget(WidgetNames.Output, WidgetNames.Output, container, {
                serialize: false,
            });
            widget.serialize = false;

            widget.computeSize = (width) => computeDisplayWidgetSize(this, width);
            widget.element = container;

            requestAnimationFrame(() => fitDisplayNodeSize(this));

            return result;
        };
    },
});
