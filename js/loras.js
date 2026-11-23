import { app } from "../../scripts/app.js";
import { neApi } from "./api.js";
import { ExtensionNames, NodeNames, WidgetNames } from "./constants.js";
import { BaseWidget, moveArrayItem, removeArrayItem } from "./base.js";
import { isLowQuality, fitString, drawTogglePart, drawRoundedRectangle, drawNumberWidgetPart } from "./draw.js";

const DEFAULT_LORA_WIDGET_DATA = {
    on: true,
    lora: null,
    strength: 1,
};

// The expert loader shows one entry per Wan 2.2 high/low pair, so the chooser has to know whose it is.
function loraNamesFor(node) {
    const type = node?.comfyClass ?? node?.type;
    return type === NodeNames.WanExpertLoras ? neApi.getExpertLoras() : neApi.getLoras();
}

async function showLoraChooser(event, callback, parentMenu, node) {
    const loras = ["None", ...(await loraNamesFor(node))];

    new LiteGraph.ContextMenu(loras, {
        event,
        parentMenu,
        title: "Choose Lora",
        scale: Math.max(1, app.canvas?.ds?.scale ?? 1),
        className: "dark",
        callback,
    });
}

function showLoraInfoDialog(title, content, previewUrl = null) {
    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed;
        inset: 0;
        z-index: 100000;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0, 0, 0, 0.55);
    `;

    const dialog = document.createElement("div");
    dialog.style.cssText = `
        width: min(980px, calc(100vw - 48px));
        max-height: min(720px, calc(100vh - 48px));
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 16px;
        border-radius: 8px;
        background: #202020;
        color: #ddd;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
        font-family: Arial, sans-serif;
    `;

    const header = document.createElement("div");
    header.style.cssText = `
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    `;

    const heading = document.createElement("div");
    heading.textContent = title || "LoRA info";
    heading.style.cssText = `
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 16px;
        font-weight: 600;
    `;

    const closeButton = document.createElement("button");
    closeButton.textContent = "Close";
    closeButton.style.cssText = `
        cursor: pointer;
        padding: 6px 12px;
        border: 1px solid #555;
        border-radius: 6px;
        background: #303030;
        color: #ddd;
    `;

    const body = document.createElement("div");
    body.style.cssText = `
        display: grid;
        grid-template-columns: ${previewUrl ? "220px minmax(0, 1fr)" : "1fr"};
        gap: 12px;
        min-height: ${previewUrl ? "220px" : "0"};
        min-width: 0;
        overflow: hidden;
    `;

    if (previewUrl) {
        const preview = document.createElement("img");
        preview.src = previewUrl;
        preview.alt = "LoRA preview";
        preview.style.cssText = `
            width: 220px;
            height: 220px;
            object-fit: cover;
            border-radius: 6px;
            background: #111;
        `;
        preview.addEventListener("error", () => {
            preview.remove();
            body.style.gridTemplateColumns = "1fr";
            body.style.minHeight = "0";
        });
        body.append(preview);
    }

    const text = document.createElement("pre");
    text.textContent = content || "";
    text.style.cssText = `
        min-width: 0;
        margin: 0;
        padding: 12px;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
        border-radius: 6px;
        background: #111;
        color: #ddd;
        font: 13px/1.45 monospace;
    `;

    const close = () => {
        document.removeEventListener("keydown", onKeyDown);
        overlay.remove();
    };

    const onKeyDown = (event) => {
        if (event.key === "Escape") {
            close();
        }
    };

    closeButton.addEventListener("click", close);
    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) {
            close();
        }
    });
    document.addEventListener("keydown", onKeyDown);

    header.append(heading, closeButton);
    body.append(text);
    dialog.append(header, body);
    overlay.append(dialog);
    document.body.append(overlay);
}

async function buildLoraSlotMenuItems(node, widget, index) {
    const canMoveUp = node.widgets[index - 1]?.name?.startsWith(WidgetNames.LoraPrefix);
    const canMoveDown = node.widgets[index + 1]?.name?.startsWith(WidgetNames.LoraPrefix);

    const menuItems = [
        {
            content: "Move Up",
            disabled: !canMoveUp,
            callback: () => moveArrayItem(node.widgets, widget, index - 1),
        },
        {
            content: "Move Down",
            disabled: !canMoveDown,
            callback: () => moveArrayItem(node.widgets, widget, index + 1),
        },
    ];

    const lora = widget.value?.lora;
    if (lora) {
        try {
            const info = await neApi.getLoraInfo(lora);
            if (info?.has_info || info?.has_preview) {
                menuItems.push({
                    content: "Show info",
                    callback: () => showLoraInfoDialog(
                        info.filename || lora,
                        info.content,
                        info.has_preview ? neApi.getLoraPreviewUrl(lora) : null,
                    ),
                });
            }
        } catch (error) {
            console.error(`[NE.LorasLoader] Failed to load LoRA info for ${lora}:`, error);
        }
    }

    menuItems.push({
        content: "Remove",
        callback: () => removeArrayItem(node.widgets, widget),
    });

    return menuItems;
}

function removeWidget(node, widgetOrSlot) {
    if (!node.widgets) return;

    if (typeof widgetOrSlot === "number") {
        node.widgets.splice(widgetOrSlot, 1);
        return;
    }

    if (!widgetOrSlot) {
        return;
    }

    const index = node.widgets.indexOf(widgetOrSlot);
    if (index > -1) {
        node.widgets.splice(index, 1);
    }
}

function addNewLoraWidget(node, lora) {
    node.loraWidgetsCounter++;

    const widget = node.addCustomWidget(new LoraLoaderWidget(`${WidgetNames.LoraPrefix}${node.loraWidgetsCounter}`));

    if (lora) {
        widget.setLora(lora);
    }

    if (node.addLoraWidget) {
        moveArrayItem(node.widgets, widget, node.widgets.indexOf(node.addLoraWidget));
    }

    return widget;
}

function addNonLoraWidgets(node) {
    if (node.addLoraWidget && node.widgets?.includes(node.addLoraWidget)) {
        return;
    }

    node.addLoraWidget = node.addWidget("button", "Add Lora", "AddLoraButton", (source, canvas, node, pos, event) => {
        showLoraChooser(
            event,
            (value) => {
                if (typeof value !== "string" || value === "None") {
                    return;
                }

                addNewLoraWidget(node, value);

                const computed = node.computeSize();
                node.size[1] = Math.max(node.size[1], computed[1]);
                node.setDirtyCanvas(true, true);
            },
            null,
            node,
        );
    });

    // The button must never take a slot in widgets_values. LiteGraph writes those by widget index and
    // applies them back the same way, so a lora value lands on the button while it is the only widget
    // present, and comes back as a phantom copy of the first lora on the next save — one more each cycle.
    node.addLoraWidget.serialize = false;
    // The prompt builder checks `options.serialize` instead, so without this the button is sent along
    // with the loras on every run and only dropped because the loader ignores names not starting lora_.
    node.addLoraWidget.options = { ...node.addLoraWidget.options, serialize: false };

    moveArrayItem(node.widgets, node.addLoraWidget, node.widgets.length - 1);
}

class LoraLoaderWidget extends BaseWidget {
    constructor(name) {
        super(name);

        this.haveMouseMovedStrength = false;

        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown },
            lora: { bounds: [0, 0], onDown: this.onLoraDown },
            strengthDec: { bounds: [0, 0], onDown: this.onStrengthDecDown },
            strengthVal: { bounds: [0, 0], onUp: this.onStrengthValUp },
            strengthInc: { bounds: [0, 0], onDown: this.onStrengthIncDown },
            strengthAny: { bounds: [0, 0], onMove: this.onStrengthAnyMove },
        };

        this._value = { ...DEFAULT_LORA_WIDGET_DATA };
    }

    set value(value) {
        this._value = typeof value === "object" && value !== null
            ? { ...DEFAULT_LORA_WIDGET_DATA, ...value, strengthTwo: undefined }
            : { ...DEFAULT_LORA_WIDGET_DATA };

        delete this._value.strengthTwo;
    }

    get value() {
        return this._value;
    }

    setLora(lora) {
        this._value.lora = lora;
    }

    draw(ctx, node, width, posY, height) {
        ctx.save();

        const margin = 10;
        const innerMargin = margin * 0.33;
        const widgetWidth = Math.max(120, width ?? node.size?.[0] ?? 120);
        const lowQuality = isLowQuality();
        const midY = posY + height * 0.5;

        this.lastWidth = widgetWidth;

        let posX = margin;

        drawRoundedRectangle(ctx, {
            posX,
            posY,
            height,
            width: Math.max(0, widgetWidth - margin * 2),
        });

        this.hitAreas.toggle.bounds = drawTogglePart(ctx, {
            posX,
            posY,
            height,
            value: this.value.on,
        });

        posX += this.hitAreas.toggle.bounds[1] + innerMargin;

        if (lowQuality) {
            ctx.restore();
            return;
        }

        if (!this.value.on) {
            ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
        }

        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;

        const strength = this.value.strength ?? 1;
        const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
            posX: widgetWidth - margin - innerMargin * 2,
            posY,
            height,
            value: strength,
            direction: -1,
        });

        this.hitAreas.strengthDec.bounds = leftArrow;
        this.hitAreas.strengthVal.bounds = text;
        this.hitAreas.strengthInc.bounds = rightArrow;
        this.hitAreas.strengthAny.bounds = [leftArrow[0], rightArrow[0] + rightArrow[1] - leftArrow[0]];

        const loraWidth = Math.max(0, leftArrow[0] - innerMargin - posX);

        ctx.textAlign = "left";
        ctx.textBaseline = "middle";

        const loraLabel = String(this.value?.lora || "None");
        ctx.fillText(fitString(ctx, loraLabel, loraWidth), posX, midY);

        this.hitAreas.lora.bounds = [posX, loraWidth];

        ctx.globalAlpha = app.canvas.editor_alpha;
        ctx.restore();
    }

    serializeValue() {
        const value = { ...this.value };
        delete value.strengthTwo;
        return value;
    }

    onToggleDown(event, pos, node) {
        this.value.on = !this.value.on;
        this.cancelMouseDown();
        node.setDirtyCanvas?.(true, true);
        return true;
    }

    onLoraDown(event, pos, node) {
        showLoraChooser(event, (value) => {
            if (typeof value === "string") {
                this.value.lora = value;
            }

            node.setDirtyCanvas(true, true);
        }, null, node);

        this.cancelMouseDown();
    }

    onStrengthDecDown() {
        this.stepStrength(-1);
    }

    onStrengthIncDown() {
        this.stepStrength(1);
    }

    onStrengthAnyMove(event) {
        if (event.deltaX) {
            this.haveMouseMovedStrength = true;
            this.value.strength = (this.value.strength ?? 1) + event.deltaX * 0.05;
        }
    }

    onStrengthValUp(event, pos, node) {
        if (this.haveMouseMovedStrength) {
            return;
        }

        app.canvas.prompt("Value", this.value.strength, (value) => {
            const number = Number(value);
            if (!Number.isNaN(number)) {
                this.value.strength = number;
                node.setDirtyCanvas?.(true, true);
            }
        }, event);
    }

    onMouseUp(event, pos, node) {
        super.onMouseUp(event, pos, node);
        this.haveMouseMovedStrength = false;
    }

    stepStrength(direction) {
        const strength = (this.value.strength ?? 1) + 0.05 * direction;
        this.value.strength = Math.round(strength * 100) / 100;
    }
}

function fitLorasNodeSize(node) {
    if (!node?.computeSize || !node?.size) {
        return;
    }

    const [computedWidth, computedHeight] = node.computeSize();
    node.setSize?.([
        Math.max(node.size[0] ?? 0, computedWidth ?? 0),
        Math.max(node.size[1] ?? 0, computedHeight ?? 0),
    ]);
    node.setDirtyCanvas?.(true, true);
}

// Every node whose widgets are lora rows. LorasList keeps a leading widget of its own, the others do not.
const LORA_ROW_NODES = new Set([NodeNames.LorasLoader, NodeNames.LorasList, NodeNames.WanExpertLoras]);

app.registerExtension({
    name: ExtensionNames.LorasLoader,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!LORA_ROW_NODES.has(nodeData.name)) {
            return;
        }

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);

            while (this.widgets?.length) {
                if (nodeData.name === NodeNames.LorasList) {
                    if (this.widgets.length === 1) {
                        break;
                    }

                    removeWidget(this, 1);
                } else {
                    removeWidget(this, 0);
                }
            }

            this.addLoraWidget = null;
            this._tempWidth = this.size[0];
            this._tempHeight = this.size[1];

            for (const widgetValue of info.widgets_values || []) {
                if (widgetValue?.lora !== undefined) {
                    const widget = addNewLoraWidget(this);
                    widget.value = { ...widgetValue };
                }
            }

            addNonLoraWidgets(this);

            this.size[0] = this._tempWidth;
            this.size[1] = Math.max(this._tempHeight, this.computeSize()[1]);

            requestAnimationFrame(() => fitLorasNodeSize(this));

            return result;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            this.loraWidgetsCounter ??= 0;

            neApi.getLoras();
            addNonLoraWidgets(this);

            requestAnimationFrame(() => fitLorasNodeSize(this));

            return result;
        };

        const getSlotInPosition = nodeType.prototype.getSlotInPosition;
        nodeType.prototype.getSlotInPosition = function (canvasX, canvasY) {
            const result = getSlotInPosition?.apply(this, arguments);

            if (result) {
                return result;
            }

            let lastWidget = null;

            for (const widget of this.widgets) {
                if (!widget.last_y) {
                    return;
                }

                if (canvasY > this.pos[1] + widget.last_y) {
                    lastWidget = widget;
                    continue;
                }

                break;
            }

            if (lastWidget?.name?.startsWith(WidgetNames.LoraPrefix)) {
                return { widget: lastWidget, output: { type: "LORA WIDGET" } };
            }

            return null;
        };

        const getSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
        nodeType.prototype.getSlotMenuOptions = function (slot) {
            if (slot?.widget?.name?.startsWith(WidgetNames.LoraPrefix)) {
                const widget = slot.widget;
                const index = this.widgets.indexOf(widget);

                buildLoraSlotMenuItems(this, widget, index).then((menuItems) => {
                    new LiteGraph.ContextMenu(
                        menuItems,
                        { title: "LORA WIDGET", event: app.lastAdjustedMouseEvent },
                        app.canvas.getCanvasWindow(),
                    );
                });

                return null;
            }

            return getSlotMenuOptions?.apply(this, arguments);
        };
    },
});