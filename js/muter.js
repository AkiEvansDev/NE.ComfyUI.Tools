import { app } from "../../scripts/app.js";
import { ExtensionNames, NodeNames, WidgetNames } from "./constants.js";
import { SERVICE as FAST_GROUPS_SERVICE } from "./services/fast_groups_service.js";
import { fitString, drawNodeWidget } from "./draw.js";

const PROPERTY_SORT = "sort";
const PROPERTY_SORT_CUSTOM_ALPHA = "customSortAlphabet";
const PROPERTY_MATCH_TITLE = "matchTitle";
const PROPERTY_SHOW_NAV = "showNav";
const PROPERTY_RESTRICTION = "toggleRestriction";

const WIDGETS_START_Y = 34;
// _arrangeWidgets lays every widget out at computeSize().height + 4 and grows the node itself only on the
// canvas renderer — under the Vue one nothing corrects a node that came out too short, and the toggles
// simply draw past its bottom edge. So the height has to be right here, in the same arithmetic.
const WIDGET_GAP = 4;
const BOTTOM_PADDING = 6;

function getCustomAlphabet(value) {
    const str = value?.replace(/\n/g, "").trim();

    if (!str) {
        return null;
    }

    const alphabet = str.includes(",")
        ? str.toLocaleLowerCase().split(",").map((x) => x.trim()).filter(Boolean)
        : str.toLocaleLowerCase().split("");

    return alphabet.length ? alphabet : null;
}

function fitGroupsMuterSize(node) {
    // getLayoutWidgets() skips hidden ones, so the hidden groups widget takes no space and must not be counted.
    const widgetCount = (node.widgets ?? []).filter((widget) => !widget.hidden).length;
    const widgetsStartY = node.widgets_start_y ?? WIDGETS_START_Y;
    const widgetHeight = (LiteGraph.NODE_WIDGET_HEIGHT ?? 20) + WIDGET_GAP;

    const requiredHeight = widgetsStartY + widgetCount * widgetHeight + BOTTOM_PADDING;

    if ((node.size?.[1] ?? 0) < requiredHeight) {
        node.setSize([
            node.size?.[0] ?? 200,
            requiredHeight,
        ]);

        markNodeDirty(node);
    }
}

function sortByCustomAlphabet(groups, customAlphabet) {
    groups.sort((a, b) => {
        const titleA = String(a.title ?? "").toLocaleLowerCase();
        const titleB = String(b.title ?? "").toLocaleLowerCase();

        let aIndex = -1;
        let bIndex = -1;

        for (const [index, alpha] of customAlphabet.entries()) {
            if (aIndex < 0 && titleA.startsWith(alpha)) {
                aIndex = index;
            }

            if (bIndex < 0 && titleB.startsWith(alpha)) {
                bIndex = index;
            }

            if (aIndex > -1 && bIndex > -1) {
                break;
            }
        }

        if (aIndex > -1 && bIndex > -1) {
            const result = aIndex - bIndex;
            return result === 0 ? titleA.localeCompare(titleB) : result;
        }

        if (aIndex > -1) {
            return -1;
        }

        if (bIndex > -1) {
            return 1;
        }

        return titleA.localeCompare(titleB);
    });
}

function createTitleMatcher(value) {
    const matchTitle = value?.trim();

    if (!matchTitle) {
        return null;
    }

    try {
        return new RegExp(matchTitle, "i");
    } catch (error) {
        console.error("[NE.GroupsMuter] Invalid matchTitle regex:", error);
        return null;
    }
}

function removeWidget(node, widgetOrSlot) {
    if (!node.widgets) {
        return;
    }

    if (typeof widgetOrSlot === "number") {
        node.widgets.splice(widgetOrSlot, 1);
        return;
    }

    if (widgetOrSlot) {
        const index = node.widgets.indexOf(widgetOrSlot);
        if (index > -1) {
            node.widgets.splice(index, 1);
        }
    }
}

function markNodeDirty(node) {
    node.setDirtyCanvas?.(true, false);
    app.canvas?.setDirty?.(true, true);
}

function updateHiddenText(node, text) {
    const hidden = node.widgets?.find((widget) => widget.name === WidgetNames.Groups);

    if (hidden) {
        hidden.value = text;
    }
}

function centerCanvasOnGroup(group) {
    const canvas = app.canvas;
    const scale = canvas?.ds?.scale ?? 1;

    if (!canvas || scale <= 0.5) {
        return;
    }

    canvas.centerOnNode(group);

    const groupWidth = group._size?.[0] ?? group.size?.[0] ?? 1;
    const groupHeight = group._size?.[1] ?? group.size?.[1] ?? 1;

    const zoomCurrent = canvas.ds?.scale ?? 1;
    const zoomX = canvas.canvas.width / groupWidth - 0.02;
    const zoomY = canvas.canvas.height / groupHeight - 0.02;

    canvas.setZoom(Math.min(zoomCurrent, zoomX, zoomY), [
        canvas.canvas.width / 2,
        canvas.canvas.height / 2,
    ]);

    canvas.setDirty(true, true);
}

function createGroupWidget(node, group) {
    const widget = node.addCustomWidget({
        name: `group_${group.id ?? group.title}`,
        label: group.title,
        value: false,
        disabled: false,
        // Kept out of both widgets_values (top-level flag) and the prompt (`options.serialize`).
        options: { on: "yes", off: "no", serialize: false },

        serialize: false,

        computeSize() {
            return [0, LiteGraph.NODE_WIDGET_HEIGHT];
        },

        draw(ctx, currentNode, width, posY, height) {
            // Everything below is laid out from the right edge of this width, so a value larger than the node
            // draws the whole row past its side. drawWidgets hands over the node width and nothing here sets
            // widget.width, yet a row was once seen hanging out sideways — clamping makes that impossible.
            const nodeWidth = currentNode.size?.[0] ?? 0;
            const widgetData = drawNodeWidget(ctx, {
                width: nodeWidth > 0 ? Math.min(width ?? nodeWidth, nodeWidth) : width,
                height,
                posY,
            });

            this.lastWidth = Math.max(0, widgetData.width ?? width ?? nodeWidth);

            const showNav = currentNode.properties?.[PROPERTY_SHOW_NAV] !== false;
            let currentX = this.lastWidth - widgetData.margin;

            if (!widgetData.lowQuality && showNav) {
                currentX -= 7;

                const midY = widgetData.posY + widgetData.height * 0.5;

                ctx.fillStyle = ctx.strokeStyle = "#89A";
                ctx.lineJoin = "round";
                ctx.lineCap = "round";

                const arrow = new Path2D(`M${currentX} ${midY} l -7 6 v -3 h -7 v -6 h 7 v -3 z`);

                ctx.fill(arrow);
                ctx.stroke(arrow);

                currentX -= 14;
                currentX -= 7;

                ctx.strokeStyle = widgetData.colorOutline;
                ctx.stroke(new Path2D(`M ${currentX} ${widgetData.posY} v ${widgetData.height}`));
            } else if (widgetData.lowQuality && showNav) {
                currentX -= 28;
            }

            currentX -= 7;

            ctx.fillStyle = this.value ? "#89A" : "#333";
            ctx.beginPath();

            const toggleRadius = height * 0.36;

            ctx.arc(currentX - toggleRadius, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
            ctx.fill();

            currentX -= toggleRadius * 2;

            if (!widgetData.lowQuality) {
                currentX -= 4;

                ctx.textAlign = "right";
                ctx.fillStyle = this.value ? widgetData.colorText : widgetData.colorTextSecondary;

                const label = this.label || this.name;
                const toggleLabelOn = this.options.on || "true";
                const toggleLabelOff = this.options.off || "false";

                ctx.fillText(this.value ? toggleLabelOn : toggleLabelOff, currentX, posY + height * 0.7);

                currentX -= Math.max(
                    ctx.measureText(toggleLabelOn).width,
                    ctx.measureText(toggleLabelOff).width,
                );

                currentX -= 7;
                ctx.textAlign = "left";

                const maxLabelWidth = this.lastWidth - widgetData.margin - 10 - (this.lastWidth - currentX);

                if (label != null) {
                    ctx.fillText(
                        fitString(ctx, label, maxLabelWidth),
                        widgetData.margin + 10,
                        posY + height * 0.7,
                    );
                }
            }
        },

        serializeValue() {
            return this.value;
        },

        mouse(event, pos, currentNode) {
            if (event.type !== "pointerdown") {
                return true;
            }

            const showNav = currentNode.properties?.[PROPERTY_SHOW_NAV] !== false;
            const widgetWidth = this.lastWidth ?? currentNode.size?.[0] ?? 0;
            const clickedNav = showNav && pos[0] >= widgetWidth - 15 - 28 - 1;

            if (clickedNav) {
                centerCanvasOnGroup(this.neGroup);
                return true;
            }

            this.value = !this.value;

            setTimeout(() => {
                this.callback?.call(this, this.value, app.canvas, currentNode, pos, event);
            }, 20);

            return true;
        },
    });

    widget.neGroup = group;

    widget.doModeChange = function (force, skipOtherNodeCheck) {
        const group = this.neGroup;

        FAST_GROUPS_SERVICE.recomputeInsideNodesForGroup(group);

        const hasAnyActiveNodes = group._nodes?.some((groupNode) => groupNode.mode === LiteGraph.ALWAYS);
        let newValue = force != null ? force : !hasAnyActiveNodes;

        if (skipOtherNodeCheck !== true) {
            const restriction = node.properties?.[PROPERTY_RESTRICTION];

            if (newValue && restriction?.includes(" one")) {
                for (const otherWidget of node.widgets ?? []) {
                    otherWidget.doModeChange?.(false, true);
                }
            } else if (!newValue && restriction === "always one") {
                newValue = node.widgets?.every((otherWidget) => !otherWidget.value || otherWidget === this) ?? newValue;
            }
        }

        for (const groupNode of group._nodes ?? []) {
            groupNode.mode = newValue ? node.modeOn : node.modeOff;
        }

        group._hasAnyActiveNode = newValue;
        this.value = newValue;

        app.graph?.setDirtyCanvas?.(true, false);
        app.canvas?.setDirty?.(true, true);
    };

    widget.callback = function () {
        this.doModeChange();
    };

    return widget;
}

function refreshWidgets() {
    let sort = this.properties?.[PROPERTY_SORT] || "position";
    let customAlphabet = null;

    if (sort === "custom alphabet") {
        customAlphabet = getCustomAlphabet(this.properties?.[PROPERTY_SORT_CUSTOM_ALPHA]);

        if (!customAlphabet) {
            sort = "alphanumeric";
        }
    }

    const groups = [...FAST_GROUPS_SERVICE.getGroups(sort)];

    if (customAlphabet) {
        sortByCustomAlphabet(groups, customAlphabet);
    }

    const titleMatcher = createTitleMatcher(this.properties?.[PROPERTY_MATCH_TITLE]);

    let index = 1;
    const enabledTitles = [];

    for (const group of groups) {
        const groupTitle = String(group.title ?? "");

        if (titleMatcher && !titleMatcher.exec(groupTitle)) {
            continue;
        }

        let widget = this.widgets.find((candidate) => candidate.neGroup === group);

        if (!widget) {
            widget = createGroupWidget(this, group);

            const size = this.computeSize();

            this.setSize([
                Math.max(this.size[0], size[0]),
                Math.max(this.size[1], size[1]),
            ]);
        }

        if (widget.label !== groupTitle) {
            widget.label = groupTitle;
            markNodeDirty(this);
        }

        if (widget.value !== group._hasAnyActiveNode) {
            widget.value = group._hasAnyActiveNode;
            markNodeDirty(this);
        }

        const oldIndex = this.widgets.findIndex((candidate) => candidate === widget);

        if (oldIndex > -1 && oldIndex !== index) {
            this.widgets.splice(index, 0, this.widgets.splice(oldIndex, 1)[0]);
            markNodeDirty(this);
        }

        if (widget.value) {
            enabledTitles.push(groupTitle);
        }

        index++;
    }

    updateHiddenText(this, enabledTitles.join(", "));

    while ((this.widgets || [])[index]) {
        removeWidget(this, index);
    }

    fitGroupsMuterSize(this);
}

app.registerExtension({
    name: ExtensionNames.GroupsMuter,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NodeNames.GroupsMuter) {
            return;
        }

        nodeType["@matchTitle"] = { type: "string" };
        nodeType["@showNav"] = { type: "boolean" };
        nodeType["@sort"] = {
            type: "combo",
            values: ["position", "alphanumeric", "custom alphabet"],
        };
        nodeType["@customSortAlphabet"] = { type: "string" };
        nodeType["@toggleRestriction"] = {
            type: "combo",
            values: ["default", "max one", "always one"],
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            this.modeOn = LiteGraph.ALWAYS;
            this.modeOff = LiteGraph.NEVER;
            this.serialize_widgets = false;
            this.widgets_start_y = WIDGETS_START_Y;

            this.properties ??= {};
            this.properties[PROPERTY_MATCH_TITLE] ??= "";
            this.properties[PROPERTY_SHOW_NAV] ??= true;
            this.properties[PROPERTY_SORT] ??= "alphanumeric";
            this.properties[PROPERTY_SORT_CUSTOM_ALPHA] ??= "";
            this.properties[PROPERTY_RESTRICTION] ??= "default";

            const hiddenWidget = this.widgets?.find((widget) => widget.name === WidgetNames.Groups);
            if (hiddenWidget) {
                hiddenWidget.hidden = true;
            }

            this.outputs ??= [];

            if (!this.outputs.length) {
                this.addOutput("options", "STRING");
            }

            this.outputs.length = 1;
            this.outputs[0].name = "options";
            this.outputs[0].type = "STRING";

            this.removeWidget = function (widgetOrSlot) {
                removeWidget(this, widgetOrSlot);
            };

            this.refreshWidgets = refreshWidgets;

            requestAnimationFrame(() => {
                this.refreshWidgets?.();
                fitGroupsMuterSize(this);
            });

            return result;
        };

        const onAdded = nodeType.prototype.onAdded;

        nodeType.prototype.onAdded = function () {
            const result = onAdded?.apply(this, arguments);
            FAST_GROUPS_SERVICE.addFastGroupNode(this);
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;

        nodeType.prototype.onRemoved = function () {
            const result = onRemoved?.apply(this, arguments);
            FAST_GROUPS_SERVICE.removeFastGroupNode(this);
            return result;
        };
    },
});