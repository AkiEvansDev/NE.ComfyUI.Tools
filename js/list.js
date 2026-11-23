import { app } from "../../scripts/app.js";
import { ExtensionNames, NodeNames, WidgetNames } from "./constants.js";

const LIST_NODE_NAMES = new Set([
    NodeNames.IntList,
    NodeNames.FloatList,
    NodeNames.StringList,
    NodeNames.RangeList,
]);

const INTEGER_LIST_NODE_NAMES = new Set([
    NodeNames.IntList,
    NodeNames.RangeList,
]);

const MAX_LINES = 99;
const LINE_NUMBERS_WIDTH = 24;
const LINE_NUMBERS_SYNC_FRAMES = 8;

function styleListInput(input) {
    if (!input) return;

    input.style.border = `0.5px solid ${LiteGraph.WIDGET_OUTLINE_COLOR}`;
    input.style.borderRadius = "4px";
    input.style.boxSizing = "border-box";
    input.style.paddingLeft = `${LINE_NUMBERS_WIDTH + 6}px`;
    input.style.textWrap = "nowrap";
    input.style.whiteSpace = "pre";
    input.style.overflow = "auto";
}

function copyTextMetrics(source, target) {
    const style = getComputedStyle(source);

    target.style.font = style.font;
    target.style.fontFamily = style.fontFamily;
    target.style.fontSize = style.fontSize;
    target.style.fontWeight = style.fontWeight;
    target.style.lineHeight = style.lineHeight;
    target.style.paddingTop = style.paddingTop;
    target.style.paddingBottom = style.paddingBottom;
}

function createLineNumbersInput(input) {
    const numbers = document.createElement("textarea");

    numbers.className = input.className;
    numbers.tabIndex = -1;
    numbers.readOnly = true;
    numbers.setAttribute("aria-hidden", "true");

    numbers.style.position = "absolute";
    numbers.style.left = "0";
    numbers.style.top = "0";
    numbers.style.bottom = "0";
    numbers.style.width = `${LINE_NUMBERS_WIDTH}px`;
    numbers.style.height = "100%";
    numbers.style.margin = "0";
    numbers.style.boxSizing = "border-box";
    numbers.style.border = "0.5px solid rgba(255, 0, 0, 0)";
    numbers.style.borderRadius = "4px 0 0 4px";
    numbers.style.paddingLeft = "2px";
    numbers.style.paddingRight = "4px";
    numbers.style.textAlign = "right";
    numbers.style.resize = "none";
    numbers.style.overflow = "hidden";
    numbers.style.background = "transparent";
    numbers.style.pointerEvents = "none";
    numbers.style.opacity = "0.4";
    numbers.style.zIndex = "1";

    copyTextMetrics(input, numbers);

    return numbers;
}

function filterListValue(nodeName, value) {
    if (INTEGER_LIST_NODE_NAMES.has(nodeName)) {
        return value
            .split("\n")
            .map((line) => line.replace(/(?!^-)-|[^-\d]/g, ""))
            .join("\n");
    }

    if (nodeName === NodeNames.FloatList) {
        return value
            .split("\n")
            .map((line) => line.replace(/(?!^-)-|[^\d.-]|(?<=\..*)\./g, ""))
            .join("\n");
    }

    return value;
}

function clampAndFilterValue(nodeName, input) {
    const lines = input.value.split("\n");
    const limitedValue = lines.length > MAX_LINES
        ? lines.slice(0, MAX_LINES).join("\n")
        : input.value;
    const filteredValue = filterListValue(nodeName, limitedValue);

    if (input.value !== filteredValue) {
        const selectionStart = input.selectionStart;
        const selectionEnd = input.selectionEnd;

        input.value = filteredValue;

        if (document.activeElement === input) {
            const end = input.value.length;
            input.setSelectionRange(
                Math.min(selectionStart ?? end, end),
                Math.min(selectionEnd ?? end, end),
            );
        }
    }
}

function lineNumbersValue(value) {
    const lineCount = Math.max(1, value.split("\n").length);

    return Array.from({ length: lineCount }, (_, index) => String(index + 1)).join("\n");
}

function ensureLineNumbersMounted(input, numbers) {
    const parent = input.parentElement;

    if (!parent) {
        return false;
    }

    const parentStyle = getComputedStyle(parent);

    if (parentStyle.position === "static") {
        parent.style.position = "relative";
    }

    if (numbers.parentElement !== parent) {
        numbers.remove();
        parent.insertBefore(numbers, input);
    }

    return true;
}

function syncLineNumbers(input, numbers, nodeName) {
    if (!ensureLineNumbersMounted(input, numbers)) {
        return;
    }

    styleListInput(input);
    copyTextMetrics(input, numbers);
    clampAndFilterValue(nodeName, input);

    const nextNumbersValue = lineNumbersValue(input.value);

    if (numbers.value !== nextNumbersValue) {
        numbers.value = nextNumbersValue;
    }

    numbers.scrollTop = input.scrollTop;
    numbers.scrollLeft = 0;
}

function scheduleSync(state) {
    if (state.syncFrame) {
        return;
    }

    state.syncFrame = requestAnimationFrame(() => {
        state.syncFrame = null;
        syncLineNumbers(state.input, state.numbers, state.nodeName);
    });
}

function scheduleStartupSync(state) {
    let frame = 0;

    const tick = () => {
        if (state.disposed) {
            return;
        }

        syncLineNumbers(state.input, state.numbers, state.nodeName);

        if (++frame < LINE_NUMBERS_SYNC_FRAMES) {
            requestAnimationFrame(tick);
        }
    };

    requestAnimationFrame(tick);
}

function createLineNumbersState(nodeName, input) {
    const controller = new AbortController();
    const numbers = createLineNumbersInput(input);

    const state = {
        input,
        numbers,
        nodeName,
        controller,
        disposed: false,
        resizeObserver: null,
        mutationObserver: null,
        syncFrame: null,
    };

    const onChange = () => scheduleSync(state);
    const onScroll = () => {
        numbers.scrollTop = input.scrollTop;
        numbers.scrollLeft = 0;
    };

    input.addEventListener("input", onChange, { signal: controller.signal });
    input.addEventListener("change", onChange, { signal: controller.signal });
    input.addEventListener("keyup", onChange, { signal: controller.signal });
    input.addEventListener("scroll", onScroll, { signal: controller.signal });

    state.resizeObserver = new ResizeObserver(onChange);
    state.resizeObserver.observe(input);

    state.mutationObserver = new MutationObserver(onChange);
    state.mutationObserver.observe(document.body, {
        childList: true,
        subtree: true,
    });

    state.dispose = () => {
        state.disposed = true;
        controller.abort();
        state.resizeObserver?.disconnect();
        state.mutationObserver?.disconnect();

        if (state.syncFrame) {
            cancelAnimationFrame(state.syncFrame);
            state.syncFrame = null;
        }

        numbers.remove();
    };

    input.__neLineNumbersState = state;
    scheduleStartupSync(state);

    return state;
}

function setupLineNumbers(node, nodeName) {
    const input = node.widgets?.find((widget) => widget.name === WidgetNames.List)?.element;

    if (!input) {
        return;
    }

    styleListInput(input);

    const state = input.__neLineNumbersState ?? createLineNumbersState(nodeName, input);

    node.numbers = state.numbers;
    node.updateNumbers = () => syncLineNumbers(state.input, state.numbers, state.nodeName);
    node.__neListLineNumbersState = state;

    node.updateNumbers();
}

app.registerExtension({
    name: ExtensionNames.Lists,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!LIST_NODE_NAMES.has(nodeData.name)) {
            return;
        }

        const onConfigure = nodeType.prototype.onConfigure;

        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);

            requestAnimationFrame(() => {
                setupLineNumbers(this, nodeData.name);
                this.updateNumbers?.();
            });

            return result;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            setupLineNumbers(this, nodeData.name);

            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;

        nodeType.prototype.onRemoved = function () {
            this.__neListLineNumbersState?.dispose?.();
            this.__neListLineNumbersState = null;
            this.numbers = null;
            this.updateNumbers = null;

            return onRemoved?.apply(this, arguments);
        };
    },
});
