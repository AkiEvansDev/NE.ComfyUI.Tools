import { app } from "../../scripts/app.js";
import { ExtensionNames } from "./constants.js";
import { PROMPT_SERVICE } from "./services/prompt_service.js";
import { addStylesheet } from "./base.js";

const ELEMENT_NAME = "ne-progress-bar";
const DEFAULT_HEIGHT = 20;
const TICK_INTERVAL_MS = 200;
// Below this a node is simply fast, and a number flickering past would be noise. Above it the node is
// the one worth looking at — which is the whole point of showing the time.
const SLOW_NODE_MS = 1000;

addStylesheet(import.meta.url);

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function formatPercent(value) {
    if (value == null || Number.isNaN(value)) {
        return "??%";
    }

    return `${Math.round(clamp(value, 0, 100))}%`;
}

function getStepProgress(current) {
    if (!current?.maxSteps) {
        return null;
    }

    return clamp((current.step / current.maxSteps) * 100, 0, 100);
}

function getElapsedText(current) {
    if (!current?.startedAt) {
        return "";
    }

    const elapsed = performance.now() - current.startedAt;

    if (elapsed < SLOW_NODE_MS) {
        return "";
    }

    const seconds = elapsed / 1000;

    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    }

    return `${Math.floor(seconds / 60)}m ${String(Math.floor(seconds % 60)).padStart(2, "0")}s`;
}

function getErrorText(prompt) {
    const error = prompt?.errorDetails;

    if (!error) {
        return null;
    }

    return [
        error.exception_type || "ExecutionError",
        error.node_id || "",
        error.node_type || "",
    ]
        .filter(Boolean)
        .join(" ");
}

export class NEProgressBar extends HTMLElement {
    constructor() {
        super();

        this.currentPromptExecution = null;
        this.currentQueue = 0;
        this.tickTimer = null;
        this.handleProgressUpdate = this.handleProgressUpdate.bind(this);

        this.attachShadow({ mode: "open" });
        this.createShadowDom();
    }

    connectedCallback() {
        PROMPT_SERVICE.addEventListener("progress-update", this.handleProgressUpdate);
        this.renderIdle();
    }

    disconnectedCallback() {
        PROMPT_SERVICE.removeEventListener("progress-update", this.handleProgressUpdate);
        this.stopTicking();
    }

    createShadowDom() {
        const style = document.createElement("style");

        style.textContent = `
            :host {
                position: relative;
                display: block;
                overflow: hidden;
                box-sizing: border-box;

                height: 20px;
                width: 100%;

                background: var(--ne-progress-bg-color, rgba(23, 23, 23, 0.92));
                color: var(--ne-progress-text-color, #fff);

                font-family: Arial, sans-serif;
                font-size: 10px;
                line-height: 1;
            }

            :host *,
            :host *::before,
            :host *::after {
                box-sizing: inherit;
            }

            .bar {
                position: absolute;
                left: 0;
                width: 0%;
                height: 50%;
                transition: width 80ms ease-in-out;
            }

            .bar.nodes {
                top: 0;
                z-index: 1;
                background: var(--ne-progress-nodes-color, rgb(0, 128, 0));
            }

            .bar.steps {
                top: 50%;
                z-index: 2;
                background: var(--ne-progress-steps-color, rgb(0, 170, 0));
            }

            .bar.error {
                height: 100%;
                top: 0;
                background: var(--ne-progress-error-color, rgb(128, 0, 0));
            }

            .overlay {
                position: absolute;
                inset: 0;
                z-index: 3;
                pointer-events: none;
                background: linear-gradient(
                    to bottom,
                    rgba(255, 255, 255, 0.25),
                    rgba(0, 0, 0, 0.25)
                );
                mix-blend-mode: overlay;
            }

            .text {
                position: relative;
                z-index: 4;

                display: flex;
                align-items: center;

                width: 100%;
                height: 100%;
                padding: 0 6px;

                overflow: hidden;
                white-space: nowrap;
                text-overflow: ellipsis;

                color: var(--ne-progress-text-color, #fff);
                text-shadow: 0 0 2px #000, 1px 1px 0 #000;
            }
        `;

        this.nodesBar = document.createElement("div");
        this.nodesBar.className = "bar nodes";
        this.nodesBar.part = "progress-nodes";

        this.stepsBar = document.createElement("div");
        this.stepsBar.className = "bar steps";
        this.stepsBar.part = "progress-steps";

        this.overlay = document.createElement("div");
        this.overlay.className = "overlay";
        this.overlay.part = "overlay";

        this.text = document.createElement("span");
        this.text.className = "text";
        this.text.part = "text";

        this.shadowRoot.append(style, this.nodesBar, this.stepsBar, this.overlay, this.text);
    }

    handleProgressUpdate(event) {
        const { queue, prompt } = event.detail ?? {};

        this.currentPromptExecution = prompt ?? null;
        this.currentQueue = queue ?? 0;

        if (prompt?.errorDetails) {
            this.renderError(prompt);
            return;
        }

        if (prompt?.currentlyExecuting) {
            this.startTicking();
            this.renderProgress(queue, prompt);
            return;
        }

        if (queue > 0) {
            this.renderOtherTab(queue);
            return;
        }

        this.renderIdle();
    }

    // A node that reports no steps sends no events either, so the elapsed time has to redraw itself.
    startTicking() {
        if (this.tickTimer != null) {
            return;
        }

        this.tickTimer = setInterval(() => {
            const prompt = this.currentPromptExecution;

            if (!prompt?.currentlyExecuting || prompt.errorDetails) {
                this.stopTicking();
                return;
            }

            this.renderProgress(this.currentQueue, prompt);
        }, TICK_INTERVAL_MS);
    }

    stopTicking() {
        if (this.tickTimer == null) {
            return;
        }

        clearInterval(this.tickTimer);
        this.tickTimer = null;
    }

    renderProgress(queue, prompt) {
        const current = prompt.currentlyExecuting;

        const nodePercent = prompt.nodeProgressPercent;
        const stepPercent = getStepProgress(current);

        this.nodesBar.classList.remove("error");
        this.stepsBar.classList.remove("error");

        this.nodesBar.style.width = nodePercent == null
            ? "0%"
            : `${Math.max(2, clamp(nodePercent, 0, 100))}%`;

        this.stepsBar.style.width = stepPercent == null
            ? "0%"
            : `${clamp(stepPercent, 0, 100)}%`;

        const nodeLabel = current.nodeLabel?.trim() || "???";

        let stepText = "";

        if (stepPercent != null) {
            if (current.pass > 1 || current.maxPasses != null) {
                stepText += `#${current.pass}`;

                if (current.maxPasses > 0) {
                    stepText += `/${current.maxPasses}`;
                }

                stepText += " - ";
            }

            stepText += formatPercent(stepPercent);
        }

        const queueText = `(${queue ?? 0})`;
        const nodeText = formatPercent(nodePercent);
        const currentText = stepText
            ? `${nodeLabel} (${stepText})`
            : nodeLabel;
        const elapsedText = getElapsedText(current);

        this.text.textContent = [`${queueText} ${nodeText} -`, currentText, elapsedText]
            .filter(Boolean)
            .join(" ");
    }

    renderError(prompt) {
        this.stopTicking();
        this.nodesBar.classList.add("error");
        this.stepsBar.classList.add("error");

        this.nodesBar.style.width = "100%";
        this.stepsBar.style.width = "0%";

        this.text.textContent = getErrorText(prompt) || "Execution error";
    }

    renderOtherTab(queue) {
        this.stopTicking();
        this.nodesBar.classList.remove("error");
        this.stepsBar.classList.remove("error");

        this.nodesBar.style.width = "0%";
        this.stepsBar.style.width = "0%";

        this.text.textContent = `(${queue}) Running... in another tab`;
    }

    renderIdle() {
        this.stopTicking();
        this.nodesBar.classList.remove("error");
        this.stepsBar.classList.remove("error");

        this.nodesBar.style.width = "0%";
        this.stepsBar.style.width = "0%";

        this.text.textContent = "Idle";
    }
}

if (!customElements.get(ELEMENT_NAME)) {
    customElements.define(ELEMENT_NAME, NEProgressBar);
}

function createProgressBar() {
    const existing = document.querySelector(ELEMENT_NAME);

    if (existing) {
        return existing;
    }

    const progressBar = document.createElement(ELEMENT_NAME);

    progressBar.style.height = `${DEFAULT_HEIGHT}px`;
    progressBar.style.fontSize = `${Math.max(10, DEFAULT_HEIGHT - 10)}px`;
    progressBar.style.fontWeight = DEFAULT_HEIGHT <= 22 ? "bold" : "normal";

    return progressBar;
}

function mountProgressBar() {
    const progressBar = createProgressBar();

    const container =
        document.querySelector(".comfyui-body-top") ||
        document.querySelector(".comfyui-body-bottom") ||
        document.body;

    if (!progressBar.isConnected) {
        container.prepend(progressBar);
    }
}

app.registerExtension({
    name: ExtensionNames.ProgressBar,

    async setup() {
        mountProgressBar();
    },
});