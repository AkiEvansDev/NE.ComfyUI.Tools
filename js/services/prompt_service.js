import { api } from "../../../scripts/api.js";

function createDeferred() {
    let resolve;
    let reject;

    const promise = new Promise((res, rej) => {
        resolve = res;
        reject = rej;
    });

    return { promise, resolve, reject };
}

function getComfyGraph() {
    return window?.app?.graph ?? null;
}

function getExecutingNodeId(detail) {
    // The end of a run arrives as { node: null }, and a bare id is the older shape. Reading the id as
    // `detail?.node ?? detail` turned the first into the detail object itself, which then rendered as
    // "[object Object]" for the last frame before the queue went idle.
    if (detail != null && typeof detail === "object") {
        return detail.node ?? null;
    }

    return detail ?? null;
}

function getApiNodeTitle(apiNode) {
    return apiNode?._meta?.title || apiNode?.class_type || null;
}

function getGraphNodeTitle(nodeId) {
    const graphNode = getComfyGraph()?.getNodeById?.(Number(nodeId));

    return graphNode?.title || graphNode?.type || null;
}

export class PromptExecution {
    constructor(id) {
        this.id = id;
        this.promptApi = null;
        this.promptReady = createDeferred();

        this.totalNodes = 0;
        this.executedNodeIds = new Set();
        this.currentlyExecuting = null;
        this.errorDetails = null;
        this.isFinished = false;
    }

    setPrompt(prompt) {
        this.promptApi = prompt?.output ?? null;
        this.totalNodes = this.promptApi ? Object.keys(this.promptApi).length : 0;
        this.promptReady.resolve();
    }

    getApiNode(nodeId) {
        if (!this.promptApi || nodeId == null) {
            return null;
        }

        return this.promptApi[String(nodeId)] ?? null;
    }

    getNodeLabel(nodeId) {
        const apiNode = this.getApiNode(nodeId);

        return (
            getApiNodeTitle(apiNode) ||
            getGraphNodeTitle(nodeId) ||
            String(nodeId)
        );
    }

    markExecutingNode(nodeId) {
        if (nodeId == null) {
            this.finishCurrentNode();
            this.currentlyExecuting = null;
            this.isFinished = true;
            return;
        }

        if (this.currentlyExecuting?.nodeId === nodeId) {
            return;
        }

        this.finishCurrentNode();

        this.currentlyExecuting = {
            nodeId,
            nodeLabel: this.getNodeLabel(nodeId),
            // A node with no internal progress reports nothing at all, so how long it has been running
            // is the only sign it is alive rather than stuck.
            startedAt: performance.now(),
            step: null,
            maxSteps: null,
            pass: 0,
            maxPasses: null,
        };

        this.promptReady.promise.then(() => {
            if (!this.currentlyExecuting || this.currentlyExecuting.nodeId !== nodeId) {
                return;
            }

            const apiNode = this.getApiNode(nodeId);

            this.currentlyExecuting.nodeLabel ||= this.getNodeLabel(nodeId);

            if (apiNode?.class_type === "UltimateSDUpscale") {
                this.currentlyExecuting.pass--;
                this.currentlyExecuting.maxPasses = -1;
            }

            if (apiNode?.class_type === "IterativeImageUpscale") {
                this.currentlyExecuting.maxPasses = apiNode.inputs?.steps ?? -1;
            }
        });
    }

    updateStep(nodeId, step, maxSteps) {
        this.markExecutingNode(nodeId);

        if (!this.currentlyExecuting || step == null) {
            return;
        }

        if (this.currentlyExecuting.step == null || step < this.currentlyExecuting.step) {
            this.currentlyExecuting.pass++;
        }

        this.currentlyExecuting.step = step;
        this.currentlyExecuting.maxSteps = maxSteps ?? null;
    }

    markCached(nodeId) {
        if (nodeId != null) {
            this.executedNodeIds.add(String(nodeId));
        }
    }

    finishCurrentNode() {
        if (this.currentlyExecuting?.nodeId != null) {
            this.executedNodeIds.add(String(this.currentlyExecuting.nodeId));
        }
    }

    setError(details) {
        this.errorDetails = details ?? {
            exception_type: "Unknown",
        };
    }

    get nodeProgressPercent() {
        if (!this.totalNodes) {
            return null;
        }

        return Math.min(100, (this.executedNodeIds.size / this.totalNodes) * 100);
    }
}

class PromptService extends EventTarget {
    constructor(apiInstance) {
        super();

        this.api = apiInstance;
        this.prompts = new Map();
        this.currentExecution = null;
        this.lastQueueRemaining = 0;

        this.patchQueuePrompt();
        this.registerApiEvents();
    }

    patchQueuePrompt() {
        if (this.api.__neProgressQueuePromptPatched) {
            return;
        }

        this.api.__neProgressQueuePromptPatched = true;

        const originalQueuePrompt = this.api.queuePrompt;
        const service = this;

        this.api.queuePrompt = async function (...args) {
            let response;

            try {
                response = await originalQueuePrompt.apply(this, args);
            } catch (error) {
                const prompt = service.getOrCreatePrompt("error");
                prompt.setError({ exception_type: "QueuePromptError" });
                service.currentExecution = prompt;
                service.dispatchProgressUpdate();
                throw error;
            }

            const promptPayload = args[1];
            const promptId = response?.prompt_id ?? "unknown";
            const prompt = service.getOrCreatePrompt(promptId);

            prompt.setPrompt(promptPayload);

            if (!service.currentExecution) {
                service.currentExecution = prompt;
            }

            service.dispatchEvent(
                new CustomEvent("queue-prompt", {
                    detail: { prompt },
                }),
            );

            service.dispatchProgressUpdate();

            return response;
        };
    }

    registerApiEvents() {
        this.api.addEventListener("status", (event) => {
            const queueRemaining = event.detail?.exec_info?.queue_remaining;

            if (queueRemaining == null) {
                return;
            }

            this.lastQueueRemaining = queueRemaining;
            this.dispatchProgressUpdate();
        });

        this.api.addEventListener("execution_start", (event) => {
            const prompt = this.getOrCreatePrompt(event.detail?.prompt_id ?? "unknown");

            prompt.isFinished = false;
            prompt.errorDetails = null;

            this.currentExecution = prompt;
            this.dispatchProgressUpdate();
        });

        this.api.addEventListener("executing", (event) => {
            const prompt = this.ensureCurrentPrompt(event.detail?.prompt_id);

            prompt.markExecutingNode(getExecutingNodeId(event.detail));

            this.dispatchProgressUpdate();

            if (getExecutingNodeId(event.detail) == null) {
                this.currentExecution = null;
                this.cleanupOldPrompts();
            }
        });

        this.api.addEventListener("progress", (event) => {
            const prompt = this.ensureCurrentPrompt(event.detail?.prompt_id);

            prompt.updateStep(
                event.detail?.node,
                event.detail?.value,
                event.detail?.max,
            );

            this.dispatchProgressUpdate();
        });

        this.api.addEventListener("execution_cached", (event) => {
            const prompt = this.ensureCurrentPrompt(event.detail?.prompt_id);

            for (const nodeId of event.detail?.nodes ?? []) {
                prompt.markCached(nodeId);
            }

            this.dispatchProgressUpdate();
        });

        this.api.addEventListener("executed", (event) => {
            const prompt = this.ensureCurrentPrompt(event.detail?.prompt_id);

            if (event.detail?.node != null) {
                prompt.markCached(event.detail.node);
            }

            this.dispatchProgressUpdate();
        });

        this.api.addEventListener("execution_error", (event) => {
            const prompt = this.ensureCurrentPrompt(event.detail?.prompt_id);

            prompt.setError(event.detail);
            this.currentExecution = prompt;

            this.dispatchProgressUpdate();
        });
    }

    ensureCurrentPrompt(promptId = "unknown") {
        if (!this.currentExecution) {
            this.currentExecution = this.getOrCreatePrompt(promptId);
        }

        return this.currentExecution;
    }

    getOrCreatePrompt(id) {
        const promptId = String(id ?? "unknown");

        let prompt = this.prompts.get(promptId);

        if (!prompt) {
            prompt = new PromptExecution(promptId);
            this.prompts.set(promptId, prompt);
        }

        return prompt;
    }

    cleanupOldPrompts() {
        const maxPrompts = 20;

        if (this.prompts.size <= maxPrompts) {
            return;
        }

        const keys = [...this.prompts.keys()];

        for (const key of keys.slice(0, this.prompts.size - maxPrompts)) {
            this.prompts.delete(key);
        }
    }

    dispatchProgressUpdate() {
        this.dispatchEvent(
            new CustomEvent("progress-update", {
                detail: {
                    queue: this.lastQueueRemaining,
                    prompt: this.currentExecution,
                },
            }),
        );
    }
}

export const PROMPT_SERVICE = new PromptService(api);