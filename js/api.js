import { HeaderNames } from "./constants.js";

class NeApi {
    constructor(baseUrl) {
        this.getLorasPromise = null;
        this.getExpertLorasPromise = null;
        this.loraInfoCache = new Map();
        this.baseUrl = baseUrl || "./ne";
    }

    apiUrl(route) {
        const normalizedRoute = route.startsWith("/") ? route : `/${route}`;
        return `${this.baseUrl}${normalizedRoute}`;
    }

    fetchApi(route, options) {
        return fetch(this.apiUrl(route), options);
    }

    async fetchJson(route, options) {
        const response = await this.fetchApi(route, options);

        if (!response.ok) {
            throw new Error(`Request failed: ${response.status} ${response.statusText}`);
        }

        if (response.status === 204) {
            return null;
        }

        return await response.json();
    }

    getLoras(force = false) {
        if (!this.getLorasPromise || force) {
            this.getLorasPromise = this.fetchJson("/loras", { cache: "no-store" })
                .catch((error) => {
                    this.getLorasPromise = null;
                    throw error;
                });
        }

        return this.getLorasPromise;
    }

    getExpertLoras(force = false) {
        if (!this.getExpertLorasPromise || force) {
            this.getExpertLorasPromise = this.fetchJson("/loras?experts=1", { cache: "no-store" })
                .catch((error) => {
                    this.getExpertLorasPromise = null;
                    throw error;
                });
        }

        return this.getExpertLorasPromise;
    }

    getLoraPreviewUrl(lora) {
        return this.apiUrl(`/loras/preview?lora=${encodeURIComponent(lora)}`);
    }

    async getLoraInfo(lora, force = false) {
        if (!lora) {
            return { has_info: false };
        }

        if (!force && this.loraInfoCache.has(lora)) {
            return this.loraInfoCache.get(lora);
        }

        const promise = this.fetchJson(`/loras/info?lora=${encodeURIComponent(lora)}`, { cache: "no-store" })
            .catch((error) => {
                this.loraInfoCache.delete(lora);
                throw error;
            });

        this.loraInfoCache.set(lora, promise);
        return promise;
    }

    async reboot() {
        try {
            const response = await this.fetchApi("/reboot", {
                method: "POST",
                headers: { [HeaderNames.Reboot]: HeaderNames.RebootValue },
            });

            if (!response.ok) {
                console.error("Failed to reboot the server!", response.status, response.statusText);
            }

            return response;
        } catch {
            // The server replaces its own process, so losing the connection is the expected outcome.
            return null;
        }
    }

    async resetNode(id) {
        const response = await this.fetchApi(`/reset/${id}`, { method: "POST" });

        if (!response.ok) {
            console.error("Failed to reset node!", response.status, response.statusText);
        }

        return response;
    }
}

export const neApi = new NeApi();