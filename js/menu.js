import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { Commands, ExtensionNames } from "./constants.js";
import { neApi } from "./api.js";

async function customConfirm(message) {
    try {
        return await window.app.extensionManager.dialog.confirm({
            title: "Confirm",
            message,
        });
    } catch {
        return window.confirm(message);
    }
}

function showToast(message) {
    app.extensionManager.toast.add({
        severity: "success",
        summary: "NE Tools",
        detail: message,
        life: 3000,
    });
}

async function rebootApi() {
    if ("electronAPI" in window) {
        window.electronAPI.restartApp();
        return;
    }

    const confirmed = await customConfirm("Are you sure you'd like to reboot the server?");
    if (!confirmed) {
        return;
    }

    await neApi.reboot();
}

async function freeModels(freeExecutionCache = false) {
    const body = freeExecutionCache
        ? { unload_models: true, free_memory: true }
        : { unload_models: true };

    const response = await api.fetchApi("/free", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    if (!response.ok) {
        throw new Error(`Failed to free models: ${response.status}`);
    }

    showToast(
        freeExecutionCache
            ? "Models and execution cache have been cleared."
            : "Models have been unloaded.",
    );
}

app.registerExtension({
    name: ExtensionNames.Menu,

    commands: [
        {
            id: Commands.RestartServer,
            label: "Restart Server",
            function: async () => rebootApi(),
        },
        {
            id: Commands.UnloadModels,
            label: "Unload Models",
            function: async () => freeModels(false),
        },
        {
            id: Commands.FreeModelCache,
            label: "Free Models and Cache",
            function: async () => freeModels(true),
        },
    ],

    menuCommands: [
        {
            path: ["Extensions", "NE Tools"],
            commands: [
                Commands.RestartServer,
                Commands.UnloadModels,
                Commands.FreeModelCache,
            ],
        },
    ],
});