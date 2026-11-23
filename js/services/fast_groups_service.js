import { app } from "../../../scripts/app.js";

class FastGroupsService {
    constructor() {
        this.msThreshold = 400;

        this.msLastUnsorted = 0;
        this.msLastAlpha = 0;
        this.msLastPosition = 0;

        this.groupsUnsorted = [];
        this.groupsSortedAlpha = [];
        this.groupsSortedPosition = [];

        this.fastGroupNodes = [];

        this.runScheduleTimeout = null;
        this.runScheduleAnimation = null;

        this.cachedNodeBoundings = null;
        this.clearBoundingsTimeout = null;
    }

    addFastGroupNode(node) {
        if (!this.fastGroupNodes.includes(node)) {
            this.fastGroupNodes.push(node);
        }

        this.scheduleRun(100);
    }

    removeFastGroupNode(node) {
        const index = this.fastGroupNodes.indexOf(node);

        if (index > -1) {
            this.fastGroupNodes.splice(index, 1);
        }

        if (!this.fastGroupNodes.length) {
            this.clearScheduledRun();
            this.invalidateGroups();
        }
    }

    invalidateGroups() {
        this.groupsUnsorted = [];
        this.groupsSortedAlpha = [];
        this.groupsSortedPosition = [];

        this.msLastUnsorted = 0;
        this.msLastAlpha = 0;
        this.msLastPosition = 0;
    }

    run() {
        if (!this.fastGroupNodes.length) {
            this.clearScheduledRun();
            return;
        }

        for (const node of [...this.fastGroupNodes]) {
            if (node.removed) {
                this.removeFastGroupNode(node);
                continue;
            }

            node.refreshWidgets?.();
        }

        this.clearScheduledRun();
        this.scheduleRun(500);
    }

    scheduleRun(ms = 500) {
        if (!this.fastGroupNodes.length) {
            return;
        }

        if (this.runScheduleTimeout || this.runScheduleAnimation) {
            return;
        }

        this.runScheduleTimeout = setTimeout(() => {
            this.runScheduleTimeout = null;
            this.runScheduleAnimation = requestAnimationFrame(() => {
                this.runScheduleAnimation = null;
                this.run();
            });
        }, ms);
    }

    clearScheduledRun() {
        if (this.runScheduleTimeout) {
            clearTimeout(this.runScheduleTimeout);
        }

        if (this.runScheduleAnimation) {
            cancelAnimationFrame(this.runScheduleAnimation);
        }

        this.runScheduleTimeout = null;
        this.runScheduleAnimation = null;
    }

    getBoundingsForAllNodes() {
        if (!this.cachedNodeBoundings) {
            this.cachedNodeBoundings = {};

            for (const node of app.graph?._nodes ?? []) {
                this.cachedNodeBoundings[node.id] = node.getBounding();
            }

            clearTimeout(this.clearBoundingsTimeout);
            this.clearBoundingsTimeout = setTimeout(() => {
                this.cachedNodeBoundings = null;
                this.clearBoundingsTimeout = null;
            }, 50);
        }

        return this.cachedNodeBoundings;
    }

    recomputeInsideNodesForGroup(group) {
        const cachedBoundings = this.getBoundingsForAllNodes();
        const nodes = group.graph?._nodes ?? app.graph?._nodes ?? [];
        const groupBounding = group._bounding ?? group.getBounding?.();

        if (!groupBounding) {
            return;
        }

        group._nodes = group._nodes ?? [];
        group._nodes.length = 0;

        for (const node of nodes) {
            const nodeBounding = cachedBoundings[node.id];

            if (!nodeBounding || !LiteGraph.overlapBounding(groupBounding, nodeBounding)) {
                continue;
            }

            group._nodes.push(node);
        }
    }

    getGroupsUnsorted(now) {
        const canvas = app.canvas;
        const graph = app.graph;

        if (
            !canvas?.selected_group_moving &&
            (!this.groupsUnsorted.length || now - this.msLastUnsorted > this.msThreshold)
        ) {
            this.groupsUnsorted = [...(graph?._groups ?? [])];

            for (const group of this.groupsUnsorted) {
                this.recomputeInsideNodesForGroup(group);
                group._hasAnyActiveNode = group._nodes.some((node) => node.mode === LiteGraph.ALWAYS);
            }

            this.msLastUnsorted = now;
        }

        return this.groupsUnsorted;
    }

    getGroupsAlpha(now) {
        if (!this.groupsSortedAlpha.length || now - this.msLastAlpha > this.msThreshold) {
            this.groupsSortedAlpha = [...this.getGroupsUnsorted(now)].sort((a, b) => {
                return String(a.title ?? "").localeCompare(String(b.title ?? ""));
            });

            this.msLastAlpha = now;
        }

        return this.groupsSortedAlpha;
    }

    getGroupsPosition(now) {
        if (!this.groupsSortedPosition.length || now - this.msLastPosition > this.msThreshold) {
            this.groupsSortedPosition = [...this.getGroupsUnsorted(now)].sort((a, b) => {
                const aY = Math.floor((a._pos?.[1] ?? 0) / 30);
                const bY = Math.floor((b._pos?.[1] ?? 0) / 30);

                if (aY === bY) {
                    const aX = Math.floor((a._pos?.[0] ?? 0) / 30);
                    const bX = Math.floor((b._pos?.[0] ?? 0) / 30);

                    return aX - bX;
                }

                return aY - bY;
            });

            this.msLastPosition = now;
        }

        return this.groupsSortedPosition;
    }

    getGroups(sort) {
        const now = Date.now();

        if (sort === "alphanumeric") {
            return this.getGroupsAlpha(now);
        }

        if (sort === "position") {
            return this.getGroupsPosition(now);
        }

        return this.getGroupsUnsorted(now);
    }
}

export const SERVICE = new FastGroupsService();