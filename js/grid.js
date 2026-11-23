import { app } from "../../scripts/app.js";
import { ExtensionNames, SettingNames } from "./constants.js";

const DEFAULT_LINE_LIGHTNESS = 22;
const DEFAULT_MINOR_LINE_LIGHTNESS = 15;
const DEFAULT_MINIMUM_ZOOM = 100;
const CANVAS_WAIT_FRAMES = 600;

// The tile holds two families of lines: the cell borders are the darkest pixels, the 10x10 sub-grid
// inside them is a few levels lighter. Anything at least this far towards the darkest counts as a border.
const MAJOR_LINE_RATIO = 0.75;

// The canvas grid is a tiled image, not drawn lines: the cell is filled with one grey and the lines are
// painted a few levels *darker*, which on a dark theme is what makes them so hard to see. Repainting the
// non-fill pixels keeps the tile's geometry exactly as it was and only changes how the lines read.
function recolorGrid(source, lightness, minorLightness) {
    const image = new Image();
    image.src = source;

    return image.decode().then(() => {
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;

        const context = canvas.getContext("2d", { willReadFrequently: true });
        context.drawImage(image, 0, 0);

        const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
        const fill = dominantLevel(pixels.data);
        const major = toLevel(lightness);
        const minor = toLevel(minorLightness);

        // How far the darkest pixel sits from the fill: the scale the two families are told apart on.
        let deepest = 0;
        for (let index = 0; index < pixels.data.length; index += 4) {
            deepest = Math.max(deepest, Math.abs(fill - pixels.data[index]));
        }

        for (let index = 0; index < pixels.data.length; index += 4) {
            const distance = Math.abs(fill - pixels.data[index]);
            if (distance === 0) {
                continue;
            }

            const level = distance / deepest >= MAJOR_LINE_RATIO ? major : minor;
            pixels.data[index] = level;
            pixels.data[index + 1] = level;
            pixels.data[index + 2] = level;
        }

        context.putImageData(pixels, 0, 0);
        return canvas.toDataURL();
    });
}

function toLevel(lightness) {
    return Math.round((Math.max(0, Math.min(100, lightness)) / 100) * 255);
}

function dominantLevel(data) {
    const counts = new Uint32Array(256);
    for (let index = 0; index < data.length; index += 4) {
        counts[data[index]]++;
    }

    let best = 0;
    for (let level = 1; level < counts.length; level++) {
        if (counts[level] > counts[best]) {
            best = level;
        }
    }

    return best;
}

function lineLightness() {
    const value = app.extensionManager?.setting?.get(SettingNames.GridLineLightness);
    return typeof value === "number" ? value : DEFAULT_LINE_LIGHTNESS;
}

function minorLineLightness() {
    const value = app.extensionManager?.setting?.get(SettingNames.GridMinorLineLightness);
    return typeof value === "number" ? value : DEFAULT_MINOR_LINE_LIGHTNESS;
}

function minimumZoom() {
    const value = app.extensionManager?.setting?.get(SettingNames.GridMinimumZoom);
    return (typeof value === "number" ? value : DEFAULT_MINIMUM_ZOOM) / 100;
}

// A color palette assigns background_image whenever it loads, which would drop the recolored tile.
// Owning the property means the source is recolored again no matter who writes it or when.
function hookCanvas(canvas) {
    if (!canvas || canvas.__ne_gridPatched) {
        return;
    }

    let source = canvas.background_image;
    let painted = null;

    // The grid is faded by zoom: drawBackCanvas uses globalAlpha = 1 - 0.5 / scale, which at a working
    // zoom of 56% leaves it at 10% opacity and at 50% hides it outright. No amount of recoloring shows
    // through that, so the fade is switched off and the lightness slider becomes the only control.
    canvas.zoom_modify_alpha = false;

    const repaint = () => {
        if (!source) {
            return;
        }

        recolorGrid(source, lineLightness(), minorLineLightness())
            .then((result) => {
                painted = result;

                // drawBackCanvas builds the fill pattern once and then reuses it forever: it only rebuilds
                // when _pattern is null, so swapping background_image alone repaints nothing at all.
                canvas._pattern = null;
                canvas._pattern_img = null;
                canvas.setDirty(true, true);
            })
            .catch((error) => console.error("[NE.Grid] failed to recolor the grid tile", error));
    };

    Object.defineProperty(canvas, "background_image", {
        configurable: true,
        // drawBackCanvas reads this every frame and skips the whole background block when it is empty,
        // so the zoom cutoff needs no listener: the getter simply hides the tile below the threshold.
        get: () => ((canvas.ds?.scale ?? 1) > minimumZoom() ? painted ?? source : null),
        set(value) {
            if (value === painted) {
                return;
            }

            source = value;
            painted = null;
            repaint();
        },
    });

    canvas.__ne_gridPatched = true;
    canvas.__ne_gridRepaint = repaint;
    repaint();
}

// setup() does not guarantee the canvas exists yet, and returning quietly when it does not is how the
// first version of this ended up doing nothing at all with no error to show for it.
function hookWhenReady(remaining = CANVAS_WAIT_FRAMES) {
    if (app.canvas) {
        hookCanvas(app.canvas);
        return;
    }

    if (remaining <= 0) {
        console.warn("[NE.Grid] no canvas appeared, the grid was left as it was");
        return;
    }

    requestAnimationFrame(() => hookWhenReady(remaining - 1));
}

app.registerExtension({
    name: ExtensionNames.Grid,

    settings: [
        {
            id: SettingNames.GridLineLightness,
            name: "Grid major line lightness",
            category: ["NE Tools", "Canvas", "Grid major line lightness"],
            tooltip: "Brightness of the thick cell borders: 0 is black, 100 is white.",
            type: "slider",
            attrs: { min: 0, max: 100, step: 1 },
            defaultValue: DEFAULT_LINE_LIGHTNESS,
            onChange: () => app.canvas?.__ne_gridRepaint?.(),
        },
        {
            id: SettingNames.GridMinorLineLightness,
            name: "Grid minor line lightness",
            category: ["NE Tools", "Canvas", "Grid minor line lightness"],
            tooltip: "Brightness of the fine sub-grid inside each cell. Keep it near the background to stop it from shouting.",
            type: "slider",
            attrs: { min: 0, max: 100, step: 1 },
            defaultValue: DEFAULT_MINOR_LINE_LIGHTNESS,
            onChange: () => app.canvas?.__ne_gridRepaint?.(),
        },
        {
            id: SettingNames.GridMinimumZoom,
            name: "Grid minimum zoom",
            category: ["NE Tools", "Canvas", "Grid minimum zoom"],
            tooltip: "Zoom percentage above which the grid is drawn: at 100 it shows from 101% and is hidden at 99%.",
            type: "slider",
            attrs: { min: 0, max: 400, step: 1 },
            defaultValue: DEFAULT_MINIMUM_ZOOM,
            onChange: () => app.canvas?.setDirty(true, true),
        },
    ],

    async setup() {
        hookWhenReady();
    },
});

// Also recolor the static default, so a canvas created after this module loads starts out right.
if (typeof LGraphCanvas !== "undefined" && !LGraphCanvas.__ne_gridDefaultPatched) {
    LGraphCanvas.__ne_gridDefaultPatched = true;

    recolorGrid(LGraphCanvas.DEFAULT_BACKGROUND_IMAGE, DEFAULT_LINE_LIGHTNESS, DEFAULT_MINOR_LINE_LIGHTNESS)
        .then((result) => {
            LGraphCanvas.DEFAULT_BACKGROUND_IMAGE = result;
        })
        .catch((error) => console.error("[NE.Grid] failed to recolor the default grid tile", error));
}
