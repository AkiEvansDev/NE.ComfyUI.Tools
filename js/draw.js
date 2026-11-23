import { app } from "../../scripts/app.js";

export function isLowQuality() {
    return (app.canvas?.ds?.scale ?? 1) <= 0.5;
}

function binarySearch(max, getValue, match) {
    let min = 0;
    while (min <= max) {
        let guess = Math.floor((min + max) / 2);
        const compareVal = getValue(guess);
        if (compareVal === match)
            return guess;
        if (compareVal < match)
            min = guess + 1;
        else
            max = guess - 1;
    }
    return max;
}

export function fitString(ctx, str, maxWidth) {
    str = String(str ?? "");

    if (maxWidth <= 0) {
        return "";
    }

    const ellipsis = "...";
    const ellipsisWidth = measureText(ctx, ellipsis);
    const width = measureText(ctx, str);

    if (width <= maxWidth) {
        return str;
    }

    if (maxWidth <= ellipsisWidth) {
        return ellipsis;
    }

    const index = binarySearch(
        str.length,
        (guess) => measureText(ctx, str.substring(0, guess)),
        maxWidth - ellipsisWidth,
    );

    return str.substring(0, Math.max(0, index)) + ellipsis;
}

export function measureText(ctx, str) {
    return ctx.measureText(str).width;
}

function roundedRect(ctx, x, y, width, height, radius) {
    if (ctx.roundRect) {
        ctx.roundRect(x, y, width, height, radius);
        return;
    }

    const r = Array.isArray(radius) ? radius[0] : radius;
    ctx.rect(x, y, width, height);
}

export function drawRoundedRectangle(ctx, options) {
    const lowQuality = isLowQuality();
    options = { ...options };
    ctx.strokeStyle = options.colorStroke || LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = options.colorBackground || LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    roundedRect(ctx, options.posX, options.posY, options.width, options.height, lowQuality ? [0] : options.borderRadius ? [options.borderRadius] : [options.height * 0.5]);
    ctx.fill();
    !lowQuality && ctx.stroke();
}

export function drawTogglePart(ctx, options) {
    const lowQuality = isLowQuality();
    ctx.save();
    const { posX, posY, height, value } = options;
    const toggleRadius = height * 0.36;
    const toggleBgWidth = height * 1.5;
    const alpha = app.canvas?.editor_alpha ?? 1;
    if (!lowQuality) {
        ctx.beginPath();
        roundedRect(ctx, posX + 4, posY + 4, toggleBgWidth - 8, height - 8, [height * 0.5]);
        ctx.globalAlpha = alpha * 0.25;
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fill();
        ctx.globalAlpha = alpha;
    }
    ctx.fillStyle = value === true ? "#89B" : "#888";
    const toggleX = lowQuality || value === false
        ? posX + height * 0.5
        : value === true
            ? posX + height
            : posX + height * 0.75;
    ctx.beginPath();
    ctx.arc(toggleX, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return [posX, toggleBgWidth];
}

export function drawNumberWidgetPart(ctx, options) {
    const arrowWidth = 9;
    const arrowHeight = 10;
    const innerMargin = 3;
    const numberWidth = 32;
    const xBoundsArrowLess = [0, 0];
    const xBoundsNumber = [0, 0];
    const xBoundsArrowMore = [0, 0];
    ctx.save();
    let posX = options.posX;
    const { posY, height, value, textColor } = options;
    const midY = posY + height / 2;
    if (options.direction === -1) {
        posX = posX - arrowWidth - innerMargin - numberWidth - innerMargin - arrowWidth;
    }
    ctx.fill(new Path2D(`M ${posX} ${midY} l ${arrowWidth} ${arrowHeight / 2} l 0 -${arrowHeight} L ${posX} ${midY} z`));
    xBoundsArrowLess[0] = posX;
    xBoundsArrowLess[1] = arrowWidth;
    posX += arrowWidth + innerMargin;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const oldTextColor = ctx.fillStyle;
    if (textColor) {
        ctx.fillStyle = textColor;
    }
    ctx.fillText(fitString(ctx, Number(value ?? 0).toFixed(2), numberWidth), posX + numberWidth / 2, midY);
    ctx.fillStyle = oldTextColor;
    xBoundsNumber[0] = posX;
    xBoundsNumber[1] = numberWidth;
    posX += numberWidth + innerMargin;
    ctx.fill(new Path2D(`M ${posX} ${midY - arrowHeight / 2} l ${arrowWidth} ${arrowHeight / 2} l -${arrowWidth} ${arrowHeight / 2} v -${arrowHeight} z`));
    xBoundsArrowMore[0] = posX;
    xBoundsArrowMore[1] = arrowWidth;
    ctx.restore();
    return [xBoundsArrowLess, xBoundsNumber, xBoundsArrowMore];
}
drawNumberWidgetPart.WIDTH_TOTAL = 9 + 3 + 32 + 3 + 9;

export function drawNodeWidget(ctx, options) {
    const lowQuality = isLowQuality();
    const data = {
        width: options.width,
        height: options.height,
        posY: options.posY,
        lowQuality,
        margin: 15,
        colorOutline: LiteGraph.WIDGET_OUTLINE_COLOR,
        colorBackground: LiteGraph.WIDGET_BGCOLOR,
        colorText: LiteGraph.WIDGET_TEXT_COLOR,
        colorTextSecondary: LiteGraph.WIDGET_SECONDARY_TEXT_COLOR,
    };
    ctx.strokeStyle = options.colorStroke || data.colorOutline;
    ctx.fillStyle = options.colorBackground || data.colorBackground;
    ctx.beginPath();
    roundedRect(ctx, data.margin, data.posY, data.width - data.margin * 2, data.height, lowQuality ? [0] : options.borderRadius ? [options.borderRadius] : [options.height * 0.5]);
    ctx.fill();
    if (!lowQuality) {
        ctx.stroke();
    }
    return data;
}
