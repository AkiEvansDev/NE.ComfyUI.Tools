export const NE_PREFIX = "NE.";

export const NodeNames = Object.freeze({
    AnySwitch: "NE.AnySwitch",
    AnyTypeSwitch: "NE.AnyTypeSwitch",
    DetailerStep: "NE.DetailerStep",
    DisplayAny: "NE.DisplayAny",
    FloatList: "NE.FloatList",
    GroupsMuter: "NE.GroupsMuter",
    IntList: "NE.IntList",
    LorasList: "NE.LorasList",
    LorasLoader: "NE.LorasLoader",
    WanExpertLoras: "NE.WanExpertLoras",
    PreviewVideo: "NE.PreviewVideo",
    PromptConfig: "NE.PromptConfig",
    Range: "NE.Range",
    RangeList: "NE.RangeList",
    VideoModelConfig: "NE.VideoModelConfig",
    Reroute: "NE.Reroute",
    StringList: "NE.StringList",
    Text: "NE.Text",
    ToStringConcat: "NE.ToStringConcat",
    XYRange: "NE.XYRange",
    XYZRange: "NE.XYZRange",
});

export const ExtensionNames = Object.freeze({
    AutoCompleter: "NE.AutoCompleter",
    DisplayAny: "NE.DisplayAny",
    Grid: "NE.Grid",
    Lists: "NE.Lists",
    LorasLoader: "NE.LorasLoader",
    Menu: "NE.Menu",
    GroupsMuter: "NE.GroupsMuter",
    NodeFixer: "NE.NodeFixer",
    NodeStyle: "NE.NodeStyle",
    Preview: "NE.Preview",
    ProgressBar: "NE.ProgressBar",
    Prompt: "NE.Prompt",
    Range: "NE.Range",
    Reroute: "NE.Reroute",
    Seed: "NE.Seed",
    ToStringConcat: "NE.ToStringConcat",
    AnySwitch: "NE.AnySwitch",
    Text: "NE.Text",
});

export const WidgetNames = Object.freeze({
    Groups: "groups",
    List: "list",
    LoraPrefix: "lora_",
    Output: "output",
    SeedValue: "seed_value",
    Value: "value",
});

export const EventNames = Object.freeze({
    DisplayFeedback: "ne-display-node-feedback",
    RangeFeedback: "ne-range-node-feedback",
    SeedFeedback: "ne-seed-node-feedback",
    XYRangeFeedback: "ne-xy-range-node-feedback",
    XYZRangeFeedback: "ne-xyz-range-node-feedback",
});

export const SettingNames = Object.freeze({
    GridLineLightness: "NE.Canvas.GridLineLightness",
    GridMinorLineLightness: "NE.Canvas.GridMinorLineLightness",
    GridMinimumZoom: "NE.Canvas.GridMinimumZoom",
    PreviewMaxHeight: "NE.Preview.MaxHeight",
});

export const Commands = Object.freeze({
    RestartServer: "ne-tools.restart-server",
    UnloadModels: "ne-tools.unload-models",
    FreeModelCache: "ne-tools.free-model-cache",
});

export const HeaderNames = Object.freeze({
    Reboot: "X-NE-Reboot",
    RebootValue: "1",
});
