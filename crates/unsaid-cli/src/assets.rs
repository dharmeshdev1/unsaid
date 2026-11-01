//! The web interface, compiled into the binary.

use unsaid_server::Asset;

const JS: &str = "text/javascript; charset=utf-8";

/// Every file the web interface needs.
pub const ASSETS: &[Asset] = &[
    Asset {
        path: "/index.html",
        content_type: "text/html; charset=utf-8",
        bytes: include_bytes!("../../../web/index.html"),
    },
    Asset {
        path: "/style.css",
        content_type: "text/css; charset=utf-8",
        bytes: include_bytes!("../../../web/style.css"),
    },
    Asset {
        path: "/app.js",
        content_type: JS,
        bytes: include_bytes!("../../../web/app.js"),
    },
    Asset {
        path: "/engine.worker.js",
        content_type: JS,
        bytes: include_bytes!("../../../web/engine.worker.js"),
    },
    Asset {
        path: "/fonts/besley.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../../../web/fonts/besley.woff2"),
    },
    Asset {
        path: "/fonts/atkinson-400.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../../../web/fonts/atkinson-400.woff2"),
    },
    Asset {
        path: "/fonts/atkinson-700.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../../../web/fonts/atkinson-700.woff2"),
    },
    Asset {
        path: "/pkg/unsaid_wasm.js",
        content_type: JS,
        bytes: include_bytes!("../../../web/pkg/unsaid_wasm.js"),
    },
    Asset {
        path: "/pkg/unsaid_wasm_bg.wasm",
        content_type: "application/wasm",
        bytes: include_bytes!("../../../web/pkg/unsaid_wasm_bg.wasm"),
    },
];
