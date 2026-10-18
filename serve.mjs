// A tiny static file server for the Sprout app — no dependencies.
//
// The browser app needs the project folder to be served over HTTP: the page
// fetches `training.json` and `model.json`, and browsers block `fetch` for
// `file://` pages.
//
// Usage:
//   node serve.mjs            # http://localhost:8000
//   node serve.mjs 8080       # a different port
//   PORT=3000 node serve.mjs  # or via the environment

import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

const MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".md": "text/markdown; charset=utf-8"
};

// Start the server. `port` may be 0, which asks the operating system for a
// free port — `ready` then resolves once the port is known.
export function startServer(port = 8000, host = "0.0.0.0") {
    const server = http.createServer((request, response) => {
        const requested = decodeURIComponent((request.url || "/").split("?")[0]);
        const relative = requested === "/" ? "index.html" : requested.replace(/^\/+/, "");
        const filePath = path.resolve(here, relative);

        // Never serve anything outside the project folder.
        if (!filePath.startsWith(here + path.sep)) {
            response.writeHead(403, { "Content-Type": "text/plain" });
            response.end("Forbidden");
            return;
        }

        fs.readFile(filePath, (error, data) => {
            if (error) {
                response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
                response.end(`Not found: ${requested}`);
                return;
            }

            response.writeHead(200, {
                "Content-Type": MIME_TYPES[path.extname(filePath).toLowerCase()] || "application/octet-stream",
                "Cache-Control": "no-cache"
            });
            response.end(data);
        });
    });

    const ready = new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(port, host, () => {
            const address = server.address();
            resolve({
                port: address.port,
                url: `http://localhost:${address.port}/`
            });
        });
    });

    return { server, ready };
}

// Only start listening when this file is run directly, so other modules
// (such as the app smoke test) can import it.
const isDirectRun = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isDirectRun) {
    const port = Number(process.argv[2] || process.env.PORT || 8000);
    const host = process.env.HOST || "0.0.0.0";

    const { ready } = startServer(port, host);
    ready.then(info => {
        console.log(`Sprout Slop Filter is running at ${info.url}`);
        console.log(`Serving ${here}`);
    });
}
