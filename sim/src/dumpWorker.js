// Runs the (potentially long) snapshot generation off the UI thread.
// Receives a dump config, streams progress, posts back the serialized bundle.
import { generateBundle, serializeBundle } from "./sim.js";

self.onmessage = (e) => {
  const config = e.data;
  try {
    const bundle = generateBundle(config, (p) => self.postMessage({ type: "progress", ...p }));
    const bytes = serializeBundle(bundle);
    // transfer the underlying buffer (zero-copy) back to the main thread
    self.postMessage(
      { type: "done", buffer: bytes.buffer, meta: bundle.meta },
      [bytes.buffer]
    );
  } catch (err) {
    self.postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
  }
};
