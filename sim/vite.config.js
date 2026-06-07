import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Worker uses ES modules so it can `import` from sim.js directly.
export default defineConfig({
  plugins: [react()],
  worker: { format: "es" },
});
