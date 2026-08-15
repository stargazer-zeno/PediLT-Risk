import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `npm run dev`, proxy API and compatibility routes to FastAPI.
const backend = process.env.VITE_BACKEND_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: backend, changeOrigin: true },
      "/health": { target: backend, changeOrigin: true },
      "/predict": { target: backend, changeOrigin: true },
      "/batch_predict": { target: backend, changeOrigin: true },
    },
  },
});
