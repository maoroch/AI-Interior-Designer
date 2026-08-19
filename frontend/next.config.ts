import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // standalone-сборка нужна для лёгкого прод-образа (Dockerfile копирует
  // только .next/standalone + .next/static + public, без node_modules целиком).
  output: "standalone",
};

export default nextConfig;
