const withPWA = require("next-pwa");

const isDev = process.env.NODE_ENV === "development";

/** @type {import('next-pwa').PWAConfig} */
const pwa = withPWA({
  dest: "public",
  disable: isDev,
  register: true,
  skipWaiting: true,
  // Take control of already-open tabs immediately so users never get stuck on
  // a stale build after a deploy (otherwise the new SW only controls fresh
  // navigations and the open tab keeps running old JS until full close).
  clientsClaim: true,
  buildExcludes: [/middleware-manifest\.json$/],
  /** Do not precache TF.js assets (large, change often; stale precache + SW bugs caused empty shards). */
  publicExcludes: ["!noprecache/**/*", "!model/**"],
  runtimeCaching: [
    {
      /** Same-origin TF.js assets: serve from SW cache first so hard reload still avoids re-fetch when possible. */
      urlPattern: ({ request, url }) =>
        request.method === "GET" && url.pathname.startsWith("/model/"),
      handler: "CacheFirst",
      options: {
        cacheName: "pawdetect-workbox-model",
        expiration: { maxEntries: 32, maxAgeSeconds: 365 * 24 * 60 * 60 },
        cacheableResponse: { statuses: [0, 200] },
      },
    },
    {
      urlPattern: /^https:\/\/fonts\.(?:googleapis|gstatic)\.com\/.*/i,
      handler: "CacheFirst",
      options: {
        cacheName: "google-fonts",
        expiration: { maxEntries: 4, maxAgeSeconds: 365 * 24 * 60 * 60 },
      },
    },
    {
      urlPattern: /\.(?:eot|otf|ttc|ttf|woff|woff2|font.css)$/i,
      handler: "StaleWhileRevalidate",
      options: { cacheName: "static-font-assets" },
    },
    {
      urlPattern: /\.(?:jpg|jpeg|gif|png|svg|ico|webp)$/i,
      handler: "StaleWhileRevalidate",
      options: { cacheName: "static-image-assets" },
    },
    {
      urlPattern: /\/_next\/image\?url=.+$/i,
      handler: "StaleWhileRevalidate",
      options: { cacheName: "next-image" },
    },
  ],
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async headers() {
    return [
      {
        source: "/model/infer.json",
        headers: [
          { key: "Cache-Control", value: "public, max-age=300, stale-while-revalidate=86400" },
          { key: "Content-Type", value: "application/json; charset=utf-8" },
        ],
      },
      {
        source: "/model/model.json",
        headers: [
          { key: "Cache-Control", value: "public, max-age=300, stale-while-revalidate=86400" },
          { key: "Content-Type", value: "application/json; charset=utf-8" },
        ],
      },
      {
        source: "/model/group1-shard.bin",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=604800, stale-while-revalidate=2592000",
          },
          { key: "Content-Type", value: "application/octet-stream" },
          { key: "Content-Disposition", value: "inline" },
        ],
      },
    ];
  },
  webpack: (config) => {
    config.ignoreWarnings ??= [];
    config.ignoreWarnings.push({ module: /node_modules\/@tensorflow/ });
    return config;
  },
};

module.exports = pwa(nextConfig);
