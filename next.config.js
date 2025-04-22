/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      allowedOrigins: ['localhost:3000', '192.168.100.44:3000'],
    },
  },
  env: {
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    PINECONE_API_KEY: process.env.PINECONE_API_KEY,
    PINECONE_ENVIRONMENT: process.env.PINECONE_ENVIRONMENT,
    PINECONE_INDEX: process.env.PINECONE_INDEX,
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://final-rag-project-dep-4.onrender.com',
    NEXT_PUBLIC_ESG_URL: process.env.NEXT_PUBLIC_ESG_URL || 'https://final-rag-project-dep-4.onrender.com',
    NEXT_PUBLIC_ADDITIONAL_URL: process.env.NEXT_PUBLIC_ADDITIONAL_URL || 'https://final-rag-project-dep-4.onrender.com',
  },
  async rewrites() {
    const rewrites = [
      {
        source: '/api/:path*',
        destination: 'https://final-rag-project-dep-4.onrender.com/:path*',
      },
      {
        source: '/esg/:path*',
        destination: 'https://final-rag-project-dep-4.onrender.com/:path*',
      },
      {
        source: '/additional/:path*',
        destination: 'https://final-rag-project-dep-4.onrender.com/:path*',
      },
    ];

    console.log("\n🔐 Env Variables:");
    console.log("OPENAI_API_KEY:", process.env.OPENAI_API_KEY);
    console.log("PINECONE_API_KEY:", process.env.PINECONE_API_KEY);
    console.log("PINECONE_ENVIRONMENT:", process.env.PINECONE_ENVIRONMENT);
    console.log("PINECONE_INDEX:", process.env.PINECONE_INDEX);
    console.log("NEXT_PUBLIC_API_URL:", process.env.NEXT_PUBLIC_API_URL);
    console.log("NEXT_PUBLIC_ESG_URL:", process.env.NEXT_PUBLIC_ESG_URL);
    console.log("NEXT_PUBLIC_ADDITIONAL_URL:", process.env.NEXT_PUBLIC_ADDITIONAL_URL);

    console.log("\n🔁 Rewrite Rules:");
    rewrites.forEach((r) => {
      console.log(`  ${r.source} → ${r.destination}`);
    });

    return rewrites;
  },
};

module.exports = nextConfig;
