/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      allowedOrigins: ['localhost:3000', 'final-app-deploy-2.onrender.com'],
    },
  },
  env: {
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    PINECONE_API_KEY: process.env.PINECONE_API_KEY,
    PINECONE_ENVIRONMENT: process.env.PINECONE_ENVIRONMENT,
    PINECONE_INDEX: process.env.PINECONE_INDEX,
    NEXT_PUBLIC_API_URL: process.env.NODE_ENV === 'development' 
      ? 'http://localhost:5001'
      : 'https://final-app-deploy-6d92.onrender.com',
  },
  async rewrites() {
    return process.env.NODE_ENV === 'development' ? [] : [
      {
        source: '/api/:path*',
        destination: 'https://final-app-deploy-6d92.onrender.com/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
