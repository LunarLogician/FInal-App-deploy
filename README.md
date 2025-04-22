# Resonate AI Chat Interface

A modern chat interface built with Next.js and OpenAI, featuring real-time streaming responses and a clean, Google-inspired design.

## Features

- Real-time streaming chat responses
- Modern, responsive UI with a three-column layout
- OpenAI GPT-4 integration
- Document source management
- Tools & Insights panel


## Tech Stack

- Next.js 14 with App Router
- TypeScript
- Tailwind CSS
- OpenAI API
- Edge Runtime for optimal performance

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/antoniadumitriu/reports_rag.git
cd reports_rag
```

2. Install dependencies:
```bash
pnpm install
```

3. Create a `.env.local` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

4. Run the development server:
```bash
pnpm dev
```

5. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key

## Development

The project uses:
- TypeScript for type safety
- Tailwind CSS for styling
- Edge Runtime for API routes
- OpenAI's streaming API for real-time responses

## License

MIT
