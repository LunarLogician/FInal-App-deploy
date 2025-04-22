import { OpenAI } from "openai";
import pinecone from "./pinecone";
import { v4 as uuidv4 } from "uuid";

// Init OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const indexName = process.env.PINECONE_INDEX_NAME!;
const namespace = "default"; // Or customize if needed

async function getEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    input: text,
    model: "text-embedding-3-small"
  });
  return response.data[0].embedding;
}

export async function embedAndUpload(docText: string, docName: string) {
  // 1. Chunk the document
  const chunkSize = 500;
  const textChunks = docText.match(new RegExp(`.{1,${chunkSize}}`, 'g')) || [];

  // 2. Embed chunks
  const embeddings = await Promise.all(
    textChunks.map(async (chunk) => {
      const embedding = await getEmbedding(chunk);
      return embedding;
    })
  );

  // 3. Prepare vectors
  const vectors = textChunks.map((chunk, i) => ({
    id: uuidv4(),
    values: embeddings[i],
    metadata: {
      text: chunk,
      chunk_index: i,
      doc_name: docName
    }
  }));

  // 4. Upsert to Pinecone
  const pineconeIndex = pinecone.Index(indexName);
  await pineconeIndex.upsert(vectors);

  return { success: true, chunksUploaded: vectors.length };
}
