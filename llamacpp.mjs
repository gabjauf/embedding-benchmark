import { getLlama, LlamaEmbedding } from 'node-llama-cpp';
import path from "node:path";

const __dirname = path.resolve(path.dirname(''));

const llamaPath = __dirname + '/mxbai-embed-large-v1.Q8_0.gguf';

const llama = await getLlama();
const model = await llama.loadModel({
  modelPath: llamaPath
});
const context = await model.createEmbeddingContext();

export async function llamacpp(documents) {
  return Promise.all(
    documents.map((document) => {
      return context.getEmbeddingFor(document);
    })
  );
}

