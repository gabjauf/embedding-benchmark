import { pipeline } from '@huggingface/transformers';
import { parentPort } from 'node:worker_threads';

const pipe = await pipeline('feature-extraction', 'mixedbread-ai/mxbai-embed-large-v1', {
  dtype: 'q8'
});

async function embed(text) {
  const res = await pipe(text[0]);
  return res.tolist();
}

parentPort.on('message', async (data) => {
  const { messageId, args, workerId } = data;
  const result = await embed(...args);
  parentPort.postMessage({
    result,
    messageId
  });
});
