import { pipeline } from '@huggingface/transformers';

const pipe = await pipeline('feature-extraction', 'mixedbread-ai/mxbai-embed-large-v1', {
  dtype: 'q8'
});

export async function transformerJS(text) {
  const res = await pipe(text);
  return res.tolist();
}
