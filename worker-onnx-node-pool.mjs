import workerpool from 'workerpool';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { AutoTokenizer } from '@huggingface/transformers';

const session = await InferenceSession.create('./model_quantized.onnx');
const tokenizer = await AutoTokenizer.from_pretrained('mixedbread-ai/mxbai-embed-large-v1');

// use an async context to call onnxruntime functions.
async function embed(texts) {
  try {
    // prepare feeds. use model input names as keys.
    const feeds = await tokenizer(texts);

    // feed inputs and run
    const results = await session.run(feeds);

    return results;
  } catch (e) {
    console.error(`failed to inference ONNX model: ${e}.`);
  }
}

workerpool.worker({
  embed
});
