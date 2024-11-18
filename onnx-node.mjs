import { AutoTokenizer } from "@huggingface/transformers";
import { env, InferenceSession, Tensor } from "onnxruntime-node";

const session = await InferenceSession.create("./model_quantized.onnx");

const tokenizer = await AutoTokenizer.from_pretrained(
  "mixedbread-ai/mxbai-embed-large-v1"
);
export async function onnxNodeEmbed(texts) {
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
