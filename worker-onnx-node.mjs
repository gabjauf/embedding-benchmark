import { parentPort } from 'node:worker_threads';
import { onnxNodeEmbed } from "./onnx-node.mjs";

parentPort.on('message', async (data) => {
  const { messageId, args } = data;
  const result = await onnxNodeEmbed(...args);

  parentPort.postMessage({
    result,
    messageId
  });
});
