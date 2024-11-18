import { boxplot, bench, run } from "mitata";
import workerpool from "workerpool";
import path from "node:path";
import { CallWorker } from "./call-worker.mts";
import { AutoTokenizer } from "@huggingface/transformers";
import { onnxWebEmbed } from "./onnx-web.mjs";
import { onnxNodeEmbed } from "./onnx-node.mjs";
import { transformerJS } from "./transformer.mjs";
import { OllamaEmbeddings } from "@langchain/ollama";
import { llamacpp } from "./llamacpp.mjs";
const __dirname = path.resolve(path.dirname(""));
const poolOnnx = workerpool.pool(__dirname + "/worker-onnx-node-pool.mjs", {
  workerOpts: {
    type: "module",
  },
  minWorkers: 4,
  maxWorkers: 4,
  workerTerminateTimeout: 60000,
});

const poolTransformer = workerpool.pool(
  __dirname + "/worker-transformer-pool.mjs",
  {
    workerOpts: {
      type: "module",
    },
    minWorkers: 4,
    maxWorkers: 4,
    workerTerminateTimeout: 60000,
  }
);

const ollamaEmbeder = new OllamaEmbeddings({
  // model: 'nomic-embed-text'
  // model: 'snowflake-arctic-embed:137m'
  model: "mxbai-embed-large:latest",
});

const texts = [
  `Home   Blog    About me      Go to Astro's GitHub repo          About Me           Gabriel Jauffret   Full-stack developer \nTypescript Full-stack developer with over 5+ years of experience\n            building scalable web applications and leading innovative projects.I’ve held roles as CTO and co-founder, guiding startups from\n            MVP development to market launch. My expertise spans full-stack\n            development, AI-driven solutions, and cloud platforms, with a strong`,
];

boxplot(() => {
  bench("pool onnx", async () => {
    await Promise.all([
      poolOnnx.exec("embed", [texts]),
      poolOnnx.exec("embed", [texts]),
      poolOnnx.exec("embed", [texts]),
      poolOnnx.exec("embed", [texts]),
    ]);
  });
  bench("pool transformer.js", async () => {
    await Promise.all([
      poolTransformer.exec("embed", [texts]),
      poolTransformer.exec("embed", [texts]),
      poolTransformer.exec("embed", [texts]),
      poolTransformer.exec("embed", [texts]),
    ]);
  });
  bench("onnx Node", async () => {
    await Promise.all([
      onnxNodeEmbed(texts),
      onnxNodeEmbed(texts),
      onnxNodeEmbed(texts),
      onnxNodeEmbed(texts),
    ]);
  });

  bench("onnx Web", async () => {
    await Promise.all([
      onnxWebEmbed(texts),
      onnxWebEmbed(texts),
      onnxWebEmbed(texts),
      onnxWebEmbed(texts),
    ]);
  });
  bench("transformer.js", async () =>
    Promise.all([
      transformerJS(texts),
      transformerJS(texts),
      transformerJS(texts),
      transformerJS(texts),
    ])
  );
  bench("LlamaCPP", async () =>
    Promise.all([
      llamacpp(texts),
      llamacpp(texts),
      llamacpp(texts),
      llamacpp(texts),
    ])
  );
  bench("Ollama", async () =>
    Promise.all([
      ollamaEmbeder.embedDocuments(texts),
      ollamaEmbeder.embedDocuments(texts),
      ollamaEmbeder.embedDocuments(texts),
      ollamaEmbeder.embedDocuments(texts),
    ])
  );
});

await run();
await poolOnnx.terminate();
await poolTransformer.terminate();
