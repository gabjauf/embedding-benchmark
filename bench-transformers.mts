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

const texts = [
  `Home   Blog    About me      Go to Astro's GitHub repo          About Me           Gabriel Jauffret   Full-stack developer \nTypescript Full-stack developer with over 5+ years of experience\n            building scalable web applications and leading innovative projects.I’ve held roles as CTO and co-founder, guiding startups from\n            MVP development to market launch. My expertise spans full-stack\n            development, AI-driven solutions, and cloud platforms, with a strong`,
];

boxplot(() => {
  bench("transformer.js", async () =>
    Promise.all([
      transformerJS(texts),
      transformerJS(texts),
      transformerJS(texts),
      transformerJS(texts),
    ])
  );
  bench("1 only transformer.js", async () =>
    Promise.all([transformerJS(texts)])
  );
  bench("onnx Node", async () => {
    await Promise.all([
      onnxNodeEmbed(texts),
      onnxNodeEmbed(texts),
      onnxNodeEmbed(texts),
      onnxNodeEmbed(texts),
    ]);
  });
  bench("1 only onnx node", async () => Promise.all([onnxNodeEmbed(texts)]));
});

await run();
