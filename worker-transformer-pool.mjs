import workerpool from 'workerpool';
import { transformerJS } from './transformer.mjs';

workerpool.worker({
  embed: transformerJS
});
