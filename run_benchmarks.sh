set -x
# Setup the output directory
#backend="eager"
#backend="aot_eager"
#backend="inductor"
backend="torchxla_trace_once"
models="huggingface"
#models="timm_models"
#models="torchbench"
precision=float32
#precision=amp
#precision=float16
#device=cuda
device=xla
rm -rf ${models}_${backend}_logs/${backend}_${models}_${precision}_${device}_performance.csv
mkdir -p ${models}_${backend}_logs
# Commands for huggingface for device=${device}, dtype=${precision} for training and for performance testing
#GPU_NUM_DEVICES=1 python benchmarks/dynamo/${models}.py --performance --${precision} -d${device} --output=${models}_${backend}_logs/${backend}_${models}_${precision}_${device}_performance.csv --output-directory=./ --backend=${backend}   --no-skip --dashboard -x Reformer -x GPTNeoForCausalLM -x GPTJForQuestionAnswering -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForSequenceClassification --batch_size 1 "$@"
#models="huggingface"
#GPU_NUM_DEVICES=1 python benchmarks/dynamo/${models}.py --${precision} -d${device} --output=${models}_${backend}_logs/${backend}_${models}_${precision}_${device}_performance.csv --output-directory=./ --backend=${backend}   --no-skip --dashboard -x Reformer -x GPTNeoForCausalLM -x GPTJForQuestionAnswering -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForSequenceClassification --batch_size 1 "$@"
#models="timm_models"
#GPU_NUM_DEVICES=1 python benchmarks/dynamo/${models}.py --${precision} -d${device} --output=${models}_${backend}_logs/${backend}_${models}_${precision}_${device}_performance.csv --output-directory=./ --backend=${backend}   --no-skip --dashboard -x Reformer -x GPTNeoForCausalLM -x GPTJForQuestionAnswering -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForSequenceClassification --batch_size 1 "$@"
#models="torchbench"
#GPU_NUM_DEVICES=1 python benchmarks/dynamo/${models}.py --${precision} -d${device} --output=${models}_${backend}_logs/${backend}_${models}_${precision}_${device}_performance.csv --output-directory=./ --backend=${backend}   --no-skip --dashboard -x Reformer -x GPTNeoForCausalLM -x GPTJForQuestionAnswering -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForSequenceClassification --batch_size 1 "$@"
#PJRT_DEVICE=GPU python benchmarks/dynamo/${models}.py --performance --${precision} -d${device} --output=${models}_${backend}_logs/${backend}_${models}_${precision}_${device}_performance.csv --output-directory=./ --backend=${backend}  --trace-on-xla --no-skip --dashboard -x Reformer -x GPTNeoForCausalLM -x GPTJForQuestionAnswering -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForSequenceClassification --batch_size 1 "$@"
GPU_NUM_DEVICES=1 python benchmarks/dynamo/${models}.py --${precision} -d${device} --backend=${backend}  --output-directory=./ --no-skip --dashboard --batch_size 1 --trace-on-xla "$@"
#GPU_NUM_DEVICES=1 python benchmarks/dynamo/${models}.py --${precision} -d${device} --output-directory=./ --no-skip --dashboard --batch_size 1 "$@"
