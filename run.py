import torch, time
from WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
checkpoint = torch.load('./WavLM-Base+.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# extract the representation of last layer
wav_input_16khz = torch.randn(1,160000)
print(wav_input_16khz.shape)
for i in range(10) : 
    start = time.time()
    rep = model.extract_features(wav_input_16khz)[0]
    end = time.time()
    #print(rep.shape)
    print("執行時間：%f 秒" % (end - start))

# extract the representation of each layer
for i in range(10) :
    wav_input_16khz = torch.randn(1,160000)
    start = time.time()
    rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
#for x, _ in layer_results : 
#    print(x.shape)
