from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from datasets import load_dataset
import torch
import time

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate
print("sampling_rate : %d" %(sampling_rate))

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")

raw = [d["array"] for d in dataset[10:11]["audio"]]
duration = float(raw[0].shape[0]) / float(sampling_rate)
for i in range(10) : 
    start = time.time()
    # audio file is decoded on the fly
    
    inputs = feature_extractor(
        raw, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )
    print(raw[0].shape)
    with torch.no_grad():
        embeddings = model(**inputs).embeddings

    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    end = time.time()
    t = (end - start)
    rtf = t / duration
    print("執行時間：%0.3f 秒" % (t))
    print("RTF     ：%0.3f " % (rtf))

# audio file is decoded on the fly
raw = [d["array"] for d in dataset[:2]["audio"]]
inputs = feature_extractor(
    [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
)
with torch.no_grad():
    embeddings = model(**inputs).embeddings

print(embeddings.shape)

# the resulting embeddings can be used for cosine similarity-based retrieval
cosine_sim = torch.nn.CosineSimilarity(dim=-1)
similarity = cosine_sim(embeddings[0], embeddings[1])
threshold = 0.7  # the optimal threshold is dataset-dependent
if similarity < threshold:
    print("Speakers are not the same!")
else : 
    print(round(similarity.item(), 2))