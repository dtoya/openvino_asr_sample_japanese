import torch
import librosa
from transformers import Wav2Vec2Processor

from openvino.runtime import Core
import numpy as np

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
#model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
core = Core()
model = core.read_model('public/wav2vec2-large-xlsr-53-japanese/FP32/wav2vec2-large-xlsr-53-japanese.xml')
compiled_model = core.compile_model(model, 'CPU')
infer_request = compiled_model.create_infer_request()
input_tensor_name = model.inputs[0].get_any_name()
output_tensor = compiled_model.outputs[0]

speech_array, sampling_rate = librosa.load('test.wav', sr=16_000)

inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
input_data = {input_tensor_name: inputs.input_values}

with torch.no_grad():
    #logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    logits = infer_request.infer(input_data)[output_tensor]

#predicted_ids = torch.argmax(logits, dim=-1)
predicted_ids = np.squeeze(np.argmax(logits, -1))
predicted_token = processor.batch_decode(predicted_ids)
pad_token = '<pad>'
predicted_token = [t for t in predicted_token if t != pad_token]
sentence = ''.join(predicted_token)
print("Prediction: ",sentence)

