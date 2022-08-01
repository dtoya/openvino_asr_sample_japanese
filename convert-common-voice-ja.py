# $ python convert-common-voice-ja.py --dst_dir /data/dataset/openvino_accuracy_check --split test
import torch
import re
import librosa
from datasets import load_dataset, load_metric
import warnings
import os
import soundfile as sf
from tqdm import tqdm
import logging

args = []
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "'", "ʻ", "ˆ"]

    test_dataset = load_dataset("common_voice", args.lang, split=args.split)
    logger.info(test_dataset)
    for i in range(5):
        logger.info('{}: sentence={} path={}'.format(i, test_dataset["sentence"][i], test_dataset["path"][i]))

    chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"

# Preprocessing the datasets.
# We need to read the audio files as arrays
    def speech_file_to_array_fn(batch):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
        batch["speech"] = speech_array
        batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
        return batch

    test_dataset = test_dataset.map(speech_file_to_array_fn)

    dst_dataset_root='{}/common_voice/{}/{}'.format(args.dst_dir, args.lang, args.split)
    if not args.test:
        if not os.path.exists(dst_dataset_root):
            os.makedirs(dst_dataset_root)

    with open('{}/trans.txt'.format(dst_dataset_root), 'w') as f:
        for i, speech in enumerate(tqdm(test_dataset["speech"])):
            if not args.test:
                sf.write('{}/{:05d}.wav'.format(dst_dataset_root, i), speech, 16000, format="WAV")
                f.write('{:05d} {}\n'.format(i, test_dataset["sentence"][i]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to convert Common Voice dataset for OpenVINO Accuracy Checker.')
    parser.add_argument('--dst_dir', default='./output', help='Directry to store converted datasets.')
    parser.add_argument('--split', default='test', help='Name of split of Common Voice dataset.')
    parser.add_argument('--lang', default='ja', help='Language of target dataset.')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='Debug output.')
    parser.add_argument('-t', '--test', default=False, action='store_true', help='Run in test mode.')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main()

