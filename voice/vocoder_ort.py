"""vocoder_ort.py — вокодер через ONNX Runtime: int8-спина, джит-голова.

Замеры на Tanix W2 (4×Cortex-A35, 2 потока): фраза 2 с речи — джит-вокодер
целиком 1.15 с, из них спина (backbone, свёртки + MatMul) ~1.0 с. Та же спина
в ONNX Runtime с динамическим int8-квантованием MatMul (веса s8 per-channel,
активации u8 на лету; свёртки и LayerNorm остаются fp32 — int8-Conv на A35
медленнее fp32) занимает 0.55 с. Голова (Linear 512→2402 + ISTFT) дешёвая
(163 мс на той же фразе) и остаётся родным джитом: head(feat) совпадает с
voc() до нуля, переписывать её незачем. Качество: CER через VOSK 0.000,
спектральная огибающая +9 дБ к статическому int8 — пробники и подбор режима
в проекте silero-int8.

Модель vocoder_dyn.onnx лежит рядом с весами голоса и запекается в образ
вместе с ними. Нет файла или onnxruntime — голос просто остаётся на джите.
"""
import torch


class VocoderORT:
    """Подмена tm.vocoder: callable(mel, sr, coef, inference) -> wav (1, N).

    Сигнатура повторяет джит-вокодер, чтобы synth() не различал пути. sr/coef/
    inference в сервисе константы (48 кГц, без Denoiser); PQMF-ветки 24/8 кГц
    из vocoder.forward тут не нужны.
    """

    def __init__(self, path, tm, threads):
        import onnxruntime as ort    # импорт здесь: образ может быть и без ORT
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads   # столько же потоков, сколько джиту
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, ["CPUExecutionProvider"])
        self.head = tm.vocoder.head

    def __call__(self, mel, sr, coef=0.0, inference=True):    # noqa: ARG002
        feat = self.sess.run(None, {"mel": mel.numpy()})[0]
        return self.head(torch.from_numpy(feat))
