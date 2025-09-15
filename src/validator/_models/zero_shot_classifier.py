import os
from typing import Any
import numpy as np

# ONNX 모델 처리와 Transformer 토크나이저를 위한 import
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer
from .classifier import Classifier


class ZeroShotClassifier (Classifier):
    # 토큰화할 때 사용할 최대 길이
    max_length: int = 512
    
    # NLI(Natural Language Inference) 모델의 분류 라벨들
    # entailment: 함의(가설이 전제에서 참), neutral: 중립, contradiction: 모순
    label_names = ["entailment", "neutral", "contradiction"]


    def __init__(self, model_name: str):
        # 환경변수에서 모델 경로를 가져오거나 현재 디렉토리 사용
        model_path = os.getenv("MODEL_PATH", ".") + "/" + model_name
        
        # 모델 파일 존재 여부와 유효성 검사
        self._validate_model(model_path)
        
        # 텍스트 토큰화를 위한 토크나이저 로드
        self.tokenizer = self._load_tokenizer(model_name)
        
        # ONNX 런타임 세션 생성 (실제 추론 실행)
        self.session = self._load_session(model_path)

    def evaluate(self, sequence: str|dict[str, Any]) -> bool:
        print(f"Sequence: {sequence}")
        # 빈 입력 검사
        if not sequence:
            raise ValueError("Sequence is empty")

        # 딕셔너리 형태의 메타데이터 처리 (예: {title: "...", description: "..."})
        if isinstance(sequence, dict):
            # 각 값에 대해 개별적으로 분류 수행
            for key, value in sequence.items():
                sequence[key] = self._classify(str(value))
            
            # 모든 키의 결과를 평균내어 최종 예측값 계산
            pred_dict = {
                "entailment": 0.0,
                "neutral": 0.0,
                "contradiction": 0.0,
            }
            for key, value in sequence.items():
                pred_dict["entailment"] += value["entailment"]
                pred_dict["neutral"] += value["neutral"]
                pred_dict["contradiction"] += value["contradiction"]
            
            # 평균 계산 (0으로 나누기 방지)
            pred_dict["entailment"] /= (len(sequence) if len(sequence) > 0 else 1)
            pred_dict["neutral"] /= (len(sequence) if len(sequence) > 0 else 1)
            pred_dict["contradiction"] /= (len(sequence) if len(sequence) > 0 else 1)
        else:
            # 단일 문자열인 경우 직접 분류
            pred_dict = self._classify(sequence)
        
        # entailment 점수가 neutral 점수보다 높으면 레시피 관련으로 판단
        if not (pred_dict["entailment"] > pred_dict["neutral"]) or not (pred_dict["entailment"] > pred_dict["contradiction"]):
            raise ValueError(
                f"Sequence is not a recipe. Classification scores: "
                f"entailment={pred_dict['entailment']:.4f}, "
                f"neutral={pred_dict['neutral']:.4f}, "
                f"contradiction={pred_dict['contradiction']:.4f}"
            )
        return True

    def _classify(self, sequence: str) -> dict[str, float]:
        # 레시피 분류를 위한 가설 문장 (한국어)
        hypothesis = "이 문장은 레시피에 관한 것이다."
        
        pred_dict = {
            "entailment": 0.0,
            "neutral": 0.0,
            "contradiction": 0.0,
        }
        
        # 문자열을 청크로 나누어 분류
        chunks = self._get_chunks(sequence)
        for chunk in chunks:
            # 전제(sequence)와 가설(hypothesis)을 함께 토큰화
            # NLI 모델은 "전제 [SEP] 가설" 형태로 입력받음
            encoded = self.tokenizer(chunk, hypothesis, return_tensors="np", truncation=True, padding="max_length", max_length=self.max_length)

            # ONNX 모델 입력 형태로 변환 (int64 타입 필요)
            onnx_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }

            # 모델 추론 실행
            outputs = self.session.run(["logits"], onnx_inputs)
            
            # 로짓 값을 소프트맥스로 확률로 변환
            probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))  # softmax
            
            pred_dict["entailment"] += float(probs[0])
            pred_dict["neutral"] += float(probs[1])
            pred_dict["contradiction"] += float(probs[2])

        # 평균 계산 (0으로 나누기 방지)
        pred_dict["entailment"] /= (len(chunks) if len(chunks) > 0 else 1)
        pred_dict["neutral"] /= (len(chunks) if len(chunks) > 0 else 1)
        pred_dict["contradiction"] /= (len(chunks) if len(chunks) > 0 else 1)

        return pred_dict

    def _get_chunks(self, sequence: str) -> list[str]:
        max_string_length = self.max_length * 2 # 문자열 길이를 일반적인 토큰 대비 2배 길이로 설정
        return [sequence[i:i+max_string_length] for i in range(0, len(sequence), max_string_length)]

    def _load_tokenizer(self, model_name: str):
        # Hugging Face에서 사전 훈련된 토크나이저 로드
        return AutoTokenizer.from_pretrained(model_name)

    def _load_session(self, model_name: str):
        # ONNX 런타임 세션 생성 (GPU 사용 가능하면 자동으로 사용)
        return ort.InferenceSession(f"{model_name}.onnx")

    def _validate_model(self, model_name: str):
        # 모델 파일 존재 여부 확인
        if not os.path.exists(f"{model_name}.onnx"):
            raise FileNotFoundError(f"Model file {model_name}.onnx not found")
        
        # ONNX 모델의 구조적 유효성 검사
        onnx.checker.check_model(onnx.load(f"{model_name}.onnx"))