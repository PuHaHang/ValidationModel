import os

import onnx
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

from .classifier import Classifier

# 레시피/요리 관련으로 분류하고 싶은 타겟 라벨들
target_labels = [
    "recipe",
    "cooking",
    "drink",
    "food",
]

# 레시피와 관련없는 기타 라벨들 (분류 성능 비교용)
other_labels = [
    "movie",
    "music",
    "sports",
    "news",
    "technology",
    "politics",
    "science",
    "health",
    "business",
    "entertainment",
    "lifestyle",
    "travel",
    "fashion",
    "art",
    "books",
    "tv",
    "game",
    "animal",
    "plant",
    "weather",
    "other",
]


class BertBaseClassifier (Classifier):
    max_length: int = 128

    def __init__(self, model_name: str):
        # 환경변수에서 모델 경로를 가져오거나 현재 디렉토리 사용
        model_path = os.getenv("MODEL_PATH", ".") + "/" + model_name
        
        # 모델 파일 존재 여부와 유효성 검사
        self._validate_model(model_path)
        
        # BERT 기반 토크나이저 로드
        self.tokenizer = self._load_tokenizer(model_name)
        
        # ONNX 런타임 세션 생성
        self.session = self._load_session(model_path)
    
    def evaluate(self, sequence: str) -> bool:
        # 입력 텍스트를 분류하여 예측 결과 획득
        pred = self._classify(sequence)
        print(pred)  # 디버깅용 출력
        
        # 레시피 점수가 0.5보다 높으면 레시피 관련으로 판단
        return pred["recipe"] > 0.5

    def _classify(self, sequence: str) -> dict[str, float]:
        # 입력 텍스트를 BERT 토큰으로 변환 (최대 128토큰)
        inputs = self.tokenizer(sequence, return_tensors="np", padding="max_length", truncation=True, max_length=128)
        
        # ONNX 모델 입력 형태로 변환
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        
        # BERT 모델로 분류 수행
        outputs = self.session.run(["logits"], onnx_inputs)
        
        # 첫 번째 배치의 첫 번째 결과 반환 (딕셔너리 형태로 가정)
        return outputs[0][0]

    def _load_tokenizer(self, model_name: str):
        # Hugging Face에서 사전 훈련된 BERT 토크나이저 로드
        return AutoTokenizer.from_pretrained(model_name)

    def _load_session(self, model_name: str):
        # ONNX 런타임 세션 생성 (BERT 모델 추론용)
        return ort.InferenceSession(f"{model_name}.onnx")

    def _validate_model(self, model_name: str):
        # 모델 파일 존재 여부 확인
        if not os.path.exists(f"{model_name}.onnx"):
            raise FileNotFoundError(f"Model file {model_name}.onnx not found")
        
        # ONNX 모델의 구조적 유효성 검사
        onnx.checker.check_model(onnx.load(f"{model_name}.onnx"))