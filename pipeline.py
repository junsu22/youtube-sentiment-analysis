from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# =============================================================================
# 유튜브 댓글 감성분석 파이프라인
# - CNN / LSTM / Bi-LSTM 세 모델을 학습하고 성능을 비교합니다.
#
# 필요 파일:
#   - comments.csv                                     : 수집한 원본 댓글
#   - comments_labeled.csv                             : 수동 라벨링 데이터
#   - auto_full_results/comments_labeled_auto_full.csv : 자동 확장 라벨 (있으면 우선 사용)
# =============================================================================

# 결과 파일 저장 경로
ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

# 랜덤 시드 고정 (재현성 보장)
RANDOM_SEED = 42

# 데이터프레임 컬럼명
TEXT_COLUMN  = "clean_text"   # 전처리된 텍스트
LABEL_COLUMN = "label"        # 감성 레이블 (0=부정, 1=긍정)
TOKEN_COLUMN = "tokens"       # 형태소 분석 결과

# 라벨 파일 경로 — 자동 확장 라벨 파일이 있으면 우선 사용, 없으면 수동 라벨 사용
AUTO_LABEL_PATH   = ROOT / "auto_full_results" / "comments_labeled_auto_full.csv"
MANUAL_LABEL_PATH = ROOT / "comments_labeled.csv"
COMMENTS_PATH     = ROOT / "comments.csv"

USE_AUTO_LABELS_IF_AVAILABLE = True  # False 로 바꾸면 항상 수동 라벨 사용

# 데이터셋 분할 비율 (train 70% / valid 15% / test 15%)
TEST_SIZE  = 0.15
VALID_SIZE = 0.15

# 모델 하이퍼파라미터
VOCAB_SIZE    = 5000   # 사용할 최대 어휘 수
EMBED_DIM     = 64     # 임베딩 벡터 차원
LEARNING_RATE = 1e-3   # Adam 옵티마이저 학습률
BATCH_SIZE    = 32     # 미니배치 크기
EPOCHS        = 12     # 최대 학습 에폭 수 (EarlyStopping으로 조기 종료 가능)

# 형태소 분석 시 제거할 불용어 목록
STOPWORDS = {
    "하다",
    "되다",
    "있다",
    "없다",
    "같다",
    "보다",
    "이다",
    "아니다",
    "그렇다",
    "이렇다",
    "저렇다",
    "진짜",
    "너무",
    "정말",
    "그냥",
    "약간",
    "댓글",
    "영상",
    "유튜브",
}


# HTML 태그, URL, 멘션, 특수문자를 제거하고 공백을 정규화합니다.
def preprocess_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)                           # HTML 태그 제거
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)        # URL 제거
    text = re.sub(r"@[가-힣a-zA-Z0-9_]+", " ", text)            # @멘션 제거
    text = re.sub(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s]", " ", text)  # 한글·영문·숫자 외 제거
    text = re.sub(r"\s+", " ", text).strip()                     # 연속 공백 정규화
    return text


# 한글이 없는 댓글(영어만, 빈 댓글 등)은 분석 대상에서 제외합니다.
def contains_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", str(text)))


# comments.csv를 읽어 전처리 후 결과를 CSV로 저장합니다.
def load_and_clean_comments() -> pd.DataFrame:
    comments = pd.read_csv(COMMENTS_PATH, encoding="utf-8-sig")
    comments["clean_text"] = comments["text"].fillna("").astype(str).apply(preprocess_text)
    comments = comments[comments["clean_text"].str.strip() != ""].copy()        # 빈 텍스트 제거
    comments = comments[comments["clean_text"].apply(contains_korean)].reset_index(drop=True)  # 한국어 필터
    comments.to_csv(RESULT_DIR / "comments_cleaned_for_team.csv", index=False, encoding="utf-8-sig")
    return comments


# 자동 확장 라벨이 있으면 우선 사용하고, 유효하지 않은 라벨(0·1 외)은 제거합니다.
def load_labeled_data() -> pd.DataFrame:
    if USE_AUTO_LABELS_IF_AVAILABLE and AUTO_LABEL_PATH.exists():
        labeled = pd.read_csv(AUTO_LABEL_PATH, encoding="utf-8-sig")
        print(f"[INFO] 자동 확장 라벨 사용: {AUTO_LABEL_PATH}")
    else:
        labeled = pd.read_csv(MANUAL_LABEL_PATH, encoding="utf-8-sig")
        print(f"[INFO] 수동 라벨 사용: {MANUAL_LABEL_PATH}")

    labeled[LABEL_COLUMN] = pd.to_numeric(labeled[LABEL_COLUMN], errors="coerce")
    labeled = labeled[labeled[LABEL_COLUMN].isin([0, 1])].copy()   # 0·1 외 이상값 제거
    labeled[LABEL_COLUMN] = labeled[LABEL_COLUMN].astype(int)
    labeled[TEXT_COLUMN] = labeled[TEXT_COLUMN].fillna("").astype(str)
    labeled = labeled[labeled[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)
    return labeled


# Okt로 명사·형용사·동사만 추출합니다. 불용어 및 1글자 토큰은 제외합니다.
def tokenize_texts(df: pd.DataFrame) -> pd.DataFrame:
    okt = Okt()
    allowed_pos = {"Noun", "Adjective", "Verb"}   # 사용할 품사

    def tokenize_one(text: str) -> str:
        tokens = []
        for token, pos in okt.pos(text, norm=True, stem=True):  # 정규화 + 어간 추출
            if pos in allowed_pos and len(token) > 1 and token not in STOPWORDS:
                tokens.append(token)
        return " ".join(tokens)

    tokenized = df.copy()
    tokenized[TOKEN_COLUMN] = tokenized[TEXT_COLUMN].apply(tokenize_one)
    tokenized = tokenized[tokenized[TOKEN_COLUMN].str.strip() != ""].reset_index(drop=True)
    tokenized.to_csv(RESULT_DIR / "comments_tokenized_for_team.csv", index=False, encoding="utf-8-sig")
    return tokenized


# 토큰을 정수 인덱스로 변환하고 최대 길이에 맞게 패딩합니다.
# Tokenizer와 max_length는 파일로 저장해 추론 시 재사용합니다.
def build_sequences(df: pd.DataFrame):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df[TOKEN_COLUMN].tolist())
    sequences = tokenizer.texts_to_sequences(df[TOKEN_COLUMN].tolist())
    max_length = max(len(seq) for seq in sequences)
    padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")  # 뒤쪽 패딩
    labels = df[LABEL_COLUMN].to_numpy()

    with open(RESULT_DIR / "tokenizer.pkl", "wb") as fp:
        pickle.dump(tokenizer, fp)
    (RESULT_DIR / "max_sequence_length.txt").write_text(str(max_length), encoding="utf-8")

    return padded, labels, tokenizer, max_length


# 클래스 비율을 유지하는 Stratified Split으로 train / valid / test 를 나눕니다.
def split_dataset(X, y):
    # 1차 분할: train vs (valid + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=TEST_SIZE + VALID_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # 2차 분할: valid vs test
    valid_ratio = VALID_SIZE / (TEST_SIZE + VALID_SIZE)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - valid_ratio,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# Conv1D로 지역적 n-gram 패턴을 포착한 후 GlobalMaxPooling으로 중요한 특징을 집계합니다.
def build_cnn(vocab_size: int, max_length: int) -> Sequential:
    model = Sequential(
        [
            Embedding(vocab_size, EMBED_DIM, input_length=max_length),  # 단어 임베딩
            Conv1D(128, 5, activation="relu"),                           # 5-gram 필터 128개
            GlobalMaxPooling1D(),                                        # 가장 중요한 특징 추출
            Dense(64, activation="relu"),
            Dropout(0.3),                                                # 과적합 방지
            Dense(1, activation="sigmoid"),                              # 이진 분류 출력
        ]
    )
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# 순방향으로 시퀀스를 처리하며 장거리 의존성을 학습합니다.
def build_lstm(vocab_size: int, max_length: int) -> Sequential:
    model = Sequential(
        [
            Embedding(vocab_size, EMBED_DIM, input_length=max_length),
            LSTM(64),                       # 순방향 LSTM
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# 순방향 + 역방향으로 문맥을 동시에 학습합니다. 출력 차원 = LSTM(64) × 2 = 128
def build_bilstm(vocab_size: int, max_length: int) -> Sequential:
    model = Sequential(
        [
            Embedding(vocab_size, EMBED_DIM, input_length=max_length),
            Bidirectional(LSTM(64)),        # 양방향 LSTM
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# EarlyStopping으로 val_loss 기준 조기 종료하고, classification report와 모델을 저장합니다.
def train_and_evaluate(model_name: str, model: Sequential, X_train, y_train, X_valid, y_valid, X_test, y_test):
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,                   # 3 에폭 연속 개선 없으면 중단
            restore_best_weights=True,    # 최적 가중치 자동 복원
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )

    # 테스트셋 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    prob = model.predict(X_test, verbose=0).reshape(-1)
    pred = (prob >= 0.5).astype(int)      # 0.5 임계값으로 이진 분류
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)

    # classification report 저장
    report = classification_report(
        y_test, pred,
        target_names=["negative", "positive"],
        zero_division=0,
    )
    (RESULT_DIR / f"{model_name.lower()}_report.txt").write_text(report, encoding="utf-8")
    model.save(RESULT_DIR / f"{model_name.lower()}.keras")   # 모델 저장

    return {
        "model_name":     model_name,
        "test_accuracy":  float(test_accuracy),
        "test_loss":      float(test_loss),
        "precision":      float(precision),
        "recall":         float(recall),
        "f1_score":       float(f1),
        "epochs_trained": len(history.history["loss"]),
    }


def main():
    print("[1/6] 댓글 정리 중...")
    cleaned = load_and_clean_comments()

    print("[2/6] 라벨 데이터 불러오는 중...")
    labeled = load_labeled_data()

    print("[3/6] 한국어 토큰화 중...")
    tokenized = tokenize_texts(labeled)

    print("[4/6] 시퀀스 데이터 생성 중...")
    X, y, tokenizer, max_length = build_sequences(tokenized)
    vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)  # 실제 어휘 수가 더 적을 수 있음
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(X, y)

    # 분할된 데이터 저장 (재학습 시 재사용 가능)
    np.save(RESULT_DIR / "X_train.npy", X_train)
    np.save(RESULT_DIR / "X_valid.npy", X_valid)
    np.save(RESULT_DIR / "X_test.npy",  X_test)
    np.save(RESULT_DIR / "y_train.npy", y_train)
    np.save(RESULT_DIR / "y_valid.npy", y_valid)
    np.save(RESULT_DIR / "y_test.npy",  y_test)

    print("[5/6] 모델 학습 시작...")
    models = {
        "CNN": build_cnn(vocab_size, max_length),
        "LSTM": build_lstm(vocab_size, max_length),
        "Bi-LSTM": build_bilstm(vocab_size, max_length),
    }

    results = []
    for model_name, model in models.items():
        print(f"  - {model_name} 학습 중...")
        result = train_and_evaluate(model_name, model, X_train, y_train, X_valid, y_valid, X_test, y_test)
        results.append(result)

    # 성능 비교 결과 저장 (정확도 기준 내림차순 정렬)
    result_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    result_df.to_csv(RESULT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    print("[6/6] 완료")
    print()
    print("=== 데이터 요약 ===")
    print(f"전처리 후 댓글 수: {len(cleaned):,}")
    print(f"라벨 데이터 수: {len(labeled):,}")
    print(f"토큰화 후 데이터 수: {len(tokenized):,}")
    print(f"최대 시퀀스 길이: {max_length}")
    print(f"사용 단어 사전 크기: {vocab_size}")
    print()
    print("=== 모델 성능 ===")
    print(result_df[["model_name", "test_accuracy", "precision", "recall", "f1_score"]].round(4).to_string(index=False))
    print()
    print(f"결과 저장 폴더: {RESULT_DIR}")


if __name__ == "__main__":
    main()
